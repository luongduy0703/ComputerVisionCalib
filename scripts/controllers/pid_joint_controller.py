#!/usr/bin/env python3
"""
Outer-Loop PID Joint Controller for RL-Tuned Trajectory Tracking

This PID sits ABOVE the servo's internal PID (or Gazebo's position controller).
It generates corrected position commands to improve trajectory tracking accuracy.

Architecture:
    [RL Agent] → Kp, Ki, Kd gains → [THIS PID] → position commands → [Servo/Gazebo PID] → motion

The RL agent learns optimal PID gains per-episode. The gains are set ONCE before
a movement begins, then held constant for the entire trajectory execution.

References:
    - Autotuning PID using Actor-Critic Deep RL (2022), arXiv:2212.00013
    - Actor-critic learning based PID control for robotic manipulators (2024)
    - Cascade-trained DRL for PID gain optimization (2025)
"""

import numpy as np
from typing import Dict, Tuple, Optional


class PIDJointController:
    """
    Outer-loop PID controller for trajectory tracking (6-DOF robot arm).
    
    Generates corrected position commands from trajectory tracking error.
    Operates on top of the servo's internal PID (position interface).
    
    PID output formula:
        q_command = q_desired + Kp * e + Ki * ∫e·dt + Kd * de/dt
    
    where e = q_desired - q_actual (tracking error per joint)
    
    Gain scaling:
        RL agent outputs actions in [-1, 1], converted to actual gains via sigmoid:
        K = K_max * sigmoid(action)
    """
    
    # Gain ranges: maximum values for each PID term
    # These define the search space for the RL agent
    GAIN_RANGES = {
        'Kp': (0.0, 5.0),    # Proportional gain range (moderate for position cascade)
        'Ki': (0.0, 1.0),    # Integral gain range (small to avoid windup)
        'Kd': (0.0, 0.5),    # Derivative gain range (small for damping)
    }
    
    # Default gains (conservative for position-controlled servos)
    DEFAULT_KP = 1.0
    DEFAULT_KI = 0.0
    DEFAULT_KD = 0.05
    
    def __init__(self, n_joints: int = 6, anti_windup_limit: float = 0.5,
                 max_correction: float = 0.2):
        """
        Initialize PID controller for all joints.
        
        Args:
            n_joints: Number of joints (default: 6 for 6-DOF arm)
            anti_windup_limit: Maximum allowed integrator accumulation (radians)
            max_correction: Maximum PID correction per joint (radians).
                           Prevents runaway corrections in cascade position control.
        """
        self.n_joints = n_joints
        self.anti_windup_limit = anti_windup_limit
        self.max_correction = max_correction
        
        # PID gains (per-joint)
        self.Kp = np.ones(n_joints) * self.DEFAULT_KP
        self.Ki = np.ones(n_joints) * self.DEFAULT_KI
        self.Kd = np.ones(n_joints) * self.DEFAULT_KD
        
        # Internal state
        self.integral = np.zeros(n_joints)
        self.prev_error = np.zeros(n_joints)
        self.first_step = True
        
        # Tracking metrics (accumulated per episode)
        self.cumulative_iae = 0.0         # Integral Absolute Error
        self.cumulative_effort = 0.0       # Control effort
        self.step_count = 0
        self.error_history = []            # Per-step error norms
    
    def set_gains(self, Kp: np.ndarray, Ki: np.ndarray, Kd: np.ndarray):
        """
        Set PID gains directly.
        
        Args:
            Kp: Proportional gains [n_joints]
            Ki: Integral gains [n_joints]
            Kd: Derivative gains [n_joints]
        """
        self.Kp = np.clip(np.array(Kp, dtype=np.float64), 
                          self.GAIN_RANGES['Kp'][0], self.GAIN_RANGES['Kp'][1])
        self.Ki = np.clip(np.array(Ki, dtype=np.float64), 
                          self.GAIN_RANGES['Ki'][0], self.GAIN_RANGES['Ki'][1])
        self.Kd = np.clip(np.array(Kd, dtype=np.float64), 
                          self.GAIN_RANGES['Kd'][0], self.GAIN_RANGES['Kd'][1])
    
    def set_gains_from_normalized(self, actions_18d: np.ndarray):
        """
        Set PID gains from RL agent output (normalized [-1,1] → actual range).
        
        Uses sigmoid scaling (Paper 1 approach):
            K = K_max * sigmoid(action)
        
        This maps any real-valued action to (0, K_max), ensuring:
        - Gains are always positive
        - Smooth, differentiable mapping
        - Natural exploration near K_max/2
        
        Args:
            actions_18d: 18D array [Kp(6), Ki(6), Kd(6)] — raw RL output
        """
        actions_18d = np.array(actions_18d, dtype=np.float64)
        
        if len(actions_18d) != 3 * self.n_joints:
            raise ValueError(f"Expected {3 * self.n_joints}D action, got {len(actions_18d)}D")
        
        def sigmoid(x):
            # Numerically stable sigmoid
            return np.where(x >= 0,
                           1.0 / (1.0 + np.exp(-x)),
                           np.exp(x) / (1.0 + np.exp(x)))
        
        kp_raw = actions_18d[0:self.n_joints]
        ki_raw = actions_18d[self.n_joints:2*self.n_joints]
        kd_raw = actions_18d[2*self.n_joints:3*self.n_joints]
        
        self.Kp = sigmoid(kp_raw) * self.GAIN_RANGES['Kp'][1]
        self.Ki = sigmoid(ki_raw) * self.GAIN_RANGES['Ki'][1]
        self.Kd = sigmoid(kd_raw) * self.GAIN_RANGES['Kd'][1]
    
    def compute(self, q_desired: np.ndarray, q_actual: np.ndarray, 
                dt: float = 0.01) -> np.ndarray:
        """
        Compute corrected position command using PID control.
        
        The output is a POSITION command (not effort):
            q_command = q_desired + correction
        
        where correction = Kp * e + Ki * ∫e·dt + Kd * de/dt
        
        Args:
            q_desired: Desired joint positions [n_joints] (from trajectory planner)
            q_actual: Current joint positions [n_joints] (from joint_states)
            dt: Time step (seconds)
        
        Returns:
            q_command: Corrected position command [n_joints]
        """
        q_desired = np.array(q_desired, dtype=np.float64)
        q_actual = np.array(q_actual, dtype=np.float64)
        
        # Tracking error
        error = q_desired - q_actual
        
        # Proportional term
        p_term = self.Kp * error
        
        # Integral term (with anti-windup clamping)
        self.integral += error * dt
        self.integral = np.clip(self.integral, 
                                -self.anti_windup_limit, 
                                self.anti_windup_limit)
        i_term = self.Ki * self.integral
        
        # Derivative term (skip on first step to avoid spike)
        if self.first_step:
            d_term = np.zeros(self.n_joints)
            self.first_step = False
        else:
            derivative = (error - self.prev_error) / max(dt, 1e-6)
            d_term = self.Kd * derivative
        
        self.prev_error = error.copy()
        
        # PID correction (clamped to prevent instability in cascade control)
        correction = p_term + i_term + d_term
        correction = np.clip(correction, -self.max_correction, self.max_correction)
        
        # Output: corrected position command
        q_command = q_desired + correction
        
        # Update tracking metrics
        self.step_count += 1
        step_iae = np.sum(np.abs(error))
        step_effort = np.sum(correction ** 2)
        self.cumulative_iae += step_iae
        self.cumulative_effort += step_effort
        self.error_history.append(np.linalg.norm(error))
        
        return q_command
    
    def reset(self):
        """Reset internal state for a new episode/movement."""
        self.integral = np.zeros(self.n_joints)
        self.prev_error = np.zeros(self.n_joints)
        self.first_step = True
        self.cumulative_iae = 0.0
        self.cumulative_effort = 0.0
        self.step_count = 0
        self.error_history = []
    
    def get_gains_dict(self) -> Dict[str, np.ndarray]:
        """Return current gains as a dictionary (for logging/saving)."""
        return {
            'Kp': self.Kp.copy(),
            'Ki': self.Ki.copy(),
            'Kd': self.Kd.copy(),
        }
    
    def get_gains_flat(self) -> np.ndarray:
        """Return current gains as flat 18D array [Kp(6), Ki(6), Kd(6)]."""
        return np.concatenate([self.Kp, self.Ki, self.Kd])
    
    def get_episode_metrics(self) -> Dict[str, float]:
        """
        Return tracking metrics for the completed episode.
        
        Returns:
            Dict with:
                - iae: Integral Absolute Error (total across all joints/steps)
                - effort: Total control effort
                - mean_error: Average per-step error norm
                - max_error: Peak error norm
                - steps: Number of control steps executed
        """
        return {
            'iae': self.cumulative_iae,
            'effort': self.cumulative_effort,
            'mean_error': np.mean(self.error_history) if self.error_history else 0.0,
            'max_error': np.max(self.error_history) if self.error_history else 0.0,
            'steps': self.step_count,
        }
    
    def __repr__(self):
        return (f"PIDJointController(n_joints={self.n_joints}, "
                f"Kp=[{', '.join(f'{k:.1f}' for k in self.Kp)}], "
                f"Ki=[{', '.join(f'{k:.2f}' for k in self.Ki)}], "
                f"Kd=[{', '.join(f'{k:.2f}' for k in self.Kd)}])")


# =============================================================================
# UNIT TEST
# =============================================================================

def test_pid():
    """Basic unit test: step response with known gains."""
    print("=" * 60)
    print("🧪 PID Joint Controller Unit Test")
    print("=" * 60)
    
    pid = PIDJointController(n_joints=2, anti_windup_limit=0.5, max_correction=0.2)
    pid.set_gains(
        Kp=np.array([2.0, 2.0]),
        Ki=np.array([0.5, 0.5]),
        Kd=np.array([0.1, 0.1])
    )
    
    # Simulate step response: desired = [1.0, 0.5], actual starts at [0, 0]
    q_desired = np.array([1.0, 0.5])
    q_actual = np.array([0.0, 0.0])
    dt = 0.01
    
    print(f"\n📊 Step response test:")
    print(f"   Desired: {q_desired}")
    print(f"   Initial: {q_actual}")
    print(f"   Gains: {pid}")
    print()
    
    # Simulate 100 steps:
    # Servo model: first-order response, q_actual moves toward q_command
    # with time constant ~0.05s (typical hobby servo response time)
    servo_tau = 0.05  # servo time constant
    servo_alpha = dt / (servo_tau + dt)  # discrete first-order filter coefficient
    
    for step in range(100):
        q_command = pid.compute(q_desired, q_actual, dt)
        
        # Servo model: first-order low-pass filter toward command
        q_actual = q_actual + servo_alpha * (q_command - q_actual)
        
        if step % 20 == 0:
            error = np.linalg.norm(q_desired - q_actual)
            print(f"   Step {step:3d}: actual={np.round(q_actual, 4)}, "
                  f"cmd={np.round(q_command, 4)}, error={error:.4f}")
    
    final_error = np.linalg.norm(q_desired - q_actual)
    metrics = pid.get_episode_metrics()
    
    print(f"\n   Final error: {final_error:.6f}")
    print(f"   IAE: {metrics['iae']:.4f}")
    print(f"   Steps: {metrics['steps']}")
    print(f"   Mean error: {metrics['mean_error']:.4f}")
    
    assert final_error < 0.05, f"PID failed to converge: final error = {final_error}"
    print("✅ PID step response test PASSED!")
    
    # Test normalized gain setting
    print("\n📊 Normalized gain test:")
    pid2 = PIDJointController(n_joints=6)
    actions = np.zeros(18)  # sigmoid(0) = 0.5 → Kp=2.5, Ki=0.5, Kd=0.25
    pid2.set_gains_from_normalized(actions)
    print(f"   Action=0 → {pid2}")
    
    actions_high = np.ones(18) * 5.0  # sigmoid(5) ≈ 0.993
    pid2.set_gains_from_normalized(actions_high)
    print(f"   Action=5 → {pid2}")
    
    actions_low = np.ones(18) * -5.0  # sigmoid(-5) ≈ 0.007
    pid2.set_gains_from_normalized(actions_low)
    print(f"   Action=-5 → {pid2}")
    
    print("\n✅ All PID tests PASSED!")


if __name__ == '__main__':
    test_pid()
