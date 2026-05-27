#!/usr/bin/env python3
"""
PID Tuning RL Environment for 6-DOF Robot Arm

This environment wraps the existing RLEnvironment via COMPOSITION (not inheritance)
to provide a PID gain tuning interface. The RL agent learns optimal Kp, Ki, Kd
gains for each joint.

Target generation:
    - Generates targets in JOINT SPACE (random valid joint configurations)
    - Uses FK to compute XYZ for visualization (sphere teleport + camera overlay)
    - No Neural IK dependency — FK is exact math from URDF

Architecture (single-step MDP per episode):
    1. Reset robot to home
    2. Generate random joint target, FK → XYZ for visualization
    3. RL agent observes state (24D) and outputs PID gains (18D)
    4. PID controller tracks trajectory from current → target
    5. Reward = -tracking_error (IAE) - effort penalty
    6. Episode ends after one complete movement

References:
    - Autotuning PID using Actor-Critic Deep RL (2022), arXiv:2212.00013
    - Actor-critic learning based PID control for robotic manipulators (2024)
"""

import os
import sys
import numpy as np
import time
from typing import Tuple, Optional, Dict

import rclpy

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from controllers.pid_joint_controller import PIDJointController
from controllers.trajectory_generator import TrajectoryGenerator

# Gym spaces
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces


# =============================================================================
# CONFIGURATION
# =============================================================================

# PID trajectory parameters
TRAJECTORY_STEPS = 50        # Updated to 50 to match Raspberry Pi 50Hz PWM Hardware loop
TRAJECTORY_DT = 0.02         # 50Hz Control Timestep
TRAJECTORY_DURATION = 1.0    # Keep physical duration at 1.0 second
SETTLE_TIME = 0.3            # Time to wait after trajectory completion

# Reward weights
REWARD_ALPHA = 1.0           # Weight for IAE (tracking error)
REWARD_BETA = 0.01           # Weight for control effort
REWARD_GAMMA = 10.0          # Weight for final position error

# Joint limits (from rl_environment.py)
JOINT_LIMITS_LOW = np.array([-1.5708, -1.0472, -1.5708, -1.5708, -1.5708, -1.5708])
JOINT_LIMITS_HIGH = np.array([1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708])

# Target sampling: how much of the joint range to sample from
TARGET_RANGE_FRACTION = 0.7


# =============================================================================
# PID TUNING ENVIRONMENT
# =============================================================================

class PIDTuningEnv:
    """
    RL Environment for PID Gain Tuning (wraps RLEnvironment via composition).
    
    State Space (24D):
        - Joint positions q_actual (6)
        - Joint velocities q̇_actual (6) 
        - Target joint positions q_goal (6)
        - Tracking errors e = q_goal - q_actual (6)
    
    Action Space (18D):
        - Kp gains for 6 joints (6)
        - Ki gains for 6 joints (6)
        - Kd gains for 6 joints (6)
    
    Each episode is a single-step MDP:
        observe state → output PID gains → execute trajectory → receive reward
    """
    
    def __init__(self, base_env, n_joints: int = 6):
        """
        Initialize PID Tuning Environment.
        
        Args:
            base_env: The existing RLEnvironment instance (provides ROS2 interface)
            n_joints: Number of joints (default: 6)
        """
        self.base_env = base_env
        self.n_joints = n_joints
        
        # PID controller and trajectory generator
        self.pid = PIDJointController(n_joints=n_joints)
        self.traj_gen = TrajectoryGenerator(
            n_joints=n_joints, 
            dt=TRAJECTORY_DT,
            default_duration=TRAJECTORY_DURATION
        )
        
        # State and action spaces
        self.state_dim = 4 * n_joints  # 24D
        self.action_dim = 3 * n_joints  # 18D
        
        # Observation space: 24D
        obs_low = np.concatenate([
            np.full(n_joints, -np.pi),      # joint positions min
            np.full(n_joints, -10.0),        # joint velocities min
            JOINT_LIMITS_LOW,                 # target joints min
            np.full(n_joints, -2 * np.pi),   # tracking error min
        ])
        obs_high = np.concatenate([
            np.full(n_joints, np.pi),        # joint positions max
            np.full(n_joints, 10.0),         # joint velocities max
            JOINT_LIMITS_HIGH,                # target joints max
            np.full(n_joints, 2 * np.pi),    # tracking error max
        ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Action space: 18D in [-1, 1] (sigmoid-scaled to gain ranges)
        self.action_space = spaces.Box(
            low=np.full(self.action_dim, -1.0),
            high=np.full(self.action_dim, 1.0),
            dtype=np.float32
        )
        
        # Current episode state
        self.current_q_goal = np.zeros(n_joints)
        self.current_target_xyz = np.zeros(3)
        self.home_position = np.zeros(n_joints)
        
        # Episode counter
        self.episode_count = 0
        
        # Logging
        self.gain_history = []  # Track gain evolution
        
        self._log("PID Tuning Environment initialized")
        self._log(f"  State dim: {self.state_dim}, Action dim: {self.action_dim}")
        self._log(f"  Target gen: joint-space random → FK for visualization")
        self._log(f"  Trajectory: {TRAJECTORY_STEPS} steps, {TRAJECTORY_DURATION}s")
        self._log(f"  PID gain ranges: Kp=[0, {self.pid.GAIN_RANGES['Kp'][1]}], "
                  f"Ki=[0, {self.pid.GAIN_RANGES['Ki'][1]}], "
                  f"Kd=[0, {self.pid.GAIN_RANGES['Kd'][1]}]")
    
    def _log(self, msg: str):
        """Log via the base environment's ROS logger."""
        self.base_env.get_logger().info(f"[PID-Tune] {msg}")
    
    def _spin(self, n: int = 5, timeout: float = 0.1):
        """Spin ROS to process callbacks."""
        for _ in range(n):
            rclpy.spin_once(self.base_env, timeout_sec=timeout)
    
    def _get_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current joint positions and velocities from the base environment."""
        self._spin(3)
        q = np.array(self.base_env.joint_positions, dtype=np.float64)
        qd = np.array(self.base_env.joint_velocities, dtype=np.float64)
        return q, qd
    
    # =========================================================================
    # TARGET GENERATION (Joint-space + FK visualization)
    # =========================================================================
    
    def _generate_random_target(self) -> np.ndarray:
        """
        Generate target ON THE BOARD and use Numerical IK to find joint angles.
        
        This perfectly matches the old visual servoing logic:
        1. Call base_env._randomize_target() (handles board constraints & visualization)
        2. Read the generated XYZ position from base_env
        3. Use scipy.optimize (Numerical IK) to find exact joint angles
        
        Returns:
            q_goal: Target joint configuration [n_joints]
        """
        # 1. Use the EXACT same target generation as old RL (board constrained)
        self.base_env._randomize_target()
        
        # 2. Get the XYZ target on the board
        target_xyz = np.array([
            self.base_env.target_x,
            self.base_env.target_y,
            self.base_env.target_z
        ])
        self.current_target_xyz = target_xyz
        
        # 3. Numerical IK to convert XYZ to joint space (bypassing Neural IK)
        from scipy.optimize import minimize
        from rl.fk_ik_utils import fk
        
        def ik_loss(q):
            xyz = np.array(fk(list(q)))
            return np.sum((xyz - target_xyz)**2)
            
        bounds = list(zip(JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH))
        q_start, _ = self._get_joint_state()
        
        # Solve IK using L-BFGS-B (fast and robust local optimizer)
        res = minimize(ik_loss, q_start, bounds=bounds, method='L-BFGS-B')
        
        if res.fun > 1e-4:
            self._log(f"⚠️ Numerical IK error high ({res.fun:.2e}), target might be unreachable")
        
        q_goal = res.x
        
        self._log(f"Target Board XYZ=[{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}] "
                  f"→ Joints={np.degrees(q_goal).astype(int)}°")
        
        return q_goal
    
    # =========================================================================
    # ENVIRONMENT INTERFACE
    # =========================================================================
    
    def get_state(self) -> np.ndarray:
        """
        Build 24D state vector.
        
        Returns:
            state: [q_actual(6), q_vel(6), q_goal(6), error(6)]
        """
        q_actual, q_vel = self._get_joint_state()
        error = self.current_q_goal - q_actual
        
        state = np.concatenate([
            q_actual,               # Joint positions (6)
            q_vel,                  # Joint velocities (6)
            self.current_q_goal,    # Target joints (6)
            error,                  # Tracking error (6)
        ]).astype(np.float32)
        
        return state
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.
        
        1. Move robot to home position
        2. Generate random joint target
        3. FK → XYZ → teleport visual sphere
        4. Return initial 24D state
        
        Returns:
            Initial state observation (24D)
        """
        self.episode_count += 1
        self._log(f"=== Episode {self.episode_count} Reset ===")
        
        # Move to home position
        self._log("Moving to home position...")
        self.base_env._move_to_joint_positions(self.home_position, duration=2.0)
        time.sleep(0.5)
        self._spin(10)
        
        # Generate random target (joint-space + FK visualization)
        self.current_q_goal = self._generate_random_target()
        
        # Reset PID controller state
        self.pid.reset()
        
        # Get initial state
        state = self.get_state()
        
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one episode: set PID gains → track trajectory → return reward.
        
        This is a SINGLE-STEP MDP: one action per episode, one reward.
        The "step" encompasses an entire trajectory execution.
        
        Args:
            action: 18D PID gains in [-1, 1] (will be sigmoid-scaled)
        
        Returns:
            next_state: Final 24D state after movement
            reward: Negative tracking error (higher = better tracking)
            done: Always True (single-step episode)
            info: Dict with tracking metrics and gain values
        """
        action = np.array(action, dtype=np.float64)
        
        # 1. Set PID gains from RL agent output
        self.pid.set_gains_from_normalized(action)
        gains = self.pid.get_gains_dict()
        
        self._log(f"PID Gains: Kp={np.round(gains['Kp'], 2)}, "
                  f"Ki={np.round(gains['Ki'], 3)}, Kd={np.round(gains['Kd'], 3)}")
        
        # 2. Generate trajectory from current position to target
        q_start, _ = self._get_joint_state()
        trajectory = self.traj_gen.linear(
            q_start, self.current_q_goal, 
            n_steps=TRAJECTORY_STEPS
        )
        
        self._log(f"Tracking: {len(trajectory)} steps, "
                  f"{np.degrees(np.linalg.norm(self.current_q_goal - q_start)):.1f}° movement")
        
        # 3. Execute trajectory with PID control
        self.pid.reset()  # Clear integrator for clean tracking
        
        for i, q_desired in enumerate(trajectory):
            step_start_time = time.time()
            
            # Get current state
            q_actual, _ = self._get_joint_state()
            
            # PID computes corrected position command
            q_command = self.pid.compute(q_desired, q_actual, dt=TRAJECTORY_DT)
            
            # Clip to joint limits
            q_command = np.clip(q_command, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
            
            # Send position command via ZERO-OVERHEAD Topic Publisher
            self.base_env._stream_joint_positions(q_command, duration=TRAJECTORY_DT)
            
            # Brief spin to process feedback
            self._spin(1, timeout=0.0)
            
            # STRICT 100Hz RUNTIME ENFORCEMENT
            elapsed = time.time() - step_start_time
            if elapsed < TRAJECTORY_DT:
                time.sleep(TRAJECTORY_DT - elapsed)
        
        # 4. Wait for robot to settle
        time.sleep(SETTLE_TIME)
        self._spin(5)
        
        # 5. Get final state and compute reward
        q_final, qd_final = self._get_joint_state()
        final_error = np.linalg.norm(self.current_q_goal - q_final)
        
        # Calculate strict Cartesian Reaching Error (in mm)
        from rl.fk_ik_utils import fk
        xyz_final = np.array(fk(q_final.tolist()))
        cartesian_dist_mm = np.linalg.norm(self.current_target_xyz - xyz_final) * 1000.0
        
        # Get PID tracking metrics
        metrics = self.pid.get_episode_metrics()
        
        # Compute reward
        reward = self._compute_reward(metrics, final_error)
        
        # Build final state
        next_state = self.get_state()
        
        # Log results
        self._log(f"Result: err={np.degrees(final_error):.2f}° "
                  f"CartesianMiss={cartesian_dist_mm:.1f}mm "
                  f"IAE={metrics['iae']:.4f} R={reward:.2f}")
        
        # Store gain history
        self.gain_history.append({
            'episode': self.episode_count,
            'Kp': gains['Kp'].copy(),
            'Ki': gains['Ki'].copy(),
            'Kd': gains['Kd'].copy(),
            'iae': metrics['iae'],
            'final_error': final_error,
            'reward': reward,
            'target_xyz': self.current_target_xyz.copy(),
        })
        
        # Info dict
        info = {
            'iae': metrics['iae'],
            'effort': metrics['effort'],
            'final_error': final_error,
            'cartesian_dist_mm': cartesian_dist_mm,
            'mean_error': metrics['mean_error'],
            'max_error': metrics['max_error'],
            'gains': gains,
            'episode': self.episode_count,
            'target_xyz': self.current_target_xyz.copy(),
        }
        
        # Single-step MDP: always done after one trajectory
        done = True
        
        return next_state, reward, done, info
    
    def _compute_reward(self, metrics: Dict, final_error: float) -> float:
        """
        Compute reward from tracking metrics.
        
        reward = -α·IAE - β·effort - γ·final_error
        """
        iae = metrics['iae']
        effort = metrics['effort']
        
        reward = (
            -REWARD_ALPHA * iae
            - REWARD_BETA * effort
            - REWARD_GAMMA * final_error
        )
        
        return reward
    
    def get_gain_history(self) -> list:
        """Return the full gain history for plotting."""
        return self.gain_history
    
    def get_best_gains(self) -> Optional[Dict]:
        """Return the gains that achieved the best (highest) reward."""
        if not self.gain_history:
            return None
        
        best = max(self.gain_history, key=lambda x: x['reward'])
        return {
            'Kp': best['Kp'],
            'Ki': best['Ki'],
            'Kd': best['Kd'],
            'reward': best['reward'],
            'iae': best['iae'],
            'episode': best['episode'],
        }
