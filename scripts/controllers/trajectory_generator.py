#!/usr/bin/env python3
"""
Trajectory Generator for PID-Tracked Joint Movements

Produces smooth reference trajectories (sequences of q_desired(t)) for
the PID controller to track. Supports:
1. Linear interpolation (simple, for training)
2. Trapezoidal velocity profile (smoother, for deployment)
3. Multi-segment chaining (for drawing tasks)

The PID controller receives q_desired at each timestep and computes
corrected position commands to send to the servo/Gazebo.
"""

import numpy as np
from typing import List, Optional, Tuple


class TrajectoryGenerator:
    """
    Generate smooth joint-space trajectories for PID tracking.
    
    Given q_start and q_goal, produces a sequence of intermediate
    q_desired(t) waypoints at the specified control rate.
    """
    
    def __init__(self, n_joints: int = 6, dt: float = 0.01, 
                 default_duration: float = 1.0):
        """
        Initialize trajectory generator.
        
        Args:
            n_joints: Number of joints
            dt: Control timestep (seconds). 0.01 = 100Hz
            default_duration: Default movement duration (seconds)
        """
        self.n_joints = n_joints
        self.dt = dt
        self.default_duration = default_duration
    
    def linear(self, q_start: np.ndarray, q_goal: np.ndarray,
               duration: Optional[float] = None,
               n_steps: Optional[int] = None) -> List[np.ndarray]:
        """
        Linear interpolation between start and goal.
        
        Simplest trajectory — constant velocity throughout.
        Good for RL training where diverse dynamics are desired.
        
        Args:
            q_start: Starting joint positions [n_joints]
            q_goal: Goal joint positions [n_joints]
            duration: Movement duration (seconds). Overridden by n_steps.
            n_steps: Number of interpolation steps. If None, computed from duration.
        
        Returns:
            List of q_desired arrays, length = n_steps
        """
        q_start = np.array(q_start, dtype=np.float64)
        q_goal = np.array(q_goal, dtype=np.float64)
        
        if n_steps is None:
            dur = duration if duration is not None else self.default_duration
            n_steps = max(int(dur / self.dt), 2)
        
        t_values = np.linspace(0.0, 1.0, n_steps)
        trajectory = []
        
        for t in t_values:
            q_desired = q_start + t * (q_goal - q_start)
            trajectory.append(q_desired.copy())
        
        return trajectory
    
    def trapezoidal(self, q_start: np.ndarray, q_goal: np.ndarray,
                    max_vel: float = 1.0, max_acc: float = 5.0) -> List[np.ndarray]:
        """
        Trapezoidal velocity profile for smoother motion.
        
        Phases: accelerate → constant velocity → decelerate
        Produces zero-jerk at start and end → smoother for real servos.
        
        Args:
            q_start: Starting joint positions [n_joints]
            q_goal: Goal joint positions [n_joints]
            max_vel: Maximum joint velocity (rad/s)
            max_acc: Maximum joint acceleration (rad/s²)
        
        Returns:
            List of q_desired arrays
        """
        q_start = np.array(q_start, dtype=np.float64)
        q_goal = np.array(q_goal, dtype=np.float64)
        
        # Distance per joint
        delta = q_goal - q_start
        max_delta = np.max(np.abs(delta))
        
        if max_delta < 1e-6:
            # Already at goal
            return [q_start.copy()]
        
        # Normalize direction
        direction = delta / max_delta
        
        # Compute trapezoidal profile for the longest-distance joint
        # Time to accelerate to max_vel
        t_acc = max_vel / max_acc
        # Distance covered during acceleration
        d_acc = 0.5 * max_acc * t_acc ** 2
        
        if 2 * d_acc >= max_delta:
            # Triangle profile (can't reach max_vel)
            t_acc = np.sqrt(max_delta / max_acc)
            t_const = 0.0
            t_total = 2 * t_acc
            actual_max_vel = max_acc * t_acc
        else:
            # Trapezoidal profile
            d_const = max_delta - 2 * d_acc
            t_const = d_const / max_vel
            t_total = 2 * t_acc + t_const
            actual_max_vel = max_vel
        
        # Generate waypoints
        trajectory = []
        t = 0.0
        
        while t <= t_total + self.dt * 0.5:
            if t < t_acc:
                # Acceleration phase
                s = 0.5 * max_acc * t ** 2
            elif t < t_acc + t_const:
                # Constant velocity phase
                s = d_acc + actual_max_vel * (t - t_acc)
            else:
                # Deceleration phase
                t_dec = t - t_acc - t_const
                s = d_acc + actual_max_vel * t_const + actual_max_vel * t_dec - 0.5 * max_acc * t_dec ** 2
            
            # Clamp progress to [0, max_delta]
            s = np.clip(s, 0.0, max_delta)
            
            q_desired = q_start + (s / max_delta) * delta
            trajectory.append(q_desired.copy())
            
            t += self.dt
        
        # Ensure final point is exactly at goal
        if len(trajectory) > 0:
            trajectory[-1] = q_goal.copy()
        
        return trajectory
    
    def multi_segment(self, waypoints: List[np.ndarray],
                      method: str = 'linear',
                      duration_per_segment: Optional[float] = None,
                      **kwargs) -> List[np.ndarray]:
        """
        Chain multiple segments for multi-waypoint trajectories (drawing).
        
        Args:
            waypoints: List of joint configurations [wp0, wp1, wp2, ...]
            method: 'linear' or 'trapezoidal'
            duration_per_segment: Duration for each segment (seconds)
            **kwargs: Additional arguments for the selected method
        
        Returns:
            Complete trajectory (list of q_desired arrays)
        """
        if len(waypoints) < 2:
            return [np.array(waypoints[0])] if waypoints else []
        
        full_trajectory = []
        
        for i in range(len(waypoints) - 1):
            q_start = np.array(waypoints[i])
            q_goal = np.array(waypoints[i + 1])
            
            if method == 'linear':
                segment = self.linear(q_start, q_goal, 
                                      duration=duration_per_segment, **kwargs)
            elif method == 'trapezoidal':
                segment = self.trapezoidal(q_start, q_goal, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Skip first point of subsequent segments (avoid duplicates)
            if i > 0 and len(segment) > 0:
                segment = segment[1:]
            
            full_trajectory.extend(segment)
        
        return full_trajectory
    
    def get_duration(self, trajectory: List[np.ndarray]) -> float:
        """Get total duration of a trajectory in seconds."""
        return len(trajectory) * self.dt
    
    def get_velocities(self, trajectory: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute joint velocities along the trajectory (numerical differentiation).
        
        Returns:
            List of velocity arrays (same length as trajectory)
        """
        velocities = []
        for i in range(len(trajectory)):
            if i == 0:
                vel = (trajectory[1] - trajectory[0]) / self.dt if len(trajectory) > 1 else np.zeros(self.n_joints)
            elif i == len(trajectory) - 1:
                vel = (trajectory[-1] - trajectory[-2]) / self.dt
            else:
                vel = (trajectory[i+1] - trajectory[i-1]) / (2 * self.dt)
            velocities.append(vel)
        return velocities


# =============================================================================
# UNIT TEST
# =============================================================================

def test_trajectory_generator():
    """Test trajectory generation methods."""
    print("=" * 60)
    print("🧪 Trajectory Generator Unit Test")
    print("=" * 60)
    
    tg = TrajectoryGenerator(n_joints=6, dt=0.01, default_duration=1.0)
    
    q_start = np.zeros(6)
    q_goal = np.array([0.5, -0.3, 0.8, 0.0, -0.5, 0.2])
    
    # Test 1: Linear interpolation
    print("\n📊 Test 1: Linear interpolation")
    traj_linear = tg.linear(q_start, q_goal, duration=1.0)
    print(f"   Steps: {len(traj_linear)}")
    print(f"   Duration: {tg.get_duration(traj_linear):.2f}s")
    print(f"   Start: {np.round(traj_linear[0], 3)}")
    print(f"   End:   {np.round(traj_linear[-1], 3)}")
    
    # Verify start and end
    assert np.allclose(traj_linear[0], q_start), "Linear: start point mismatch"
    assert np.allclose(traj_linear[-1], q_goal), "Linear: end point mismatch"
    print("   ✅ Start/end points correct")
    
    # Test 2: Trapezoidal profile
    print("\n📊 Test 2: Trapezoidal velocity profile")
    traj_trap = tg.trapezoidal(q_start, q_goal, max_vel=1.0, max_acc=5.0)
    print(f"   Steps: {len(traj_trap)}")
    print(f"   Duration: {tg.get_duration(traj_trap):.2f}s")
    print(f"   Start: {np.round(traj_trap[0], 3)}")
    print(f"   End:   {np.round(traj_trap[-1], 3)}")
    
    assert np.allclose(traj_trap[-1], q_goal), "Trapezoidal: end point mismatch"
    print("   ✅ End point correct")
    
    # Test 3: Multi-segment
    print("\n📊 Test 3: Multi-segment trajectory")
    waypoints = [
        np.zeros(6),
        np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
        np.zeros(6),  # Return to start
    ]
    traj_multi = tg.multi_segment(waypoints, method='linear', 
                                   duration_per_segment=0.5)
    print(f"   Waypoints: {len(waypoints)}")
    print(f"   Total steps: {len(traj_multi)}")
    print(f"   Total duration: {tg.get_duration(traj_multi):.2f}s")
    
    assert np.allclose(traj_multi[0], waypoints[0]), "Multi: start mismatch"
    assert np.allclose(traj_multi[-1], waypoints[-1]), "Multi: end mismatch"
    print("   ✅ Start/end points correct")
    
    # Test 4: Velocities
    print("\n📊 Test 4: Velocity computation")
    velocities = tg.get_velocities(traj_linear)
    print(f"   Velocity samples: {len(velocities)}")
    # Linear trajectory should have constant velocity
    mid_vel = velocities[len(velocities)//2]
    print(f"   Mid-point velocity: {np.round(mid_vel, 3)}")
    print("   ✅ Velocities computed")
    
    # Test 5: Zero movement
    print("\n📊 Test 5: Zero movement (start == goal)")
    traj_zero = tg.trapezoidal(q_start, q_start)
    print(f"   Steps: {len(traj_zero)}")
    assert len(traj_zero) >= 1, "Zero movement should produce at least 1 point"
    print("   ✅ Zero movement handled")
    
    print("\n✅ All trajectory generator tests PASSED!")


if __name__ == '__main__':
    test_trajectory_generator()
