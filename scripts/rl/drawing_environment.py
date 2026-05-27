#!/usr/bin/env python3
"""
Drawing Environment for RL Training

Extends the base RLEnvironment to support multi-waypoint drawing tasks.
The robot must reach a sequence of waypoints to draw a shape.

Features:
- Dynamic waypoint spawning based on ArUco board detection
- Subscribes to /vision/board_pose for workspace centering
- Automatically adjusts Y_PLANE from detected board position
"""

import rclpy
from rclpy.node import Node
import numpy as np
import time
from typing import Tuple, Optional, List
from geometry_msgs.msg import Point, PoseStamped, PointStamped
from std_msgs.msg import Float32MultiArray, Bool
import tf2_ros
from rclpy.duration import Duration

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.rl_environment import RLEnvironment
from rl.board_transform import BoardTransform
from drawing.shape_generator import ShapeGenerator, Shape


class DrawingEnvironment(RLEnvironment):
    """
    RL Environment for drawing shapes by following waypoint sequences.
    """
    
    def __init__(self, 
                 max_episode_steps=300,
                 waypoint_tolerance=0.01,
                 shape_type='triangle',
                 shape_size=0.06,
                 x_plane=0.50,
                 randomize_shape=False,
                 use_dynamic_workspace=True):
        """
        Initialize Drawing Environment with dynamic ArUco-based workspace.
        
        Args:
            max_episode_steps: Max steps per episode
            waypoint_tolerance: Distance to consider waypoint reached (1cm)
            shape_type: 'triangle', 'square', 'line', or 'random_triangle'
            shape_size: Size of shape in meters (default 6cm)
            x_plane: Default X coordinate (overridden by board detection)
            randomize_shape: Whether to randomize shape position each episode
            use_dynamic_workspace: Enable ArUco board detection for workspace
        """
        super().__init__(max_episode_steps=max_episode_steps, goal_tolerance=waypoint_tolerance)
        
        self.get_logger().info("✏️ Initializing Drawing Environment...")
        
        self.waypoint_tolerance = waypoint_tolerance
        self.shape_type = shape_type
        self.shape_size = shape_size
        self.randomize_shape = randomize_shape
        self.use_dynamic_workspace = use_dynamic_workspace
        
        # Board-local transform pipeline
        self.board_detected = False
        self.board_pose: Optional[PoseStamped] = None
        # Use the x_plane parameter for the default forward distance, Y=0 (center line)
        self.dynamic_workspace_center = np.array([x_plane, 0.0, 0.35])
        
        # TF2 for board transform
        self.drawing_tf_buffer = tf2_ros.Buffer()
        self.drawing_tf_listener = tf2_ros.TransformListener(self.drawing_tf_buffer, self)
        self.board_transform = BoardTransform(self.drawing_tf_buffer)
        
        # Subscribe to board detection
        if self.use_dynamic_workspace:
            self.board_sub = self.create_subscription(
                PoseStamped, '/vision/board_pose',
                self._board_callback, 10
            )
            self.get_logger().info("📡 Subscribed to /vision/board_pose for dynamic workspace")
        
        # Shape generator now works in board-local 2D coordinates
        safe_zone_m = shape_size / 2  # half of shape size as safe zone radius
        self.shape_generator = ShapeGenerator(safe_zone_m=safe_zone_m)
        
        self.current_shape: Optional[Shape] = None
        self.waypoints: np.ndarray = np.array([])
        self.waypoint_index = 0
        self.total_waypoints = 0
        self.waypoints_reached = 0
        self.line_points: List[np.ndarray] = []
        
        # Publishers for camera overlay
        self.target_pub = self.create_publisher(Point, '/rl/current_target', 10)
        self.pen_pub = self.create_publisher(PointStamped, '/rl/pen_position', 10)
        self.shape_pub = self.create_publisher(Float32MultiArray, '/rl/shape_waypoints', 10)
        self.reset_traj_pub = self.create_publisher(Bool, '/rl/reset_trajectory', 10)
        
        # Publisher for drawing (legacy - keeping for compatibility)
        self.pen_position_pub = self.create_publisher(Point, '/drawing/pen_position', 10)
        
        # Service client for line reset
        from std_srvs.srv import Empty
        self.reset_line_client = self.create_client(Empty, '/drawing/reset_line')
        
        # Observation space: 18D (-Y workspace)
        # [joints(6), EE(3), target(3), dist(3), dist3d(1), progress(1), remaining(1)]
        self.observation_space = spaces.Box(
            low=np.array(
                [-3.14159, -1.04719, -3.14159, -3.14159, -1.57079, -3.14159] +              # joint positions
                [0.20, -0.30, 0.0] +      # EE position (+X workspace)
                [0.20, -0.30, 0.0] +      # target position (+X workspace)
                [-0.60]*3 +                   # distance components
                [0.0, 0.0, 0.0]             # dist3d, progress, remaining
            ),
            high=np.array(
                [3.14159, 1.57079, 3.14159, 3.14159, 1.57079, 3.14159] +               # joint positions
                [0.80, 0.40, 0.60] +       # EE position (+X workspace)
                [0.80, 0.40, 0.60] +       # target position (+X workspace)
                [0.60]*3 +                    # distance components
                [1.0, 1.0, 30.0]            # dist3d, progress, remaining
            ),
            dtype=np.float32
        )
        
        self.get_logger().info(f"📊 Drawing: shape={shape_type}, size={shape_size*100:.0f}cm")
        self.get_logger().info(f"📊 State: 18D (6 joints + 12 other), -Y workspace")
        if self.use_dynamic_workspace:
            self.get_logger().info("⏳ Waiting for ArUco board detection...")
        self.get_logger().info("✅ Drawing Environment ready!")
    
    def _board_callback(self, msg: PoseStamped):
        """Build board-to-base_link transform from ArUco detection.
        
        Stores the full 4x4 T_vision matrix (board->camera) and combines
        with TF2 (camera->base_link) for the complete transform pipeline.
        Board-local shapes will be transformed through this pipeline.
        """
        if self.board_detected:
            return  # Already locked
        
        # Build combined transform: board-local -> camera -> base_link
        success = self.board_transform.update_from_pose(msg)
        
        if not success:
            self.get_logger().debug("TF2 not ready for board transform")
            return
        
        self.board_detected = True
        self.board_pose = msg
        
        # Get board center in base_link for workspace bounds
        center = self.board_transform.get_board_center_base()
        self.dynamic_workspace_center = center
        
        self.get_logger().info(
            f"🔒 Board LOCKED (board->base_link transform ready)\n"
            f"   Board center at base_link: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]"
        )
    
    def wait_for_initial_detection(self, timeout_sec=10.0):
        """Wait for initial ArUco board detection before training."""
        if not self.use_dynamic_workspace:
            return True
        
        if self.board_detected:
            return True
        
        self.get_logger().info(f"⏳ Waiting for initial board detection...")
        start_time = time.time()
        
        while not self.board_detected and (time.time() - start_time) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if self.board_detected:
            self.get_logger().info(
                f"✅ Board detected at [{self.dynamic_workspace_center[0]:.3f}, "
                f"{self.dynamic_workspace_center[1]:.3f}, {self.dynamic_workspace_center[2]:.3f}]")
            return True
        else:
            self.get_logger().warn("⚠️  No board detected! Using default workspace.")
            return False
    
    def _generate_shape(self) -> Shape:
        """Generate shape in board-local coords, then transform to base_link."""
        from drawing.drawing_config import POINTS_PER_EDGE
        
        # 1. Generate shape in board-local 2D coords [x, y, 0, 1]
        if self.shape_type == 'triangle':
            shape = self.shape_generator.equilateral_triangle(
                size=self.shape_size, 
                points_per_edge=POINTS_PER_EDGE
            )
        elif self.shape_type == 'dense_triangle':
            shape = self.shape_generator.dense_triangle(size=self.shape_size, points_per_edge=10)
        elif self.shape_type == 'square':
            shape = self.shape_generator.square(size=self.shape_size)
        elif self.shape_type == 'line':
            shape = self.shape_generator.line(length=self.shape_size)
        elif self.shape_type == 'random_triangle':
            shape = self.shape_generator.random_triangle(min_size=0.05, max_size=self.shape_size)
        else:
            self.get_logger().warn(f"Unknown shape type {self.shape_type}, falling back to triangle")
            shape = self.shape_generator.equilateral_triangle(
                size=self.shape_size,
                points_per_edge=POINTS_PER_EDGE
            )
        
        # 2. Transform waypoints: board-local -> base_link
        if self.board_transform.locked:
            base_pts = self.board_transform.board_to_base(shape.waypoints)
            shape.waypoints = base_pts  # Now (N, 3) in base_link frame
            self.get_logger().info(
                f"📐 Shape '{shape.name}' transformed to base_link "
                f"(center: [{base_pts.mean(axis=0)[0]:.3f}, {base_pts.mean(axis=0)[1]:.3f}, {base_pts.mean(axis=0)[2]:.3f}])"
            )
        else:
            self.get_logger().warn("⚠️ Board transform not ready — using raw board-local coords")
        
        return shape
    
    def get_state(self) -> Optional[np.ndarray]:
        """Get current state including waypoint progress."""
        if not self.data_ready:
            return None
        
        try:
            if self.waypoint_index < len(self.waypoints):
                target = self.waypoints[self.waypoint_index]
                self.target_x, self.target_y, self.target_z = target[0], target[1], target[2]
            
            dist_x = self.target_x - self.robot_x
            dist_y = self.target_y - self.robot_y
            dist_z = self.target_z - self.robot_z
            dist_3d = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
            
            progress = self.waypoint_index / max(1, self.total_waypoints)
            remaining = float(self.total_waypoints - self.waypoint_index)
            
            # 18D state vector (no velocities)
            state = np.array([
                # Joint positions (6)
                *self.joint_positions,
                # End-effector position (3)
                self.robot_x, self.robot_y, self.robot_z,
                # Target waypoint position (3)
                self.target_x, self.target_y, self.target_z,
                # Distance vector (3)
                dist_x, dist_y, dist_z,
                # Distance, progress, remaining (3)
                dist_3d, progress, remaining
            ], dtype=np.float32)  # Total: 18D
            
            return state
        except Exception as e:
            self.get_logger().error(f"State error: {e}")
            return None
    
    def reset_environment(self) -> Optional[np.ndarray]:
        """Reset for new drawing episode."""
        self.get_logger().info("🔄 Resetting Drawing Environment...")
        self.current_step = 0
        
        self.current_shape = self._generate_shape()
        # Waypoints are now (N, 3) in base_link frame
        self.waypoints = self.current_shape.waypoints
        self.total_waypoints = len(self.waypoints)
        self.waypoint_index = 0
        self.waypoints_reached = 0
        self.line_points = []
        
        self.get_logger().info(f"   Shape: {self.current_shape.name} ({self.total_waypoints} waypoints)")
        
        # Reset line visualization
        self._reset_line_visualization()
        
        # Move to home
        self._move_to_joint_positions(np.zeros(6), duration=2.0)
        time.sleep(0.5)
        
        # Set first waypoint and publish
        if len(self.waypoints) > 0:
            wp = self.waypoints[0]
            self.target_x, self.target_y, self.target_z = wp[0], wp[1], wp[2]
            self._publish_target(wp)
            self._publish_shape()  # Publish full shape outline
        
        time.sleep(0.2)
        self.get_logger().info(f"✅ Drawing reset! Shape: {self.current_shape.name}")
        return self.get_state()
    
    def _publish_target(self, target):
        """Publish target for camera overlay."""
        msg = Point(x=float(target[0]), y=float(target[1]), z=float(target[2]))
        self.target_pub.publish(msg)
    
    def _publish_shape(self):
        """Publish all shape waypoints as flat array for camera overlay."""
        if self.waypoints is None or len(self.waypoints) == 0:
            return
        msg = Float32MultiArray()
        msg.data = self.waypoints.flatten().tolist()  # [x0,y0,z0, x1,y1,z1, ...]
        self.shape_pub.publish(msg)
        self.get_logger().info(f"📐 Published shape outline ({len(self.waypoints)} waypoints)")
    
    def _reset_line_visualization(self):
        """Reset line visualization for new episode."""
        # Reset camera overlay trajectory
        self.reset_traj_pub.publish(Bool(data=True))
        
        # Legacy Gazebo line reset
        if self.reset_line_client.wait_for_service(timeout_sec=0.5):
            from std_srvs.srv import Empty
            self.reset_line_client.call_async(Empty.Request())
    
    def _publish_pen_position(self):
        """Publish pen position for camera overlay and line visualization."""
        # Legacy topic
        msg = Point(x=self.robot_x, y=self.robot_y, z=self.robot_z)
        self.pen_position_pub.publish(msg)
        
        # Camera overlay topic
        pen_msg = PointStamped()
        pen_msg.header.stamp = self.get_clock().now().to_msg()
        pen_msg.header.frame_id = 'base_link'
        pen_msg.point = msg
        self.pen_pub.publish(pen_msg)
        
        self.line_points.append(np.array([self.robot_x, self.robot_y, self.robot_z]))
    
    def step(self, action: np.ndarray) -> Tuple[Optional[np.ndarray], float, bool, dict]:
        """Execute one step."""
        self.current_step += 1
        
        state_before = self.get_state()
        if state_before is None:
            return None, -10.0, True, {'error': 'state_unavailable'}
        
        # 18D state: dist_3d is at index 15
        dist_before = state_before[15]
        
        target_joints = np.clip(action, self.joint_limits_low, self.joint_limits_high)
        self._move_to_joint_positions(target_joints, duration=0.8)  # Faster movement
        time.sleep(0.1)  # Reduced delay for faster line drawing
        
        self._publish_pen_position()
        
        # Continuously publish shape every step to handle late subscriptions
        # or Gazebo visualizer slow startups
        if self.current_step % 5 == 0:
            self._publish_shape()
        
        next_state = self.get_state()
        if next_state is None:
            return None, -10.0, True, {'error': 'state_unavailable'}
        
        # 18D state: dist_3d is at index 15
        dist_after = next_state[15]
        reward, done = self._calculate_drawing_reward(dist_after, dist_before)
        
        # Ground collision check
        if self.robot_z <= 0.01:
            reward = -50.0
            done = True
            self._move_to_joint_positions(np.zeros(6), duration=1.0)
            time.sleep(0.5)
        
        if self.current_step >= self.max_episode_steps:
            done = True
        
        info = {
            'distance': dist_after,
            'waypoint_index': self.waypoint_index,
            'total_waypoints': self.total_waypoints,
            'waypoints_reached': self.waypoints_reached,
            'shape_complete': self.waypoint_index >= self.total_waypoints,
            'step': self.current_step
        }
        
        return next_state, reward, done, info
    
    def _calculate_drawing_reward(self, dist_after: float, dist_before: float) -> Tuple[float, bool]:
        """
        Calculate reward with waypoint advancement.
        
        Uses SPARSE REWARD (same as reaching) per waypoint:
        - 0 when waypoint reached (success)
        - -1 when still trying (failure)
        
        This matches the successful reaching training reward structure.
        HER will help learn each waypoint efficiently.
        """
        done = False
        
        # Check if current waypoint is reached
        if dist_after < self.waypoint_tolerance:
            self.waypoints_reached += 1
            self.waypoint_index += 1
            
            if self.waypoint_index >= self.total_waypoints:
                # All waypoints reached - shape complete!
                reward = 0.0  # Sparse success
                done = True
                self.get_logger().info(f"🎨 SHAPE COMPLETE! ({self.total_waypoints} waypoints)")
            else:
                # Waypoint reached, advance to next
                reward = 0.0  # Sparse success for this waypoint
                next_wp = self.waypoints[self.waypoint_index]
                self.target_x, self.target_y, self.target_z = next_wp
                self._publish_target(next_wp)  # Update camera overlay
                self.get_logger().info(f"✓ Waypoint {self.waypoint_index}/{self.total_waypoints}")
        else:
            # Still trying to reach current waypoint
            reward = -1.0  # Sparse failure
        
        return reward, done


def main(args=None):
    rclpy.init(args=args)
    
    try:
        env = DrawingEnvironment(shape_type='triangle', shape_size=0.10)
        shape = env._generate_shape()
        print(f"\nShape: {shape.name}, {shape.num_waypoints} waypoints")
        for i, wp in enumerate(shape.waypoints):
            print(f"  P{i+1}: ({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})")
        rclpy.spin(env)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
