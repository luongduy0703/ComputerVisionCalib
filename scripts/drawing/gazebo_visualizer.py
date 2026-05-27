#!/usr/bin/env python3
"""
Gazebo Drawing Visualizer for Visual Servoing

Spawns drawing shapes and pen lines directly in Gazebo using Ignition Transport.
Uses ArUco-detected board coordinates for dynamic positioning.

Features:
- Spawn INVERTED triangle (apex at bottom) as waypoint spheres
- Draw pen path as green cylinders
- Subscribe to board pose for dynamic triangle center
- Reset lines on episode reset
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Float32MultiArray, Bool
from std_srvs.srv import Empty
import subprocess
import numpy as np
from typing import List, Tuple
import time
import tf2_ros
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation as R_scipy

import sys
import os

# Add scripts directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if 'install' in script_dir:
    ws_root = script_dir.split('install')[0]
    scripts_dir = os.path.join(ws_root, 'src', 'visual_servoing', 'scripts')
else:
    scripts_dir = os.path.dirname(script_dir)

if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from drawing.drawing_config import POINTS_PER_EDGE, SHAPE_SIZE


class GazeboDrawingVisualizer(Node):
    """Spawns drawing visuals in Gazebo using ArUco-detected board coordinates."""
    
    def __init__(self):
        super().__init__('gazebo_drawing_visualizer')
        
        self.get_logger().info("🎨 Gazebo Drawing Visualizer starting...")
        
        self.world_name = "visual_servoing_world"
        
        # Board tracking
        self.board_center = None  # Set by ArUco detection
        self.board_locked = False
        
        # TF2 for coordinate transform
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        from rl.board_transform import BoardTransform
        self.board_transform = BoardTransform(self.tf_buffer)
        
        # Pen path tracking
        self.pen_points: List[np.ndarray] = []
        self.last_point = None
        self.min_distance = 0.005  # 5mm between points
        self.line_segment_id = 0
        self.spawned_segments: List[str] = []
        
        # Triangle outline tracking
        self.triangle_spawned = False
        self.triangle_segments: List[str] = []
        self.last_waypoints_hash = None  # To prevent duplicate spawning
        
        # Reaching target sphere tracking
        self.reaching_target_name = 'reaching_target'
        self.reaching_target_spawned = False
        
        # Colors
        self.line_color = (0.0, 1.0, 0.0, 1.0)  # Green for pen path
        self.target_color = (1.0, 0.5, 0.0, 0.8)  # Orange for waypoints
        self.reaching_color = (1.0, 0.0, 0.0, 0.9)  # Red for reaching target
        
        # Subscribe to board pose (ArUco detection)
        self.board_sub = self.create_subscription(
            PoseStamped, '/vision/board_pose',
            self._board_callback, 10
        )
        
        # Subscribe to pen position
        self.position_sub = self.create_subscription(
            Point, '/drawing/pen_position', self.position_callback, 10
        )
        
        # Subscribe to shape waypoints (from drawing_environment)
        self.shape_sub = self.create_subscription(
            Float32MultiArray, '/rl/shape_waypoints',
            self.shape_waypoints_callback, 10
        )
        
        # Subscribe to reset signal
        self.reset_trajectory_sub = self.create_subscription(
            Bool, '/rl/reset_trajectory',
            self.reset_trajectory_callback, 10
        )
        
        # Service to reset
        self.reset_srv = self.create_service(
            Empty, '/drawing/reset_line', self.reset_callback
        )
        
        # Subscribe to reaching target (from rl_environment)
        self.target_sub = self.create_subscription(
            Point, '/rl/current_target',
            self._reaching_target_callback, 10
        )
        
        # Startup grace period — skip pen points until TF is stable
        self.startup_time = time.time()
        self.tf_ready = False  # Set True after first successful TF lookup
        self.STARTUP_GRACE_SEC = 3.0  # Skip pen points during first 3 seconds
        
        self.get_logger().info("✅ Gazebo Drawing Visualizer ready!")
        self.get_logger().info("   Board pose: /vision/board_pose")
        self.get_logger().info("   Pen path:   /drawing/pen_position")
        self.get_logger().info("   Shapes:     /rl/shape_waypoints")
        self.get_logger().info("   Target:     /rl/current_target")
    
    def _reaching_target_callback(self, msg: Point):
        """Spawn/respawn a target sphere for reaching mode."""
        position_base = np.array([msg.x, msg.y, msg.z])
        result = self._base_to_world(position_base)
        if result is None:
            return  # TF not ready
        position_world = result[0]
        
        # Delete old sphere
        if self.reaching_target_spawned:
            self._delete_entity(self.reaching_target_name)
            # Small delay to let Gazebo process deletion
            time.sleep(0.1)
        
        # Spawn new sphere (1cm radius, red)
        self._spawn_sphere(
            self.reaching_target_name, position_world,
            radius=0.01, color=self.reaching_color
        )
        self.reaching_target_spawned = True
        self.get_logger().info(
            f"🎯 Target sphere at world: ({position_world[0]:.3f}, "
            f"{position_world[1]:.3f}, {position_world[2]:.3f})"
        )
    
    def _board_callback(self, msg: PoseStamped):
        """Lock board transform pipeline from ArUco detection."""
        if self.board_locked:
            return
        
        success = self.board_transform.update_from_pose(msg)
        
        if success:
            center = self.board_transform.get_board_center_base()
            self.board_center = tuple(center)
            self.board_locked = True
            
            self.get_logger().info(
                f"🔒 Board transform locked for Gazebo Visualizer: "
                f"({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
            )
        else:
            self.get_logger().debug("TF2 not ready for Gazebo Visualizer board transform")
    
    def shape_waypoints_callback(self, msg: Float32MultiArray):
        """Spawn triangle from waypoints published by drawing_environment."""
        if len(msg.data) < 9:
            return
        
        # Check if waypoints have changed to avoid spamming Gazebo
        import hashlib
        data_hash = hashlib.md5(np.array(msg.data).tobytes()).hexdigest()
        if self.triangle_spawned and data_hash == self.last_waypoints_hash:
            return
            
        self.last_waypoints_hash = data_hash
        
        # Parse waypoints (flat array of [x,y,z, x,y,z, ...])
        n_points = len(msg.data) // 3
        waypoints_base = np.array(msg.data).reshape(n_points, 3)
        
        # Transform base_link -> world for Gazebo spawning
        waypoints_world = self._base_to_world(waypoints_base)
        
        self.get_logger().info(f"🔺 Received NEW {n_points} waypoints, spawning in Gazebo world frame...")
        # Debug: log first and last waypoint in both frames
        if n_points > 0:
            self.get_logger().info(f"   📍 WP[0] base_link: [{waypoints_base[0][0]:.4f}, {waypoints_base[0][1]:.4f}, {waypoints_base[0][2]:.4f}]")
            self.get_logger().info(f"   📍 WP[0] world:     [{waypoints_world[0][0]:.4f}, {waypoints_world[0][1]:.4f}, {waypoints_world[0][2]:.4f}]")
            center_base = waypoints_base.mean(axis=0)
            center_world = waypoints_world.mean(axis=0)
            self.get_logger().info(f"   📍 Center base_link: [{center_base[0]:.4f}, {center_base[1]:.4f}, {center_base[2]:.4f}]")
            self.get_logger().info(f"   📍 Center world:     [{center_world[0]:.4f}, {center_world[1]:.4f}, {center_world[2]:.4f}]")
            self.get_logger().info(f"   📍 Board surface expected at world Y≈-0.27, Z≈0.35")
        self._spawn_waypoint_spheres(waypoints_world)
    
    def _base_to_world(self, points_base: np.ndarray) -> np.ndarray:
        """Transform base_link points to Gazebo world coordinates.
        
        Returns None if TF is not ready (during startup grace period).
        """
        from rclpy.duration import Duration
        try:
            tf = self.tf_buffer.lookup_transform(
                'world', 'base_link',
                rclpy.time.Time(seconds=0), timeout=Duration(seconds=0.5)
            )
            t = tf.transform.translation
            r = tf.transform.rotation
            q_list = [r.x, r.y, r.z, r.w]
            R = R_scipy.from_quat(q_list).as_matrix()
            
            if not self.tf_ready:
                self.tf_ready = True
                self.get_logger().info(
                    f"   🔄 TF2 world←base_link ready: t=({t.x:.3f}, {t.y:.3f}, {t.z:.3f})"
                )
            
            pts = np.atleast_2d(points_base)
            transformed = (R @ pts.T).T + np.array([t.x, t.y, t.z])
            return transformed
        except Exception as e:
            # During startup, return None instead of fallback
            if not self.tf_ready:
                self.get_logger().debug(f"TF2 not ready yet: {e}")
                return None
            # After TF has been working, use fallback
            self.get_logger().warn(f"⚠️ TF2 lookup failed, using fallback: {e}")
            pts = np.atleast_2d(points_base).copy()
            transformed = pts.copy()
            transformed[:, 1] = -pts[:, 1]
            transformed[:, 2] = 0.5 - pts[:, 2]
            return transformed

    def _spawn_waypoint_spheres(self, waypoints: np.ndarray):
        """Spawn small spheres at each waypoint position in Gazebo."""
        # Clear old triangle
        self._delete_triangle()
        
        sphere_radius = 0.004  # 4mm radius

        for i, wp in enumerate(waypoints):
            name = f"waypoint_{i}"
            # Use green for first and last (start/end), orange for rest
            if i == 0 or i == len(waypoints) - 1:
                color = (0.0, 1.0, 0.0, 0.8)  # Green
            else:
                color = self.target_color  # Orange
            
            self._spawn_sphere(name, wp, sphere_radius, color)
            self.triangle_segments.append(name)
        
        self.triangle_spawned = True
        self.get_logger().info(f"✅ Spawned {len(waypoints)} waypoints in Gazebo")
    
    def _spawn_sphere(self, name: str, position: np.ndarray,
                      radius: float, color: Tuple[float, float, float, float]):
        """Spawn a sphere at given position."""
        sdf = f'''<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>{radius}</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>{color[0]} {color[1]} {color[2]} {color[3]}</ambient>
          <diffuse>{color[0]} {color[1]} {color[2]} {color[3]}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
        
        try:
            cmd = [
                'gz', 'service',
                '-s', f'/world/{self.world_name}/create',
                '--reqtype', 'gz.msgs.EntityFactory',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', f'sdf: "{sdf.replace(chr(10), " ").replace(chr(34), chr(92)+chr(34))}" '
                         f'pose: {{position: {{x: {position[0]}, y: {position[1]}, z: {position[2]}}}}}'
            ]
            
            # Non-blocking subprocess call
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.spawned_segments.append(name)
                
        except Exception as e:
            self.get_logger().debug(f"Spawn sphere error: {e}")
    
    def _spawn_line_segment(self, name: str, p1: np.ndarray, p2: np.ndarray, 
                            color: Tuple[float, float, float, float], 
                            radius: float = 0.001):
        """Spawn a line segment as a cylinder in Gazebo."""
        direction = p2 - p1
        length = np.linalg.norm(direction)
        
        if length < 0.001:
            return
        
        mid = (p1 + p2) / 2
        direction_norm = direction / length
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, direction_norm)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            if direction_norm[2] > 0:
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            else:
                qw, qx, qy, qz = 0.0, 1.0, 0.0, 0.0
        else:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, direction_norm), -1, 1))
            qw = np.cos(angle / 2)
            qx = axis[0] * np.sin(angle / 2)
            qy = axis[1] * np.sin(angle / 2)
            qz = axis[2] * np.sin(angle / 2)
        
        sdf = f'''<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{color[0]} {color[1]} {color[2]} {color[3]}</ambient>
          <diffuse>{color[0]} {color[1]} {color[2]} {color[3]}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
        
        try:
            cmd = [
                'gz', 'service',
                '-s', f'/world/{self.world_name}/create',
                '--reqtype', 'gz.msgs.EntityFactory',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', f'sdf: "{sdf.replace(chr(10), " ").replace(chr(34), chr(92)+chr(34))}" '
                         f'pose: {{position: {{x: {mid[0]}, y: {mid[1]}, z: {mid[2]}}}, '
                         f'orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}}}'
            ]
            # Non-blocking subprocess call
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.spawned_segments.append(name)
                
        except Exception as e:
            self.get_logger().debug(f"Spawn line error: {e}")
    
    def _delete_entity(self, name: str):
        """Delete entity from Gazebo."""
        try:
            cmd = [
                'gz', 'service',
                '-s', f'/world/{self.world_name}/remove',
                '--reqtype', 'gz.msgs.Entity',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', f'name: "{name}" type: MODEL'
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        except Exception as e:
            self.get_logger().debug(f"Delete error: {e}")
    
    def _delete_triangle(self):
        """Delete all triangle waypoint spheres."""
        for name in self.triangle_segments:
            self._delete_entity(name)
        self.triangle_segments = []
        self.triangle_spawned = False
    
    def add_pen_point(self, position: np.ndarray):
        """Add point to pen path and draw line segment in Gazebo."""
        if self.last_point is not None:
            dist = np.linalg.norm(position - self.last_point)
            if dist < self.min_distance:
                return
            
            segment_name = f"pen_line_{self.line_segment_id}"
            self._spawn_line_segment(
                segment_name, self.last_point, position,
                self.line_color, 0.001  # 1mm radius
            )
            self.line_segment_id += 1
        
        self.pen_points.append(position.copy())
        self.last_point = position.copy()
    
    def reset(self):
        """Clear pen path (delete all pen line segments)."""
        self.get_logger().info("🔄 Resetting pen path...")
        
        for name in self.spawned_segments:
            if name.startswith("pen_line_"):
                self._delete_entity(name)
        
        self.spawned_segments = [s for s in self.spawned_segments if not s.startswith("pen_line_")]
        self.pen_points = []
        self.last_point = None
        self.line_segment_id = 0
        
        self.get_logger().info("✅ Pen path reset")
    
    def position_callback(self, msg: Point):
        """Handle pen position updates."""
        # Skip pen points during startup grace period
        elapsed = time.time() - self.startup_time
        if elapsed < self.STARTUP_GRACE_SEC:
            return
        
        position_base = np.array([msg.x, msg.y, msg.z])
        # Transform base_link -> world for Gazebo line drawing
        result = self._base_to_world(position_base)
        if result is None:
            return  # TF not ready yet
        position_world = result[0]
        self.add_pen_point(position_world)
    
    def reset_trajectory_callback(self, msg: Bool):
        """Reset pen path when training signals episode reset."""
        if msg.data:
            self.reset()
    
    def reset_callback(self, request, response):
        """Service callback to reset."""
        self.reset()
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GazeboDrawingVisualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
