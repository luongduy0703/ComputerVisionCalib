#!/usr/bin/env python3
"""
Line Visualizer for Drawing Robot

Publishes visualization_msgs/Marker to display the pen path in Gazebo/RViz.
Tracks pen position history and renders as a line strip.
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Empty
import numpy as np
from typing import List


class LineVisualizer(Node):
    """ROS2 node for visualizing pen path as line markers."""
    
    def __init__(self):
        super().__init__('line_visualizer')
        
        self.get_logger().info("🎨 Line Visualizer starting...")
        
        # Parameters
        self.declare_parameter('line_width', 0.005)  # 5mm
        self.declare_parameter('line_color_r', 0.2)
        self.declare_parameter('line_color_g', 0.6)
        self.declare_parameter('line_color_b', 1.0)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('min_point_distance', 0.002)  # 2mm
        
        self.line_width = self.get_parameter('line_width').value
        self.color = ColorRGBA(
            r=float(self.get_parameter('line_color_r').value),
            g=float(self.get_parameter('line_color_g').value),
            b=float(self.get_parameter('line_color_b').value),
            a=1.0
        )
        self.frame_id = self.get_parameter('frame_id').value
        self.min_distance = self.get_parameter('min_point_distance').value
        
        # Line points storage
        self.line_points: List[Point] = []
        self.last_point = None
        
        # Publisher for line markers
        self.marker_pub = self.create_publisher(Marker, '/drawing/pen_path', 10)
        
        # Publisher for target shape preview
        self.shape_pub = self.create_publisher(Marker, '/drawing/target_shape', 10)
        
        # Service to reset the line
        self.reset_srv = self.create_service(Empty, '/drawing/reset_line', self.reset_callback)
        
        # Subscriber for pen position
        self.position_sub = self.create_subscription(
            Point, '/drawing/pen_position', self.position_callback, 10
        )
        
        # Timer to publish markers
        self.timer = self.create_timer(0.033, self.publish_markers)  # 30Hz for smoother drawing
        
        self.get_logger().info("✅ Line Visualizer ready!")
        self.get_logger().info("   Listen: /drawing/pen_position")
        self.get_logger().info("   Reset:  /drawing/reset_line")
    
    def add_point(self, position: np.ndarray):
        """Add a point to the line."""
        if self.last_point is not None:
            dist = np.linalg.norm(position - self.last_point)
            if dist < self.min_distance:
                return
        
        point = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
        self.line_points.append(point)
        self.last_point = np.array(position)
    
    def reset(self):
        """Clear the drawn line."""
        self.line_points = []
        self.last_point = None
        
        # Publish delete marker
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pen_path"
        marker.id = 0
        marker.action = Marker.DELETE
        self.marker_pub.publish(marker)
        
        self.get_logger().info("🔄 Line reset")
    
    def set_target_shape(self, waypoints: np.ndarray, color=(0.3, 0.3, 0.3)):
        """Display target shape as preview."""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target_shape"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = self.line_width * 0.5
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.5)
        
        for wp in waypoints:
            marker.points.append(Point(x=float(wp[0]), y=float(wp[1]), z=float(wp[2])))
        
        self.shape_pub.publish(marker)
    
    def position_callback(self, msg: Point):
        """Handle pen position updates."""
        position = np.array([msg.x, msg.y, msg.z])
        self.add_point(position)
    
    def reset_callback(self, request, response):
        """Service callback to reset line."""
        self.reset()
        return response
    
    def publish_markers(self):
        """Publish line marker."""
        if len(self.line_points) < 2:
            return
        
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pen_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = self.line_width
        marker.color = self.color
        marker.points = self.line_points
        
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = LineVisualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
