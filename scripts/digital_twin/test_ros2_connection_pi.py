#!/usr/bin/env python3
"""
Digital Twin: Communication Test — Pi Side
==========================================
Subscribes /twin/ping, echoes back on /twin/pong.
Run this on the Raspberry Pi.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TwinConnectionTestPi(Node):
    def __init__(self):
        super().__init__('twin_connection_test_pi')

        self.pong_pub = self.create_publisher(String, '/twin/pong', 10)
        self.ping_sub = self.create_subscription(
            String, '/twin/ping', self.ping_callback, 10)

        self.count = 0
        self.get_logger().info('🤖 Pi twin test node started, echoing /twin/ping → /twin/pong')

    def ping_callback(self, msg: String):
        self.pong_pub.publish(msg)  # Echo back exactly
        self.count += 1
        if self.count <= 3 or self.count % 30 == 0:
            self.get_logger().info(f'📡 Ping #{self.count} received, pong sent back!')


def main(args=None):
    rclpy.init(args=args)
    node = TwinConnectionTestPi()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
