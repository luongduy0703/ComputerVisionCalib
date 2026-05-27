#!/usr/bin/env python3
"""
Digital Twin: Communication Test — Laptop Side
===============================================
Publishes /twin/ping, subscribes /twin/pong and /pca9685_servo/joint_states.
Measures round-trip time (RTT) and confirms joint state reception.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import time


class TwinConnectionTestLaptop(Node):
    def __init__(self):
        super().__init__('twin_connection_test_laptop')

        self.ping_pub = self.create_publisher(String, '/twin/ping', 10)

        self.pong_sub = self.create_subscription(
            String, '/twin/pong', self.pong_callback, 10)
        self.js_sub = self.create_subscription(
            JointState, '/pca9685_servo/joint_states', self.js_callback, 10)

        self.timer = self.create_timer(0.5, self.send_ping)

        self.seq = 0
        self.pong_count = 0
        self.js_count = 0
        self.last_rtt = 0.0

        self.get_logger().info('🚀 Laptop twin test node started.')
        self.get_logger().info('   Waiting for Pi to respond on /twin/pong ...')

    def send_ping(self):
        msg = String()
        msg.data = f'{self.seq},{time.monotonic()}'
        self.ping_pub.publish(msg)
        self.seq += 1

        # Print summary every 6 pings (every 3 seconds)
        if self.seq % 6 == 0:
            self.get_logger().info(
                f'──  pings={self.seq}  pongs={self.pong_count}  '
                f'RTT={self.last_rtt:.1f}ms  joint_states={self.js_count}  ──'
            )
            if self.js_count == 0:
                self.get_logger().warn(
                    '⚠️  No joint_states! Check:\n'
                    '    1. Pi running: ros2 launch wicom_roboarm wicom_roboarm.launch.py\n'
                    '    2. Topic: /pca9685_servo/joint_states'
                )

    def pong_callback(self, msg: String):
        parts = msg.data.split(',')
        if len(parts) == 2:
            send_time = float(parts[1])
            self.last_rtt = (time.monotonic() - send_time) * 1000.0
            self.pong_count += 1
            self.get_logger().info(
                f'✅ PONG seq={parts[0]}  RTT={self.last_rtt:.1f}ms')

    def js_callback(self, msg: JointState):
        self.js_count += 1
        if self.js_count <= 3 or self.js_count % 30 == 0:
            names = ', '.join(msg.name)
            positions = ', '.join(f'{p:.2f}' for p in msg.position)
            self.get_logger().info(
                f'📊 JointStates #{self.js_count}: [{names}] = [{positions}]')


def main(args=None):
    rclpy.init(args=args)
    node = TwinConnectionTestLaptop()
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
