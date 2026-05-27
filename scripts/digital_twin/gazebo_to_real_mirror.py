#!/usr/bin/env python3
"""
Digital Twin: Gazebo to Real Mirror (Sim-to-Real)
==================================================
Subscribes to Gazebo's /joint_states (RADIANS)
and publishes JointState commands to Pi's /pca9685_servo/command (DEGREES).

4-DOF mode: base, shoulder, elbow, pen are forwarded.
RATE LIMITED to 10Hz to avoid overwhelming the Pi's servo controller.

Pi joint limits (degrees):
  j1 (base):     0° (left)  → 90° (home) → 180° (right)
  j2 (shoulder): 0° (down)  → 180° (up)
  j3 (elbow):    180° (down) → 0° (up)   [INVERTED]
  j4 (pen):      0° (down)  → 180° (up)
"""

import math
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


# Gazebo joint → (Pi name, pi_home_deg, inverted)
REVERSE_MAPPING = {
    "Revolute 20": ("base",         90.0,  False),
    "Revolute 22": ("shoulder",     90.0, False),  # 0 upfront, 90 down, 180 under
    "Revolute 23": ("elbow",        90.0,  True),  # 90 is home neutral
    "Revolute 26": ("wrist_roll",    0.0,  True),  # J4: Gazebo=90 -> Pi=90
    "Revolute 28": ("wrist_pitch",  90.0,  False),  # J5: Gazebo=0 -> Pi=90
    "Revolute 30": ("pen",          90.0,  False),  # J6: Gazebo=0 -> Pi=90
}

# Only send commands at this rate (Hz) — Pi can't handle 50Hz
PUBLISH_RATE_HZ = 10.0
MIN_CHANGE_DEG = 0.5  # Don't send if change is less than this


def rad_to_deg(rad):
    return rad * 180.0 / math.pi


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


class GazeboToRealMirror(Node):
    def __init__(self):
        super().__init__('gazebo_to_real_mirror')

        self.command_pub = self.create_publisher(
            JointState,
            "/pca9685_servo/command",
            10
        )

        self.js_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_states_callback,
            10
        )

        self.last_send_time = 0.0
        self.min_interval = 1.0 / PUBLISH_RATE_HZ
        self.last_sent_positions = {}  # Track last sent positions
        self.msg_count = 0

        self.get_logger().info("🔄 Sim-to-Real mirror started (4-DOF)")
        self.get_logger().info(f"   Rate limited to {PUBLISH_RATE_HZ}Hz")
        self.get_logger().info("   Gazebo (radians) → Pi (degrees)")

    def gazebo_rad_to_pi_deg(self, gazebo_rad, home_deg, inverted):
        """Convert Gazebo radians to Pi servo degrees."""
        offset_deg = rad_to_deg(gazebo_rad)
        if inverted:
            offset_deg = -offset_deg
        pi_deg = home_deg + offset_deg
        return clamp(pi_deg, 0.0, 180.0)

    def joint_states_callback(self, msg: JointState):
        # Rate limiting
        now = time.monotonic()
        if (now - self.last_send_time) < self.min_interval:
            return

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()

        has_significant_change = False

        for gz_name, position in zip(msg.name, msg.position):
            if gz_name in REVERSE_MAPPING:
                pi_name, home, inv = REVERSE_MAPPING[gz_name]
                pi_deg = self.gazebo_rad_to_pi_deg(position, home, inv)
                cmd.name.append(pi_name)
                cmd.position.append(pi_deg)

                # Check if position changed enough to warrant sending
                last = self.last_sent_positions.get(pi_name, None)
                if last is None or abs(pi_deg - last) > MIN_CHANGE_DEG:
                    has_significant_change = True

        if not cmd.name or not has_significant_change:
            return

        self.command_pub.publish(cmd)
        self.last_send_time = now

        # Update last sent positions
        for name, pos in zip(cmd.name, cmd.position):
            self.last_sent_positions[name] = pos

        self.msg_count += 1
        if self.msg_count <= 5 or self.msg_count % 50 == 0:
            pos_str = ', '.join(f'{n}={p:.1f}°' for n, p in zip(cmd.name, cmd.position))
            self.get_logger().info(f"🔄 #{self.msg_count}: {pos_str}")


def main(args=None):
    rclpy.init(args=args)
    node = GazeboToRealMirror()
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
