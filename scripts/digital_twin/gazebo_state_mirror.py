#!/usr/bin/env python3
"""
Digital Twin: Gazebo State Mirror (Real-to-Sim)
================================================
Subscribes to the Pi's /pca9685_servo/joint_states (DEGREES)
and publishes JointTrajectory commands to Gazebo's
/arm_controller/joint_trajectory (RADIANS).

4-DOF mode: base, shoulder, elbow, pen are active.
Revolute 26 and Revolute 28 are held at 0.

Pi joint limits (degrees):
  j1 (base):     0° (left)  → 90° (home) → 180° (right)
  j2 (shoulder): 0° (down)  → 180° (up)
  j3 (elbow):    180° (down) → 0° (up)   [INVERTED]
  j4 (pen):      0° (down)  → 180° (up)
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


# ── Joint Mapping ──
# Pi publishes in DEGREES, Gazebo expects RADIANS
# We also need to handle the offset: Pi home is at 90° (π/2 rad)
# whereas Gazebo home is at 0 rad.

ACTIVE_JOINTS = [
    # (pi_name, gazebo_name, pi_home_deg, pi_inverted)
    ("base",        "Revolute 20", 90.0,  False),
    ("shoulder",    "Revolute 22",  90.0, False),
    ("elbow",       "Revolute 23",  90.0,   True),
    ("wrist_roll",  "Revolute 26", 0.0,   True),  # J4
    ("wrist_pitch", "Revolute 28", 90.0,  False),  # J5
    ("pen",         "Revolute 30", 90.0,  False),  # J6
]

STATIC_JOINTS = {
    "Revolute 26": 0.0,
    "Revolute 28": 0.0,
}

ALL_GAZEBO_JOINTS = [
    "Revolute 20", "Revolute 22", "Revolute 23",
    "Revolute 26", "Revolute 28", "Revolute 30",
]


def deg_to_rad(deg):
    return deg * math.pi / 180.0


class GazeboStateMirror(Node):
    def __init__(self):
        super().__init__('gazebo_state_mirror')

        self.traj_pub = self.create_publisher(
            JointTrajectory,
            "/arm_controller/joint_trajectory",
            10
        )

        self.js_sub = self.create_subscription(
            JointState,
            "/pca9685_servo/joint_states",
            self.joint_states_callback,
            10
        )

        # Build lookup: pi_name → (gazebo_name, pi_home_deg, inverted)
        self.pi_to_gz = {
            pn: (gn, home, inv) for pn, gn, home, inv in ACTIVE_JOINTS
        }

        self.msg_count = 0
        self.get_logger().info("🪞 Real-to-Sim mirror started (4-DOF)")
        self.get_logger().info("   Pi (degrees) → Gazebo (radians)")

    def pi_deg_to_gazebo_rad(self, pi_deg, home_deg, inverted):
        """Convert Pi servo degrees to Gazebo radians.
        
        Pi servos: 0°-180°, home at home_deg
        Gazebo:    radians, home at 0 rad
        
        Formula: gazebo_rad = (pi_deg - home_deg) * π/180
        If inverted: negate the result
        """
        offset_deg = pi_deg - home_deg
        if inverted:
            offset_deg = -offset_deg
        return deg_to_rad(offset_deg)

    def joint_states_callback(self, msg: JointState):
        pi_lookup = dict(zip(msg.name, msg.position))

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ALL_GAZEBO_JOINTS

        point = JointTrajectoryPoint()
        point.time_from_start = Duration(sec=0, nanosec=50_000_000)

        for gz_joint in ALL_GAZEBO_JOINTS:
            if gz_joint in STATIC_JOINTS:
                point.positions.append(STATIC_JOINTS[gz_joint])
            else:
                # Find Pi joint for this Gazebo joint
                found = False
                for pi_name, (gz_name, home, inv) in self.pi_to_gz.items():
                    if gz_name == gz_joint and pi_name in pi_lookup:
                        rad = self.pi_deg_to_gazebo_rad(
                            pi_lookup[pi_name], home, inv
                        )
                        point.positions.append(rad)
                        found = True
                        break
                if not found:
                    point.positions.append(0.0)

        traj.points.append(point)
        self.traj_pub.publish(traj)

        self.msg_count += 1
        if self.msg_count <= 3:
            pos_str = ', '.join(f'{p:.2f}' for p in point.positions)
            self.get_logger().info(f"🪞 Frame #{self.msg_count}: [{pos_str}] rad")


def main(args=None):
    rclpy.init(args=args)
    node = GazeboStateMirror()
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
