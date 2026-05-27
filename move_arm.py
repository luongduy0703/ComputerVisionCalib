#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class ArmMover(Node):
    def __init__(self):
        super().__init__('arm_mover_test')
        self.publisher_ = self.create_publisher(
            JointTrajectory, 
            '/arm_controller/joint_trajectory', 
            10
        )
        self.get_logger().info("Đang gửi lệnh xoay khớp...")
        
        # Tạo thông điệp điều khiển
        msg = JointTrajectory()
        msg.joint_names = [
            'Revolute 20', 'Revolute 22', 'Revolute 23', 
            'Revolute 26', 'Revolute 28', 'Revolute 30'
        ]
        
        point = JointTrajectoryPoint()
        # Thay đổi một vài góc (đơn vị: radian)
        # Khớp số 1 và 2 xoay nhẹ 0.5 rad (~28 độ)
        point.positions = [0.5, -0.5, 0.5, 0.0, 0.5, 0.0] 
        
        # Đi tới góc đó trong vòng 2 giây
        point.time_from_start.sec = 2
        point.time_from_start.nanosec = 0
        
        msg.points.append(point)
        
        # Đợi 1 giây để publisher kết nối
        import time
        time.sleep(1.0)
        
        self.publisher_.publish(msg)
        self.get_logger().info("Đã gửi lệnh!")

def main(args=None):
    rclpy.init(args=args)
    node = ArmMover()
    rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
