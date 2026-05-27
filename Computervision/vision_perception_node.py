"""
AeroScript — Vision Perception ROS 2 Node
==========================================
Cầu nối giữa module Vision (YOLO + ArUco) và Kinematics (TF2).

Chức năng:
  1. Subscribe /camera/image_raw          → nhận ảnh từ camera Gazebo
  2. YOLO detect bút sơn                  → tâm pixel bút
  3. ArUco Multi-Marker                    → Z_wall, Normal Vector, Z_target
  4. TF2 lookup pen_tip_link               → tọa độ 3D thực tế (FK)
  5. Publish /aeroscript/pen_2d            → tọa độ pixel bút
  6. Publish /aeroscript/target_pose       → mục tiêu PoseStamped cho tay máy
  7. Log so sánh Z_target (ArUco) vs Z_pen_tip (FK) liên tục

Tác giả : AeroScript Team
Ngày    : 2026-05-05
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, PoseStamped

import cv2
import numpy as np
from ultralytics import YOLO

import tf2_ros
from tf2_ros import TransformException

import math


# ═══════════════════════════════════════════════════════════
#  CẤU HÌNH
# ═══════════════════════════════════════════════════════════

# --- YOLO ---
MODEL_PATH = "/home/luongduy/AeroScript_Vision/runs/detect/runs/detect/aeroscript_pen_model/weights/best.pt"
YOLO_CONFIDENCE = 0.6

# --- ArUco ---
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_250
MARKER_LENGTH_CM = 2.0      # ← Kích thước thực marker (cm). Chỉnh lại nếu cần.

# --- Camera giả định (chưa calib) ---
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# --- Cơ khí lò xo ---
DELTA_D_MM = 10.0           # Khoảng nén lò xo (mm)

# --- TF2 Frames ---
BASE_FRAME    = "base_link"       # Hệ trục gốc (world hoặc base_link)
PEN_TIP_FRAME = "bibut_1"         # Hệ trục đầu bút trên tay máy (trong URDF là bibut_1)

# --- Tần suất xử lý ---
TIMER_PERIOD_SEC = 0.05     # 20 Hz


class VisionPerceptionNode(Node):
    """
    ROS 2 Node xử lý Vision cho hệ thống AeroScript UAV.
    Kết hợp YOLO tracking + ArUco pose estimation + TF2 FK lookup.
    """

    def __init__(self):
        super().__init__('vision_perception_node')
        self.get_logger().info("=" * 55)
        self.get_logger().info("  AeroScript Vision Perception Node — Khởi động")
        self.get_logger().info("=" * 55)

        # ─────────────────────────────────────────
        # 1. KHỞI TẠO YOLO
        # ─────────────────────────────────────────
        self.get_logger().info("[INIT] Đang load YOLO model...")
        self.yolo_model = YOLO(MODEL_PATH)
        self.get_logger().info(
            f"[INIT] YOLO sẵn sàng. Classes: {self.yolo_model.names}"
        )

        # ─────────────────────────────────────────
        # 2. KHỞI TẠO ArUco (API mới OpenCV 4.8+)
        # ─────────────────────────────────────────
        self.get_logger().info("[INIT] Đang khởi tạo ArUco detector...")
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        self.marker_length = MARKER_LENGTH_CM
        self.get_logger().info(
            f"[INIT] ArUco DICT_4X4_250, marker size = {self.marker_length} cm"
        )

        # Object points cho 1 marker (4 góc, Z=0)
        half = self.marker_length / 2.0
        self.marker_obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float64)

        # ─────────────────────────────────────────
        # 3. CAMERA MATRIX (giả định)
        # ─────────────────────────────────────────
        fx = fy = float(FRAME_WIDTH)
        cx, cy = FRAME_WIDTH / 2.0, FRAME_HEIGHT / 2.0
        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self.get_logger().info(
            f"[INIT] Camera matrix (giả định {FRAME_WIDTH}x{FRAME_HEIGHT})"
        )

        # ─────────────────────────────────────────
        # 4. FRAME BUFFER (không dùng cv_bridge vì xung đột NumPy)
        # ─────────────────────────────────────────
        self.latest_frame = None  # Frame mới nhất từ camera

        # ─────────────────────────────────────────
        # 5. TF2 LISTENER
        # ─────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.get_logger().info(
            f"[INIT] TF2 Listener: {BASE_FRAME} → {PEN_TIP_FRAME}"
        )

        # ─────────────────────────────────────────
        # 6. BIẾN NHỚ TRẠNG THÁI (Fail-safe)
        # ─────────────────────────────────────────
        self.last_good_normal = np.array([0.0, 0.0, 1.0])
        self.last_good_z_wall = 0.0
        self.last_good_z_target = 0.0
        self.pose_frozen = True   # Ban đầu chưa có dữ liệu
        self.pen_2d = None        # (cx, cy) pixel

        # ─────────────────────────────────────────
        # 7. ROS 2 SUBSCRIBER
        # ─────────────────────────────────────────
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self._image_callback,
            10
        )
        self.get_logger().info("[INIT] Subscribed: /camera/image_raw")

        # ─────────────────────────────────────────
        # 8. ROS 2 PUBLISHERS
        # ─────────────────────────────────────────
        self.pen_2d_pub = self.create_publisher(
            Point,
            '/aeroscript/pen_2d',
            10
        )
        self.target_pose_pub = self.create_publisher(
            PoseStamped,
            '/aeroscript/target_pose',
            10
        )
        self.get_logger().info("[INIT] Publishers: /aeroscript/pen_2d, /aeroscript/target_pose")

        # ─────────────────────────────────────────
        # 9. TIMER — Vòng lặp xử lý chính
        # ─────────────────────────────────────────
        self.timer = self.create_timer(TIMER_PERIOD_SEC, self._process_loop)
        self.get_logger().info(
            f"[INIT] Timer: {1.0/TIMER_PERIOD_SEC:.0f} Hz. Sẵn sàng!\n"
        )

    # ═══════════════════════════════════════════════════════
    #  CALLBACK: Nhận ảnh từ camera
    # ═══════════════════════════════════════════════════════

    def _image_callback(self, msg: Image):
        """
        Chuyển ROS Image → OpenCV frame THỦ CÔNG (không dùng cv_bridge
        để tránh xung đột NumPy 1.x vs 2.x).
        """
        try:
            # Chuyển bytes → numpy array theo encoding
            if msg.encoding in ('bgr8', 'rgb8'):
                frame = np.frombuffer(msg.data, dtype=np.uint8)
                frame = frame.reshape((msg.height, msg.width, 3))
                if msg.encoding == 'rgb8':
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif msg.encoding in ('mono8',):
                frame = np.frombuffer(msg.data, dtype=np.uint8)
                frame = frame.reshape((msg.height, msg.width))
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                self.get_logger().warn(
                    f"[IMAGE] Encoding không hỗ trợ: {msg.encoding}"
                )
                return
            self.latest_frame = frame
        except Exception as e:
            self.get_logger().warn(f"[IMAGE] Lỗi chuyển đổi ảnh: {e}")

    # ═══════════════════════════════════════════════════════
    #  VÒNG LẶP XỬ LÝ CHÍNH (Timer callback)
    # ═══════════════════════════════════════════════════════

    def _process_loop(self):
        """Pipeline xử lý mỗi chu kỳ timer."""

        # --- Kiểm tra frame ---
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ─────────────────────────────────────────
        # BƯỚC A: YOLO TRACKING BÚT (Độc lập)
        # ─────────────────────────────────────────
        self.pen_2d = self._detect_pen(frame)

        if self.pen_2d is not None:
            # Publish tọa độ pixel bút
            pen_msg = Point()
            pen_msg.x = float(self.pen_2d[0])
            pen_msg.y = float(self.pen_2d[1])
            pen_msg.z = 0.0
            self.pen_2d_pub.publish(pen_msg)

        # ─────────────────────────────────────────
        # BƯỚC B: ArUco MULTI-MARKER DETECTION
        # ─────────────────────────────────────────
        corners, ids, n_markers = self._detect_aruco(frame_gray)

        # ─────────────────────────────────────────
        # BƯỚC C + D: POSE ESTIMATION + FAIL-SAFE
        # ─────────────────────────────────────────
        if n_markers > 0:
            result = self._compute_pose(corners, ids)
            if result is not None:
                normal_vec, z_wall, z_target = result
                self.last_good_normal = normal_vec.copy()
                self.last_good_z_wall = z_wall
                self.last_good_z_target = z_target
                self.pose_frozen = False
            else:
                self.pose_frozen = True
        else:
            # ── FAIL-SAFE: 0 marker → giữ nguyên mục tiêu cũ ──
            if not self.pose_frozen:
                self.get_logger().warn(
                    "WARN: Marker occluded! Giữ nguyên tư thế cũ."
                )
            self.pose_frozen = True

        # ─────────────────────────────────────────
        # BƯỚC E: PUBLISH TARGET POSE
        # ─────────────────────────────────────────
        self._publish_target_pose()

        # ─────────────────────────────────────────
        # BƯỚC F: TF2 — LẤY TỌA ĐỘ 3D THỰC TẾ (FK)
        # ─────────────────────────────────────────
        pen_tip_xyz = self._lookup_pen_tip_tf()

        # ─────────────────────────────────────────
        # BƯỚC G: LOG SO SÁNH THỜI GIAN THỰC
        # ─────────────────────────────────────────
        status = "[FROZEN]" if self.pose_frozen else "[LIVE]"
        self.get_logger().info(
            f"👉 [MỤC TIÊU ArUco] Z_target: "
            f"{self.last_good_z_target:.2f} mm  "
            f"Normal: ({self.last_good_normal[0]:.3f}, "
            f"{self.last_good_normal[1]:.3f}, "
            f"{self.last_good_normal[2]:.3f}) {status}"
        )

        if pen_tip_xyz is not None:
            self.get_logger().info(
                f"👉 [THỰC TẾ FK Bút] X: {pen_tip_xyz[0]:.4f}, "
                f"Y: {pen_tip_xyz[1]:.4f}, "
                f"Z_pen_tip: {pen_tip_xyz[2]:.4f}"
            )
        else:
            self.get_logger().info(
                "👉 [THỰC TẾ FK Bút] TF chưa sẵn sàng..."
            )

    # ═══════════════════════════════════════════════════════
    #  YOLO: Detect bút sơn
    # ═══════════════════════════════════════════════════════

    def _detect_pen(self, frame):
        """
        Chạy YOLO inference, trả về tâm pixel (cx, cy) hoặc None.
        """
        results = self.yolo_model(frame, conf=YOLO_CONFIDENCE, verbose=False)

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                return (cx, cy)
        return None

    # ═══════════════════════════════════════════════════════
    #  ArUco: Detect markers
    # ═══════════════════════════════════════════════════════

    def _detect_aruco(self, frame_gray):
        """
        Tìm ArUco markers trong frame.
        Returns: (corners, ids, n_detected)
        """
        corners, ids, _ = self.aruco_detector.detectMarkers(frame_gray)
        n_detected = 0 if ids is None else len(ids)
        return corners, ids, n_detected

    # ═══════════════════════════════════════════════════════
    #  POSE: Ước lượng tư thế từ đa marker
    # ═══════════════════════════════════════════════════════

    def _compute_pose(self, corners, ids):
        """
        Chạy solvePnP cho từng marker, tổng hợp kết quả.

        ---------------------------------------------------------------
        TOÁN HỌC:
          • Pháp tuyến: R = Rodrigues(rvec_0), n⃗ = R[:, 2]
            (Cột thứ 3 của ma trận xoay = trục Z marker = pháp tuyến
             mặt phẳng tường vì marker nằm phẳng trên tường)
          • Z_wall = mean(Z của tất cả marker)  → giảm nhiễu
          • Z_target = Z_wall + Δd (nén lò xo)
        ---------------------------------------------------------------

        Returns: (normal_vec, z_wall, z_target) hoặc None
        """
        rvecs = []
        tvecs = []

        for i in range(len(ids)):
            img_pts = corners[i][0].astype(np.float64)
            try:
                success, rvec, tvec = cv2.solvePnP(
                    self.marker_obj_points, img_pts,
                    self.K, self.dist_coeffs
                )
            except cv2.error:
                continue
            if not success:
                continue
            rvecs.append(rvec)
            tvecs.append(tvec)

        if len(rvecs) == 0:
            return None

        # ── PHÁP TUYẾN: Cột 3 của R (từ marker đầu tiên) ──
        R, _ = cv2.Rodrigues(rvecs[0])
        normal_vec = R[:, 2]

        # ── Z_WALL: Trung bình Z tất cả marker ──
        z_values = [float(tv[2][0]) for tv in tvecs]
        z_wall = float(np.mean(z_values))

        # ── NÉN LÒ XO ──
        z_target = z_wall + DELTA_D_MM

        return normal_vec, z_wall, z_target

    # ═══════════════════════════════════════════════════════
    #  TF2: Lấy tọa độ 3D đầu bút (Động học thuận - FK)
    # ═══════════════════════════════════════════════════════

    def _lookup_pen_tip_tf(self):
        """
        Dùng TF2 lookup_transform để lấy tọa độ 3D thực tế
        của pen_tip_link trong hệ base_link.

        Xử lý Exception ở những frame đầu khi Gazebo chưa kịp
        publish cây TF.

        Returns: (x, y, z) tuple hoặc None
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                PEN_TIP_FRAME,
                rclpy.time.Time()   # Lấy transform mới nhất
            )
            t = transform.transform.translation
            return (t.x, t.y, t.z)

        except TransformException as ex:
            # Những frame đầu Gazebo chưa publish TF → bỏ qua
            self.get_logger().debug(
                f"[TF2] Chưa có transform {BASE_FRAME}→{PEN_TIP_FRAME}: {ex}"
            )
            return None

    # ═══════════════════════════════════════════════════════
    #  PUBLISH: Target Pose cho Kinematics
    # ═══════════════════════════════════════════════════════

    def _publish_target_pose(self):
        """
        Publish PoseStamped chứa:
          - Position: (X, Y từ pen_2d quy đổi, Z_target)
          - Orientation: Normal Vector → Quaternion

        Lưu ý: X, Y ở đây là tọa độ pixel quy đổi tạm thời.
        Trong thực tế cần camera calibration đầy đủ.
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = BASE_FRAME

        # ── Position ──
        if self.pen_2d is not None and self.last_good_z_wall > 0:
            cx, cy = self.pen_2d
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx0, cy0 = self.K[0, 2], self.K[1, 2]
            pose_msg.pose.position.x = float((cx - cx0) * self.last_good_z_wall / fx)
            pose_msg.pose.position.y = float((cy - cy0) * self.last_good_z_wall / fy)
        else:
            pose_msg.pose.position.x = 0.0
            pose_msg.pose.position.y = 0.0

        pose_msg.pose.position.z = float(self.last_good_z_target)

        # ── Orientation: Normal Vector → Quaternion ──
        # Quy ước: normal là hướng tiếp cận (approach direction)
        # Chuyển normal vector thành quaternion biểu diễn hướng đó
        q = self._normal_to_quaternion(self.last_good_normal)
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]

        self.target_pose_pub.publish(pose_msg)

    # ═══════════════════════════════════════════════════════
    #  UTILS: Normal Vector → Quaternion
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _normal_to_quaternion(normal):
        """
        Chuyển vector pháp tuyến (nx, ny, nz) thành Quaternion.

        Ý tưởng: Tìm phép xoay từ trục Z mặc định (0, 0, 1)
        sang hướng normal. Dùng axis-angle rồi chuyển sang quaternion.

        Returns: (qx, qy, qz, qw) tuple
        """
        n = normal / (np.linalg.norm(normal) + 1e-9)
        z_axis = np.array([0.0, 0.0, 1.0])

        dot = np.dot(z_axis, n)

        # Trường hợp đặc biệt: cùng hướng
        if dot > 0.9999:
            return (0.0, 0.0, 0.0, 1.0)

        # Trường hợp đặc biệt: ngược hướng
        if dot < -0.9999:
            return (1.0, 0.0, 0.0, 0.0)  # 180° quanh trục X

        # Tính trục xoay: cross(z_axis, normal)
        axis = np.cross(z_axis, n)
        axis = axis / (np.linalg.norm(axis) + 1e-9)

        # Góc xoay
        angle = math.acos(np.clip(dot, -1.0, 1.0))

        # Axis-angle → Quaternion
        half_angle = angle / 2.0
        sin_half = math.sin(half_angle)

        qx = axis[0] * sin_half
        qy = axis[1] * sin_half
        qz = axis[2] * sin_half
        qw = math.cos(half_angle)

        return (qx, qy, qz, qw)


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)

    node = VisionPerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[EXIT] Đã tắt node.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
