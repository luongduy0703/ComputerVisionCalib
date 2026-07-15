#!/usr/bin/env python3
"""
run_pi4_ros.py  —  ROS2 Node chạy trên Raspberry Pi 4
=======================================================
Subscribe camera → TFLite inference → solvePnP → publish kết quả.

Topics:
  Subscribe : /image_raw            (sensor_msgs/Image)  — từ usb_cam
  Publish   : /aeroscript/pen_image (sensor_msgs/Image)  — frame annotated
  Publish   : /aeroscript/pen_xyz   (geometry_msgs/Point)— tọa độ 3D (mm)
  Publish   : /aeroscript/pen_pose  (geometry_msgs/PoseStamped) — đầy đủ

web_video_server sẽ tự động phát /aeroscript/pen_image ra HTTP.
Laptop xem tại: http://localhost:8080/stream?topic=/aeroscript/pen_image

Cài đặt trên Pi:
    pip3 install "numpy<2"
    pip3 install ai-edge-litert   # hoặc tflite-runtime

Chạy:
    # Terminal 1 — camera
    ros2 run usb_cam usb_cam_node_exe --ros-args \\
        -p video_device:="/dev/video0" -p image_width:=640 -p image_height:=480 \\
        -p pixel_format:="yuyv" -p auto_exposure:=true -p brightness:=128

    # Terminal 2 — web video server
    ros2 run web_video_server web_video_server

    # Terminal 3 — node này
    python3 run_pi4_ros.py --model best_float16.tflite

    # Laptop — xem stream (Chrome hoặc view_stream.py)
    http://localhost:8080/stream?topic=/aeroscript/pen_image&type=mjpeg
"""
import os
import sys
import time
import socket
import argparse
import threading
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge


# ══════════════════════════════════════════════════════════════════════════════
# 1. TFLite Inference Engine
# ══════════════════════════════════════════════════════════════════════════════
class TFLitePoseInference:
    def __init__(self, model_path: str, conf_thresh=0.55, iou_thresh=0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh

        try:
            import tflite_runtime.interpreter as tflite
            self.interp = tflite.Interpreter(model_path=model_path, num_threads=4)
            print("✅ tflite-runtime")
        except ImportError:
            try:
                from ai_edge_litert.interpreter import Interpreter
                self.interp = Interpreter(model_path=model_path, num_threads=4)
                print("✅ ai-edge-litert")
            except ImportError:
                import tensorflow as tf
                self.interp = tf.lite.Interpreter(model_path=model_path, num_threads=4)
                print("✅ tensorflow.lite")

        self.interp.allocate_tensors()
        inp = self.interp.get_input_details()[0]
        out = self.interp.get_output_details()[0]

        self.input_idx  = inp["index"]
        self.output_idx = out["index"]
        self.inp_dtype  = inp["dtype"]
        self.is_int8    = self.inp_dtype in (np.int8, np.uint8)
        self.is_fp16    = self.inp_dtype == np.float16
        self.inp_scale, self.inp_zero = inp.get("quantization", (1.0, 0))

        shape = inp["shape"]
        if shape[1] == 3:           # NCHW
            self.imgsz, self.is_nhwc = int(shape[2]), False
        else:                       # NHWC
            self.imgsz, self.is_nhwc = int(shape[1]), True

        print(f"   Model: {shape} {'NHWC' if self.is_nhwc else 'NCHW'} "
              f"{'FP16' if self.is_fp16 else 'INT8' if self.is_int8 else 'FP32'}")

    def __call__(self, bgr):
        tensor, r, dw, dh = self._preprocess(bgr)
        self.interp.set_tensor(self.input_idx, tensor)
        self.interp.invoke()
        raw   = self.interp.get_tensor(self.output_idx)
        preds = raw[0].T if raw.ndim == 3 else raw[0]
        return self._decode(preds, r, dw, dh)

    def _preprocess(self, bgr):
        h0, w0 = bgr.shape[:2]
        r = self.imgsz / max(h0, w0)
        nw, nh = int(w0*r), int(h0*r)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        dw, dh = (self.imgsz-nw)//2, (self.imgsz-nh)//2
        canvas[dh:dh+nh, dw:dw+nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = canvas[:,:,::-1].astype(np.float32)/255.0
        t = rgb[np.newaxis] if self.is_nhwc else np.transpose(rgb,(2,0,1))[np.newaxis]
        if self.is_int8:  t = (t/self.inp_scale + self.inp_zero).astype(np.int8)
        elif self.is_fp16: t = t.astype(np.float16)
        return t, r, dw, dh

    def _decode(self, preds, r, dw, dh):
        # Layout: [cx,cy,w,h,conf, kp0x,kp0y,kp0v, kp1x,...kp3v] = 17 fields, NO cls
        if preds.shape[1] < 17: return None, None
        mask = preds[:,4] > self.conf_thresh
        if not np.any(mask): return None, None
        f    = preds[mask]
        keep = self._nms(f[:,:4], f[:,4], self.iou_thresh)
        if not keep: return None, None
        best    = f[keep[0]]
        kraw    = best[5:17].reshape(4,3)   # kpts từ index 5 (không có cls field)
        kxy     = kraw[:,:2].copy()
        kv      = kraw[:,2]
        kxy[:,0] = (kxy[:,0]-dw)/r
        kxy[:,1] = (kxy[:,1]-dh)/r
        return kxy.astype(np.float32), kv.astype(np.float32)

    @staticmethod
    def _nms(boxes, scores, thr):
        if not len(boxes): return []
        cx,cy,bw,bh = boxes.T
        x1,y1,x2,y2 = cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2
        areas = bw*bh; order = scores.argsort()[::-1]; keep=[]
        while order.size:
            i=order[0]; keep.append(int(i))
            if order.size==1: break
            ix1=np.maximum(x1[i],x1[order[1:]]); iy1=np.maximum(y1[i],y1[order[1:]])
            ix2=np.minimum(x2[i],x2[order[1:]]); iy2=np.minimum(y2[i],y2[order[1:]])
            inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
            iou=inter/(areas[i]+areas[order[1:]]-inter+1e-6)
            order=order[1:][iou<=thr]
        return keep


# ══════════════════════════════════════════════════════════════════════════════
# 2. Kalman Filter 3D
# ══════════════════════════════════════════════════════════════════════════════
class PoseKalmanFilter:
    def __init__(self, process_noise=0.5, measurement_noise=15.0, max_jump_mm=250.0):
        self.pn=process_noise; self.mn=measurement_noise; self.mj=max_jump_mm
        self.initialized=False; self.last=None; self._init()

    def _init(self):
        self.kf=cv2.KalmanFilter(6,3,0)
        F=np.eye(6,dtype=np.float32); F[0,3]=F[1,4]=F[2,5]=1.0
        self.kf.transitionMatrix=F
        H=np.zeros((3,6),dtype=np.float32); H[0,0]=H[1,1]=H[2,2]=1.0
        self.kf.measurementMatrix=H
        self.kf.processNoiseCov    =np.eye(6,dtype=np.float32)*self.pn
        self.kf.measurementNoiseCov=np.eye(3,dtype=np.float32)*self.mn
        self.kf.errorCovPost       =np.eye(6,dtype=np.float32)*500.0

    def update(self, tvec):
        m=np.array([tvec[0][0],tvec[1][0],tvec[2][0]],dtype=np.float32)
        if not self.initialized: self._init_at(m); return tvec.copy()
        if np.linalg.norm(m-self.last)>self.mj: self._init_at(m); return tvec.copy()
        self.last=m.copy(); self.kf.predict()
        return self.kf.correct(m.reshape(3,1))[:3,0].reshape(3,1)

    def _init_at(self,m):
        self._init(); s=np.zeros(6,dtype=np.float32); s[:3]=m
        self.kf.statePost=s.reshape(6,1); self.last=m.copy(); self.initialized=True

    def reset(self): self._init(); self.initialized=False; self.last=None




# ══════════════════════════════════════════════════════════════════════════════
# 3b. HTTP JSON Server — trả về XYZ cho laptop (port 8081)
# ══════════════════════════════════════════════════════════════════════════════
class _JSONHandler(BaseHTTPRequestHandler):
    """Handler nhỏ — trả về JSON state mới nhất."""
    node_ref = None  # set bởi AeroScriptVisionNode

    def log_message(self, fmt, *args):
        pass  # tắt log mỗi request

    def do_GET(self):
        if self.node_ref is None:
            self.send_error(503); return
        d    = self.node_ref.get_state()
        body = json.dumps(d).encode()
        self.send_response(200)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def _start_json_server(node, port: int = 8081):
    _JSONHandler.node_ref = node
    srv = ThreadingHTTPServer(("0.0.0.0", port), _JSONHandler)
    t   = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


class _StreamHandler(BaseHTTPRequestHandler):
    """Handler phục vụ stream MJPEG trực tiếp từ camera."""
    node_ref = None

    def log_message(self, fmt, *args):
        pass  # tắt log request stream liên tục

    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                last_frame = None
                while True:
                    if self.node_ref and self.node_ref.latest_jpeg is not None:
                        if self.node_ref.latest_jpeg is not last_frame:
                            last_frame = self.node_ref.latest_jpeg
                            self.wfile.write(b"--frame\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(last_frame)))
                            self.end_headers()
                            self.wfile.write(last_frame)
                            self.wfile.write(b"\r\n")
                    time.sleep(0.033)  # ~30 FPS
            except Exception:
                pass
        else:
            self.send_error(404)


def _start_stream_server(node, port: int = 8080):
    _StreamHandler.node_ref = node
    srv = ThreadingHTTPServer(("0.0.0.0", port), _StreamHandler)
    t   = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv

# ══════════════════════════════════════════════════════════════════════════════
# 4. Geometry
# ══════════════════════════════════════════════════════════════════════════════
PEN_3D = np.array([
    [0.0,   0.0, 0.0],
    [0.0,  64.0, 0.0],
    [-11.5, 44.0, 0.0],
    [11.5,  44.0, 0.0],
], dtype=np.float32)

CAM_MTX = np.array([[770,0,320],[0,770,240],[0,0,1]], dtype=np.float32)
DIST    = np.zeros((4,1), dtype=np.float32)


def get_local_ip() -> str:
    """Lấy địa chỉ IP cục bộ thực tế của máy."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

# ══════════════════════════════════════════════════════════════════════════════
# 5. ROS2 Node
# ══════════════════════════════════════════════════════════════════════════════
class AeroScriptVisionNode(Node):
    def __init__(self, model_path: str, conf: float, cam_topic: str):
        super().__init__("aeroscript_vision")
        self.bridge = CvBridge()
        self.model  = TFLitePoseInference(model_path, conf_thresh=conf)
        self.kf     = PoseKalmanFilter()
        self.lost   = 0
        self.RESET  = 10
        self._fps_t = time.time()
        self._fps   = 0.0

        # Publishers
        self.pub_img  = self.create_publisher(Image,       "/aeroscript/pen_image", 1)
        self.pub_xyz  = self.create_publisher(Point,        "/aeroscript/pen_xyz",   1)
        self.pub_pose = self.create_publisher(PoseStamped,  "/aeroscript/pen_pose",  1)

        # Subscribe camera
        self.sub = self.create_subscription(
            Image, cam_topic, self._cb, 1)

        # State chia sẻ với JSON server
        self._state_lock = threading.Lock()
        self._state = {"x": None, "y": None, "z": None,
                       "detected": False, "fps": 0.0}

        # Khởi động HTTP JSON server và MJPEG stream server
        self.latest_jpeg = None
        _start_json_server(self, port=8081)
        _start_stream_server(self, port=8080)

        local_ip = get_local_ip()
        self.get_logger().info(f"✅ Node sẵn sàng — subscribe: {cam_topic}")
        self.get_logger().info(
            f"📺 Video : http://{local_ip}:8080/stream?topic=/aeroscript/pen_image&type=mjpeg"
        )
        self.get_logger().info(
            f"📊 Data  : http://{local_ip}:8081/"
        )

    def get_state(self) -> dict:
        """Thread-safe — trả về state mới nhất cho JSON server."""
        with self._state_lock:
            return dict(self._state)

    def _cb(self, msg: Image):
        # ROS Image → OpenCV BGR
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"CvBridge: {e}"); return

        kpts, kv = self.model(frame)
        valid = (kpts is not None and kv is not None
                 and np.all(kv > 0.45) and not np.any(kpts == 0.0))

        x = y = z = None

        if valid:
            ok, rvec, tvec = cv2.solvePnP(
                PEN_3D, kpts.astype(np.float64),
                CAM_MTX, DIST, flags=cv2.SOLVEPNP_IPPE)

            if ok and tvec[2][0] > 0:
                tf = self.kf.update(tvec)
                x, y, z = float(tf[0][0]), float(tf[1][0]), float(tf[2][0])
                self.lost = 0
                with self._state_lock:
                    self._state = {"x": x, "y": y, "z": z,
                                   "detected": True, "fps": self._fps}

                # Vẽ lên frame
                for px, py in kpts.astype(int):
                    cv2.circle(frame, (px,py), 5, (0,255,0), -1)
                cv2.drawFrameAxes(frame, CAM_MTX, DIST, rvec, tf, 30)
                cv2.putText(frame, f"X:{x:.0f} Y:{y:.0f} Z:{z:.0f}mm",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

                # Publish tọa độ
                now = self.get_clock().now().to_msg()

                pt = Point(); pt.x=x; pt.y=y; pt.z=z
                self.pub_xyz.publish(pt)

                ps = PoseStamped()
                ps.header = Header(); ps.header.stamp=now; ps.header.frame_id="camera"
                ps.pose.position.x=x/1000.0  # mm → m
                ps.pose.position.y=y/1000.0
                ps.pose.position.z=z/1000.0
                ps.pose.orientation.w=1.0
                self.pub_pose.publish(ps)
            else:
                valid = False

        if not valid:
            self.lost += 1
            if self.lost >= self.RESET:
                self.kf.reset()
                with self._state_lock:
                    self._state["detected"] = False

        # FPS
        t_now = time.time()
        fps_i = 1.0 / max(t_now - self._fps_t, 1e-6)
        self._fps = 0.9*self._fps + 0.1*fps_i
        self._fps_t = t_now
        with self._state_lock:
            self._state["fps"] = round(self._fps, 1)

        cv2.putText(frame, f"FPS:{self._fps:.1f}",
                    (frame.shape[1]-110, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200,200,0), 2)
        cv2.putText(frame, "TRACKING" if valid else "NO TARGET",
                    (frame.shape[1]-135, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,100) if valid else (0,0,200), 2)

        # Encode frame sang JPEG để phục vụ HTTP stream
        try:
            _, jpeg = cv2.imencode('.jpg', frame)
            self.latest_jpeg = jpeg.tobytes()
        except Exception as e:
            self.get_logger().error(f"JPEG encode: {e}")

        # Publish annotated image → web_video_server sẽ phát lên HTTP
        try:
            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            out_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_img.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Publish image: {e}")

        if valid:
            self.get_logger().info(
                f"PEN  X:{x:7.1f}  Y:{y:7.1f}  Z:{z:7.1f} mm  FPS:{self._fps:.1f}")


def resolve_model_path(model_arg: str) -> str:
    """Tự động tìm kiếm các đường dẫn mô hình TFLite khả dụng nếu file truyền vào không tồn tại."""
    if os.path.exists(model_arg):
        return model_arg

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best_saved_model/best_int8.tflite'),
        os.path.join(script_dir, 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best_int8.tflite'),
        os.path.join(script_dir, 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best_saved_model/best_float16.tflite'),
        os.path.join(script_dir, 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best_float16.tflite'),
        os.path.join(script_dir, 'best_float16.tflite'),
        os.path.join(script_dir, 'best_int8.tflite'),
    ]

    for path in candidates:
        if os.path.exists(path):
            print(f"🔍 Found TFLite model at: {path}")
            return path

    return model_arg

# ══════════════════════════════════════════════════════════════════════════════
# 6. Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_float16.tflite")
    parser.add_argument("--conf",  type=float, default=0.55)
    parser.add_argument("--cam-topic", default="/image_raw",
                        help="ROS2 topic camera (mặc định /image_raw)")
    # Bỏ qua các arg --ros-args của ROS2
    args, _ = parser.parse_known_args()

    model_path = resolve_model_path(args.model)

    rclpy.init()
    node = AeroScriptVisionNode(model_path, args.conf, args.cam_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()