"""
view_stream.py  —  Chạy trên Laptop
=====================================
Nhận MJPEG stream + dữ liệu XYZ từ Pi (ROS2),
hiển thị Dashboard giống hệt test_webcam.py.

Kiến trúc:
  Video : MJPEG ← web_video_server :8080  (/aeroscript/pen_image)
  Data  : JSON  ← HTTP nhỏ tích hợp trong run_pi4_ros.py :8081

Cài trên Laptop:
    pip install numpy opencv-python

Chạy trên Pi (3 terminal):
    T1: ros2 run usb_cam usb_cam_node_exe --ros-args \
            -p video_device:="/dev/video0" -p image_width:=640 \
            -p image_height:=480 -p pixel_format:="yuyv" \
            -p auto_exposure:=true -p brightness:=128
    T2: ros2 run web_video_server web_video_server
    T3: python3 run_pi4_ros.py --model best_float16.tflite

Chạy trên Laptop:
    python3 view_stream.py --pi 192.168.50.1

Phím tắt:
    q  — thoát
    c  — xoá trail + history
    s  — chụp ảnh màn hình
"""

import argparse
import time
import threading
import json
import os
from collections import deque
from urllib.request import urlopen
from urllib.error import URLError
import numpy as np
import cv2


# ══════════════════════════════════════════════════════════════════════════════
# 1. Đọc dữ liệu XYZ từ JSON endpoint (port 8081) — background thread
# ══════════════════════════════════════════════════════════════════════════════
class XYZDataClient:
    """
    Poll JSON từ http://<pi>:8081/  mỗi 50ms.
    Trả về dict {"x", "y", "z", "detected", "fps"}.
    Thread-safe.
    """
    POLL_INTERVAL = 0.05  # 20 Hz

    def __init__(self, url: str):
        self._url  = url
        self._lock = threading.Lock()
        self._data = {"x": None, "y": None, "z": None,
                      "detected": False, "fps": 0.0}
        self._connected = False
        self._running   = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            try:
                with urlopen(self._url, timeout=1.0) as resp:
                    d = json.loads(resp.read().decode())
                with self._lock:
                    self._data      = d
                    self._connected = True
            except (URLError, Exception):
                with self._lock:
                    self._connected = False
            time.sleep(self.POLL_INTERVAL)

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    def stop(self):
        self._running = False


# ══════════════════════════════════════════════════════════════════════════════
# 2. MJPEG reader — đọc frame từ web_video_server
# ══════════════════════════════════════════════════════════════════════════════
class MJPEGReader:
    """
    Đọc MJPEG stream bằng urllib thủ công — không phụ thuộc cv2 codec.
    Chạy trong background thread, main thread gọi read() lấy frame mới nhất.
    """
    _SOI = bytes([0xFF, 0xD8])  # JPEG Start Of Image
    _EOI = bytes([0xFF, 0xD9])  # JPEG End Of Image

    def __init__(self, url: str):
        self._url     = url
        self._lock    = threading.Lock()
        self._frame   = None
        self._ok      = False
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        from urllib.request import urlopen
        while self._running:
            try:
                with urlopen(self._url, timeout=10) as resp:
                    self._ok = True
                    buf = b""
                    while self._running:
                        chunk = resp.read(1024)
                        if not chunk:
                            break
                        buf += chunk
                        # Tìm JPEG SOI (FF D8) và EOI (FF D9)
                        # Giữ buffer < 2MB tránh memory leak
                        if len(buf) > 2 * 1024 * 1024:
                            s_trim = buf.rfind(self._SOI)
                            if s_trim > 0:
                                buf = buf[s_trim:]
                        s = buf.find(self._SOI)
                        if s == -1:
                            continue
                        e = buf.find(self._EOI, s)
                        if e == -1:
                            continue
                        jpg   = buf[s : e + 2]
                        buf   = buf[e + 2:]
                        arr   = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                self._frame = frame
            except Exception as ex:
                self._ok = False
                print(f"[MJPEGReader] {ex} — retry...")
                time.sleep(1.0)

    def read(self):
        with self._lock:
            f = self._frame
        if f is None:
            return False, None
        return True, f.copy()

    @property
    def connected(self) -> bool:
        return self._ok

    def release(self):
        self._running = False


# ══════════════════════════════════════════════════════════════════════════════
# 3. Dashboard — giữ nguyên 100% từ test_webcam.py
# ══════════════════════════════════════════════════════════════════════════════
class Dashboard:
    PANEL_W     = 420
    HISTORY_LEN = 120
    TRAIL_LEN   = 500

    COL_BG     = (25,  25,  30)
    COL_GRID   = (50,  50,  55)
    COL_X      = (80,  180, 255)
    COL_Y      = (80,  255, 120)
    COL_Z      = (100, 100, 255)
    COL_TRAIL  = (255, 200, 80)
    COL_LABEL  = (140, 140, 150)
    COL_ACCENT = (0,   200, 255)

    def __init__(self, cam_h: int = 480):
        self.cam_h = cam_h
        self.x_hist = deque(maxlen=self.HISTORY_LEN)
        self.y_hist = deque(maxlen=self.HISTORY_LEN)
        self.z_hist = deque(maxlen=self.HISTORY_LEN)
        self.trail  = deque(maxlen=self.TRAIL_LEN)
        self.sample_count = 0

    def push(self, x: float, y: float, z: float):
        self.x_hist.append(x)
        self.y_hist.append(y)
        self.z_hist.append(z)
        self.trail.append((x, y))
        self.sample_count += 1

    def clear(self):
        self.x_hist.clear(); self.y_hist.clear(); self.z_hist.clear()
        self.trail.clear(); self.sample_count = 0

    def render(self, cx, cy, cz, has_det: bool,
               fps_pi: float = 0.0, ros_ok: bool = False) -> np.ndarray:
        p = np.full((self.cam_h, self.PANEL_W, 3), self.COL_BG, dtype=np.uint8)

        # ── Viền + tiêu đề ──────────────────────────────────────────────────
        cv2.line(p, (0, 0), (0, self.cam_h), self.COL_ACCENT, 2)
        cv2.putText(p, "AeroScript Dashboard", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COL_ACCENT, 2)
        cv2.line(p, (15, 38), (self.PANEL_W - 15, 38), self.COL_GRID, 1)

        # ── Trạng thái kết nối ──────────────────────────────────────────────
        conn_col = (0, 220, 80) if ros_ok else (60, 60, 200)
        conn_txt = "Pi OK" if ros_ok else "Pi OFF"
        cv2.putText(p, conn_txt, (self.PANEL_W - 90, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, conn_col, 1)

        # ── Readout số ──────────────────────────────────────────────────────
        y0 = 65
        sc, st = ((0, 255, 100), "TRACKING") if has_det else ((0, 0, 200), "NO TARGET")
        cv2.putText(p, st, (15, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc, 2)
        cv2.putText(p, f"Samples: {self.sample_count}  Pi FPS:{fps_pi:.1f}",
                    (170, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COL_LABEL, 1)
        y0 += 30
        self._readout(p, "X (Ngang)", cx, "mm", self.COL_X,  15, y0); y0 += 25
        self._readout(p, "Y (Doc)  ", cy, "mm", self.COL_Y,  15, y0); y0 += 25
        self._readout(p, "Z (Sau)  ", cz, "mm", self.COL_Z,  15, y0)
        cv2.line(p, (15, y0 + 12), (self.PANEL_W - 15, y0 + 12), self.COL_GRID, 1)

        # ── Time-series chart ────────────────────────────────────────────────
        ct = y0 + 25
        cv2.putText(p, "Time-Series (X, Y, Z)", (15, ct),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COL_LABEL, 1)
        ct += 8
        ch = 130; cl = 15; cr = self.PANEL_W - 15; cw = cr - cl
        cv2.rectangle(p, (cl, ct), (cr, ct + ch), (35, 35, 40), -1)
        mid = ct + ch // 2
        cv2.line(p, (cl, mid), (cr, mid), self.COL_GRID, 1)
        cv2.line(p, (cl, ct + ch // 4),     (cr, ct + ch // 4),     (40, 40, 42), 1)
        cv2.line(p, (cl, ct + 3 * ch // 4), (cr, ct + 3 * ch // 4), (40, 40, 42), 1)

        if len(self.x_hist) > 1:
            av   = list(self.x_hist) + list(self.y_hist) + list(self.z_hist)
            vmin, vmax = min(av), max(av)
            vr   = max(vmax - vmin, 20.0)
            vc   = (vmax + vmin) / 2.0
            cv2.putText(p, f"{vc+vr/2:.0f}", (cr+2, ct+12),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, self.COL_LABEL, 1)
            cv2.putText(p, f"{vc-vr/2:.0f}", (cr+2, ct+ch),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, self.COL_LABEL, 1)
            for hist, col in [(self.x_hist, self.COL_X),
                               (self.y_hist, self.COL_Y),
                               (self.z_hist, self.COL_Z)]:
                self._series(p, hist, cl, ct, cw, ch, vc, vr, col)

        ly = ct + ch + 15
        for i, (lb, col) in enumerate([("X", self.COL_X),
                                        ("Y", self.COL_Y),
                                        ("Z", self.COL_Z)]):
            lx = 15 + i * 90
            cv2.line(p, (lx, ly - 4), (lx + 20, ly - 4), col, 2)
            cv2.putText(p, lb, (lx + 25, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        cv2.line(p, (15, ly + 12), (self.PANEL_W - 15, ly + 12), self.COL_GRID, 1)

        # ── 2D Trajectory ────────────────────────────────────────────────────
        top = ly + 25
        cv2.putText(p, "2D Trajectory (X-Y)", (15, top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COL_LABEL, 1)
        top += 8
        cs  = min(self.PANEL_W - 30, self.cam_h - top - 10)
        cs  = max(cs, 80)
        cb  = top + cs; crr = 15 + cs
        cv2.rectangle(p, (15, top), (crr, cb), (35, 35, 40), -1)
        cmx = 15 + cs // 2; cmy = top + cs // 2
        cv2.line(p, (cmx, top), (cmx, cb),  self.COL_GRID, 1)
        cv2.line(p, (15,  cmy), (crr, cmy), self.COL_GRID, 1)

        if len(self.trail) > 1:
            ta = np.array(list(self.trail))
            txn, txx = ta[:, 0].min(), ta[:, 0].max()
            tyn, tyx = ta[:, 1].min(), ta[:, 1].max()
            tr  = max(txx - txn, tyx - tyn, 20.0)
            tcx = (txx + txn) / 2.0
            tcy = (tyx + tyn) / 2.0
            ds  = cs - 20
            pts = []
            for px2, py2 in self.trail:
                sx = int(cmx + (px2 - tcx) / tr * ds)
                sy = int(cmy + (py2 - tcy) / tr * ds)
                pts.append((int(np.clip(sx, 17, crr - 2)),
                             int(np.clip(sy, top + 2, cb - 2))))
            for i in range(1, len(pts)):
                a    = int(80 + 175 * i / len(pts))
                col2 = (int(self.COL_TRAIL[0] * a / 255),
                        int(self.COL_TRAIL[1] * a / 255),
                        int(self.COL_TRAIL[2] * a / 255))
                cv2.line(p, pts[i - 1], pts[i], col2, 1, cv2.LINE_AA)
            cv2.circle(p, pts[-1], 4, (0, 255, 255), -1)

        cv2.putText(p, "X ->", (crr - 35, cb + 12),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, self.COL_LABEL, 1)
        cv2.putText(p, "Y", (13, top - 3),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, self.COL_LABEL, 1)
        return p

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _readout(self, img, lbl, val, unit, col, x, y):
        cv2.putText(img, f"{lbl}:", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COL_LABEL, 1)
        if val is not None:
            cv2.putText(img, f"{val:8.1f} {unit}", (x + 130, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        else:
            cv2.putText(img, "  ---", (x + 130, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)

    def _series(self, img, data, left, top, w, h, center, vrange, color):
        if len(data) < 2:
            return
        pts = []
        for i, v in enumerate(data):
            px = int(left + i * w / (self.HISTORY_LEN - 1))
            py = int(np.clip(top + h / 2 - (v - center) / vrange * h,
                             top + 1, top + h - 1))
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Blank "waiting" frame
# ══════════════════════════════════════════════════════════════════════════════
def make_blank(pi_ip: str, video_url: str) -> np.ndarray:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    lines = [
        ("Dang ket noi Pi...", (140, 200), 0.9, (100, 200, 100)),
        (f"IP : {pi_ip}",      (140, 245), 0.6, (80,  150, 80)),
        (f"{video_url[:55]}", (20,  280), 0.4, (60,  120, 60)),
        ("Kiem tra Pi da chay run_pi4_ros.py chua?",
                               (20,  320), 0.45, (80, 80, 200)),
    ]
    for txt, pos, sc, col in lines:
        cv2.putText(blank, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1)
    return blank


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="AeroScript Laptop Viewer")
    parser.add_argument("--pi",         default="192.168.50.1",
                        help="IP Raspberry Pi (mặc định 192.168.50.1)")
    parser.add_argument("--video-port", type=int, default=8080,
                        help="Port web_video_server (mặc định 8080)")
    parser.add_argument("--data-port",  type=int, default=8081,
                        help="Port JSON data từ run_pi4_ros.py (mặc định 8081)")
    parser.add_argument("--topic",      default="/aeroscript/pen_image",
                        help="ROS2 image topic")
    parser.add_argument("--quality",    type=int, default=70,
                        help="JPEG quality 1-95 (mặc định 70)")
    args = parser.parse_args()

    video_url = (f"http://{args.pi}:{args.video_port}/stream"
                 f"?topic={args.topic}&type=mjpeg&quality={args.quality}")
    data_url  = f"http://{args.pi}:{args.data_port}/"

    print(f"\n{'─'*58}")
    print(f"  AeroScript Remote Viewer")
    print(f"{'─'*58}")
    print(f"  Video  : {video_url}")
    print(f"  Data   : {data_url}")
    print(f"  Phím   : q=thoát  c=xoá trail  s=screenshot")
    print(f"{'─'*58}\n")

    stream = MJPEGReader(video_url)
    data   = XYZDataClient(data_url)
    dash   = Dashboard(cam_h=480)
    blank  = make_blank(args.pi, video_url)

    screenshot_dir = os.path.expanduser("~/aeroscript_screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    # Chờ frame đầu tiên tối đa 8 giây
    print("⏳ Chờ frame từ Pi...", end="", flush=True)
    for _ in range(80):
        ret, _ = stream.read()
        if ret:
            print(" OK!")
            break
        time.sleep(0.1)
        print(".", end="", flush=True)
    else:
        print()
        print("⚠️  Không nhận được frame. Kiểm tra:")
        print(f"   1. Pi đã chạy run_pi4_ros.py chưa?")
        print(f"   2. web_video_server đang chạy chưa?")
        print(f"   3. curl http://{args.pi}:{args.video_port}/ có trả về gì không?")
        print("   Vẫn tiếp tục — hiển thị blank frame...")

    # FPS đo phía laptop (tốc độ render)
    t_prev    = time.time()
    fps_local = 0.0

    # Tạo window trước, đặt kích thước cố định
    cv2.namedWindow("AeroScript — Remote View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AeroScript — Remote View", 640 + 420, 480)

    while True:
        # ── Đọc frame ────────────────────────────────────────────────────────
        ret, frame = stream.read()
        if not ret or frame is None:
            frame = blank.copy()
        else:
            if frame.shape[:2] != (480, 640):
                frame = cv2.resize(frame, (640, 480))

        # ── Lấy dữ liệu XYZ ─────────────────────────────────────────────────
        d        = data.get()
        x        = d.get("x")
        y        = d.get("y")
        z        = d.get("z")
        detected = d.get("detected", False)
        fps_pi   = d.get("fps", 0.0)
        ros_ok   = data.connected

        if detected and x is not None:
            dash.push(x, y, z)

        # ── FPS laptop ───────────────────────────────────────────────────────
        t_now     = time.time()
        fps_local = 0.9 * fps_local + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev    = t_now
        cv2.putText(frame, f"Local FPS:{fps_local:.1f}",
                    (10, frame.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

        # ── Render + ghép ────────────────────────────────────────────────────
        panel    = dash.render(x, y, z, detected,
                               fps_pi=fps_pi, ros_ok=ros_ok)
        combined = np.hstack([frame, panel])

        cv2.imshow("AeroScript — Remote View", combined)

        # ── Phím tắt ─────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            dash.clear()
            print("🗑️  Đã xoá trail và history.")
        elif key == ord("s"):
            fname = os.path.join(screenshot_dir,
                                 f"aeroscript_{int(time.time())}.png")
            cv2.imwrite(fname, combined)
            print(f"📸 Screenshot: {fname}")

    data.stop()
    stream.release()
    cv2.destroyAllWindows()
    print("👋 Đã thoát.")


if __name__ == "__main__":
    main()