#!/usr/bin/env python3
"""
Standalone 4DOF controller for your 6DOF-wired system (7 servos incl. shoulder mirror).
- Direct PCA9685 control (smbus2) + optional TCA9548A mux
- Controls 4DOF: base, shoulder (pair), elbow, wrist_pitch
- Optionally holds other channels at fixed angles via ROS params (fixed_channels/fixed_degs)
- Optionally turns off channels (off_channels) via PCA9685 full-off
- Includes per-joint sign and offset calibration
- Verbose logging prints:
  - target (cm)
  - calc_deg from old IK
  - out_deg after sign+offset for each servo (incl shoulder mirror)
  - fixed channel angles

Run example (your wiring):
  CH0 Base
  CH1/CH2 Shoulder pair (mirrored)
  CH3 Elbow
  CH4 Wrist_roll (fixed or off)
  CH5 Wrist_pitch
  CH6 Pen (fixed or off)

  python3 wicom_roboarm_4dof_standalone.py --ros-args \
    -p ch_base:=0 \
    -p ch_shoulder:=1 -p ch_shoulder_mirror:=2 -p shoulder_mirror_enabled:=true -p shoulder_mirror_angle_max:=180.0 \
    -p ch_elbow:=3 \
    -p ch_wrist_pitch:=5 \
    -p fixed_channels:="[4,6]" -p fixed_degs:="[90.0,90.0]" \
    -p sign_elbow:=-1.0 -p sign_wrist:=-1.0 \
    -p auto_draw:=true -p auto_loop:=true
"""

import math
import time
import threading

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from geometry_msgs.msg import Point
from smbus2 import SMBus

# PCA9685 registers
MODE1 = 0x00
MODE2 = 0x01
PRESCALE = 0xFE
LED0_ON_L = 0x06
ALL_LED_ON_L = 0xFA
ALL_LED_ON_H = 0xFB
ALL_LED_OFF_L = 0xFC
ALL_LED_OFF_H = 0xFD

RESTART = 0x80
SLEEP = 0x10
ALLCALL = 0x01
OUTDRV = 0x04


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class PCA9685:
    def __init__(
        self,
        bus: SMBus,
        address: int,
        oscillator_hz: int = 25_000_000,
        mux_addr: int | None = None,
        mux_channel: int | None = None,
    ):
        self.bus = bus
        self.addr = int(address)
        self.osc = int(oscillator_hz)
        self.mux_addr = None if mux_addr is None else int(mux_addr)
        self.mux_channel = None if mux_channel is None else int(mux_channel)
        self._init_device()

    def _select_mux(self):
        if self.mux_addr is None or self.mux_channel is None:
            return
        if not (0 <= int(self.mux_channel) <= 7):
            raise ValueError("mux_channel must be 0..7")
        # TCA9548A: write 1<<channel
        self.bus.write_byte(int(self.mux_addr), 1 << int(self.mux_channel))
        time.sleep(0.001)

    def _write8(self, reg, val):
        self._select_mux()
        self.bus.write_byte_data(self.addr, reg, val & 0xFF)

    def _read8(self, reg):
        self._select_mux()
        return self.bus.read_byte_data(self.addr, reg)

    def _init_device(self):
        self._write8(MODE2, OUTDRV)
        self._write8(MODE1, ALLCALL)
        time.sleep(0.005)
        mode1 = self._read8(MODE1) & ~SLEEP
        self._write8(MODE1, mode1)
        time.sleep(0.005)

    def set_pwm_freq(self, freq_hz: float):
        prescaleval = float(self.osc) / (4096.0 * float(freq_hz)) - 1.0
        prescale = int(math.floor(prescaleval + 0.5))
        oldmode = self._read8(MODE1)
        newmode = (oldmode & 0x7F) | SLEEP
        self._write8(MODE1, newmode)
        self._write8(PRESCALE, prescale)
        self._write8(MODE1, oldmode)
        time.sleep(0.005)
        self._write8(MODE1, oldmode | RESTART)

    def set_pwm_counts(self, channel: int, on: int, off: int):
        base = LED0_ON_L + 4 * int(channel)
        self._write8(base + 0, on & 0xFF)
        self._write8(base + 1, (on >> 8) & 0x0F)
        self._write8(base + 2, off & 0xFF)
        self._write8(base + 3, (off >> 8) & 0x0F)

    def set_off(self, channel: int):
        base = LED0_ON_L + 4 * int(channel)
        self._write8(base + 0, 0x00)
        self._write8(base + 1, 0x00)
        self._write8(base + 2, 0x00)
        self._write8(base + 3, 0x10)  # full off

    def set_all_off(self):
        self._write8(ALL_LED_ON_L, 0)
        self._write8(ALL_LED_ON_H, 0)
        self._write8(ALL_LED_OFF_L, 0)
        self._write8(ALL_LED_OFF_H, 0x10)


class RoboArm4DOFStandalone(Node):
    def __init__(self):
        super().__init__("wicom_roboarm_4dof_standalone")

        desc_int_arr = ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER_ARRAY)
        desc_dbl_arr = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY)

        # --- I2C/PCA ---
        self.declare_parameter("i2c_bus", 1)
        self.declare_parameter("pca_address", 0x40)
        self.declare_parameter("oscillator_hz", 25_000_000)
        self.declare_parameter("pwm_frequency_hz", 50.0)

        self.declare_parameter("use_mux", True)
        self.declare_parameter("mux_address", 0x70)
        self.declare_parameter("mux_channel", 2)

        self.declare_parameter("pulse_us_min", 500.0)
        self.declare_parameter("pulse_us_max", 2500.0)
        self.declare_parameter("period_us", 20000.0)

        # --- channels (your wiring defaults) ---
        self.declare_parameter("ch_base", 0)
        self.declare_parameter("ch_shoulder", 1)
        self.declare_parameter("ch_shoulder_mirror", 2)
        self.declare_parameter("ch_elbow", 3)
        self.declare_parameter("ch_wrist_pitch", 5)

        # shoulder mirror
        self.declare_parameter("shoulder_mirror_enabled", True)
        self.declare_parameter("shoulder_mirror_angle_max", 180.0)

        # --- calibration knobs ---
        self.declare_parameter("home_deg", 90.0)

        # +1 normal, -1 invert around home
        self.declare_parameter("sign_base", 1.0)
        self.declare_parameter("sign_shoulder", 1.0)
        self.declare_parameter("sign_elbow", 1.0)
        self.declare_parameter("sign_wrist", 1.0)

        # direct add to output degrees
        self.declare_parameter("offset_base_deg", 0.0)
        self.declare_parameter("offset_shoulder_deg", 0.0)
        self.declare_parameter("offset_elbow_deg", 0.0)
        self.declare_parameter("offset_wrist_deg", 0.0)

        # logging
        self.declare_parameter("verbose_log", True)

        # IMPORTANT: typed arrays must NOT default to [] (can become BYTE_ARRAY).
        # Use dummy defaults and strip them when reading.
        self.declare_parameter("fixed_channels", [-1], desc_int_arr)
        self.declare_parameter("fixed_degs", [-1.0], desc_dbl_arr)
        self.declare_parameter("off_channels", [-1], desc_int_arr)

        # --- old 4DOF params (from your backup node) ---
        self.declare_parameter("L1_cm", 6.0)
        self.declare_parameter("L2_cm", 5.5)
        self.declare_parameter("L3_cm", 5.5)

        self.declare_parameter("elbow_up", True)
        self.declare_parameter("z_offset_cm", 0.0)
        self.declare_parameter("zPlane_cm_default", 15.0)
        self.declare_parameter("use_yspan_as_draw_area", True)

        self.declare_parameter("send_interval_sec", 0.1)

        self.declare_parameter("base_scale", 1.0)
        self.declare_parameter("shoulder_scale", 1.0)
        self.declare_parameter("elbow_scale", 1.0)
        self.declare_parameter("wrist_scale", 1.0)

        self.declare_parameter("base_offset_deg", 90.0)
        self.declare_parameter("shoulder_offset_deg", 90.0)
        self.declare_parameter("elbow_offset_deg", 180.0)
        self.declare_parameter("wrist_offset_deg", 100.0)

        # --- auto draw ---
        self.declare_parameter("auto_draw", True)
        self.declare_parameter("auto_loop", True)
        self.declare_parameter("auto_point_interval", 0.1)
        self.declare_parameter("auto_square_side_uv", 0.1)
        self.declare_parameter("auto_points_per_side", 30)

        self.declare_parameter("input_uv_topic", "/input_uv")

        # ---- Read params ----
        self.i2c_bus = int(self.get_parameter("i2c_bus").value)
        self.pca_addr = int(self.get_parameter("pca_address").value)
        self.osc = int(self.get_parameter("oscillator_hz").value)
        self.pwm_hz = float(self.get_parameter("pwm_frequency_hz").value)

        self.use_mux = bool(self.get_parameter("use_mux").value)
        self.mux_addr = int(self.get_parameter("mux_address").value)
        self.mux_channel = int(self.get_parameter("mux_channel").value)

        self.pulse_us_min = float(self.get_parameter("pulse_us_min").value)
        self.pulse_us_max = float(self.get_parameter("pulse_us_max").value)
        self.period_us = float(self.get_parameter("period_us").value)

        self.ch_base = int(self.get_parameter("ch_base").value)
        self.ch_shoulder = int(self.get_parameter("ch_shoulder").value)
        self.ch_shoulder_mirror = int(self.get_parameter("ch_shoulder_mirror").value)
        self.ch_elbow = int(self.get_parameter("ch_elbow").value)
        self.ch_wrist = int(self.get_parameter("ch_wrist_pitch").value)

        self.shoulder_mirror_enabled = bool(self.get_parameter("shoulder_mirror_enabled").value)
        self.shoulder_mirror_angle_max = float(self.get_parameter("shoulder_mirror_angle_max").value)

        self.home = float(self.get_parameter("home_deg").value)

        self.sign_base = float(self.get_parameter("sign_base").value)
        self.sign_sh = float(self.get_parameter("sign_shoulder").value)
        self.sign_el = float(self.get_parameter("sign_elbow").value)
        self.sign_wr = float(self.get_parameter("sign_wrist").value)

        self.offset_base = float(self.get_parameter("offset_base_deg").value)
        self.offset_sh = float(self.get_parameter("offset_shoulder_deg").value)
        self.offset_el = float(self.get_parameter("offset_elbow_deg").value)
        self.offset_wr = float(self.get_parameter("offset_wrist_deg").value)

        self.verbose_log = bool(self.get_parameter("verbose_log").value)

        # fixed map (strip dummy)
        fixed_ch = list(self.get_parameter("fixed_channels").value)
        fixed_deg = list(self.get_parameter("fixed_degs").value)

        if fixed_ch == [-1] and fixed_deg == [-1.0]:
            fixed_ch, fixed_deg = [], []
        elif fixed_ch == [-1] or fixed_deg == [-1.0]:
            raise RuntimeError("fixed_channels/fixed_degs: set both or none")

        if len(fixed_ch) != len(fixed_deg):
            raise RuntimeError("fixed_channels and fixed_degs must have the same length")

        self.fixed_map = {int(ch): float(deg) for ch, deg in zip(fixed_ch, fixed_deg)}

        # off channels (strip dummy)
        off_ch = list(self.get_parameter("off_channels").value)
        if off_ch == [-1]:
            off_ch = []
        self.off_channels = [int(x) for x in off_ch]

        # old IK params
        self.L1 = float(self.get_parameter("L1_cm").value)
        self.L2 = float(self.get_parameter("L2_cm").value)
        self.L3 = float(self.get_parameter("L3_cm").value)

        self.elbow_up = bool(self.get_parameter("elbow_up").value)
        self.z_offset_cm = float(self.get_parameter("z_offset_cm").value)
        self.zPlane_default = float(self.get_parameter("zPlane_cm_default").value)
        self.use_yspan = bool(self.get_parameter("use_yspan_as_draw_area").value)

        self.send_interval = float(self.get_parameter("send_interval_sec").value)

        self.base_scale = float(self.get_parameter("base_scale").value)
        self.shoulder_scale = float(self.get_parameter("shoulder_scale").value)
        self.elbow_scale = float(self.get_parameter("elbow_scale").value)
        self.wrist_scale = float(self.get_parameter("wrist_scale").value)

        self.base_off = float(self.get_parameter("base_offset_deg").value)
        self.sh_off = float(self.get_parameter("shoulder_offset_deg").value)
        self.el_off = float(self.get_parameter("elbow_offset_deg").value)
        self.wr_off = float(self.get_parameter("wrist_offset_deg").value)

        self.auto_draw = bool(self.get_parameter("auto_draw").value)
        self.auto_loop = bool(self.get_parameter("auto_loop").value)
        self.auto_point_interval = float(self.get_parameter("auto_point_interval").value)
        self.auto_square_side_uv = float(self.get_parameter("auto_square_side_uv").value)
        self.auto_points_per_side = int(self.get_parameter("auto_points_per_side").value)

        self.uv_topic = str(self.get_parameter("input_uv_topic").value)

        # ---- I2C init ----
        self.lock = threading.Lock()
        self.bus = SMBus(self.i2c_bus)
        self.pca = PCA9685(
            self.bus,
            self.pca_addr,
            oscillator_hz=self.osc,
            mux_addr=(self.mux_addr if self.use_mux else None),
            mux_channel=(self.mux_channel if self.use_mux else None),
        )
        self.pca.set_pwm_freq(self.pwm_hz)

        # OFF channels at startup (optional)
        for ch in self.off_channels:
            try:
                self.pca.set_off(ch)
            except Exception:
                pass

        # ---- state ----
        self._zPlane_cm = self.zPlane_default + self.z_offset_cm
        self._have_uv = False
        self._last_u = 0.5
        self._last_v = 0.5
        self._last_pen_down = True
        self._last_send_time = 0.0

        # auto square path
        self._auto_path = []
        self._auto_idx = 0
        self._auto_next_t = time.monotonic()

        # ROS I/O
        self.sub_uv = self.create_subscription(Point, self.uv_topic, self._on_uv, 10)
        self.pub_debug_target = self.create_publisher(Point, "/debug_target_cm_4dof", 10)
        self.timer = self.create_timer(0.02, self._tick)

        self.home_pose()

        if self.auto_draw:
            self._build_auto_square()

        self.get_logger().info(
            "4DOF standalone ready | "
            f"4DOF CH: base={self.ch_base} shoulder={self.ch_shoulder}/{self.ch_shoulder_mirror} "
            f"elbow={self.ch_elbow} wrist_pitch={self.ch_wrist} | "
            f"fixed={self.fixed_map} off={self.off_channels} verbose_log={self.verbose_log}"
        )

    # ---------- servo helpers ----------
    def _pulse_us_for_angle(self, deg: float) -> float:
        deg = clamp(float(deg), 0.0, 180.0)
        return self.pulse_us_min + (deg / 180.0) * (self.pulse_us_max - self.pulse_us_min)

    def _counts_from_pulse_us(self, us: float) -> int:
        if self.period_us <= 0:
            self.period_us = 20000.0
        counts = int(round((float(us) / float(self.period_us)) * 4096.0))
        return int(clamp(counts, 0, 4095))

    def set_servo_deg(self, channel: int, deg: float):
        deg = clamp(deg, 0.0, 180.0)
        us = self._pulse_us_for_angle(deg)
        counts = self._counts_from_pulse_us(us)
        with self.lock:
            self.pca.set_pwm_counts(channel, 0, counts)

    def _apply_sign_around_home(self, deg_calc: float, sign: float) -> float:
        return self.home + float(sign) * (float(deg_calc) - self.home)

    def _apply_output_adjust(self, deg_calc: float, sign: float, offset_deg: float) -> float:
        d = self._apply_sign_around_home(deg_calc, sign)
        d = d + float(offset_deg)
        return d

    def _apply_shoulder_pair(self, shoulder_deg_main: float) -> float | None:
        """Apply main shoulder and mirrored shoulder. Returns mirror angle if enabled."""
        self.set_servo_deg(self.ch_shoulder, shoulder_deg_main)
        if self.shoulder_mirror_enabled:
            mirror = float(self.shoulder_mirror_angle_max) - float(shoulder_deg_main)
            self.set_servo_deg(self.ch_shoulder_mirror, mirror)
            return mirror
        return None

    def home_pose(self):
        # Set main 4DOF joints to home with offsets
        self.set_servo_deg(self.ch_base, self.home + self.offset_base)
        self._apply_shoulder_pair(self.home + self.offset_sh)
        self.set_servo_deg(self.ch_elbow, self.home + self.offset_el)
        self.set_servo_deg(self.ch_wrist, self.home + self.offset_wr)

        # Apply fixed channels once at start
        for ch, deg in self.fixed_map.items():
            self.set_servo_deg(ch, deg)

    # ---------- old workspace logic ----------
    def calculate_workspace(self, zDistanceCm: float):
        r_max_total = self.L1 + self.L2 + self.L3
        r_max_total_sq = r_max_total * r_max_total
        z_dist_sq = zDistanceCm * zDistanceCm

        xHalfSpan = 0.0 if z_dist_sq >= r_max_total_sq else math.sqrt(r_max_total_sq - z_dist_sq)

        r_max_L1_L2 = self.L1 + self.L2
        r_max_L1_L2_sq = r_max_L1_L2 * r_max_L1_L2

        r_wrist_at_X_zero = zDistanceCm - self.L3
        r_wrist_sq = r_wrist_at_X_zero * r_wrist_at_X_zero
        yHalfSpan = 0.0 if r_wrist_sq >= r_max_L1_L2_sq else math.sqrt(r_max_L1_L2_sq - r_wrist_sq)
        return xHalfSpan, yHalfSpan

    def map_to_robot_space_3d(self, u01: float, v01: float):
        # fixed Z plane only (like old node fixed mode)
        self._zPlane_cm = self.zPlane_default + self.z_offset_cm

        xHalf, yHalf = self.calculate_workspace(self._zPlane_cm)
        draw_area = (2.0 * yHalf) if self.use_yspan else (2.0 * xHalf)

        Xr_cm = (2.0 * u01 - 1.0) * draw_area
        Yr_cm = (2.0 * v01 - 1.0) * draw_area
        Zr_cm = self._zPlane_cm
        return Xr_cm, Yr_cm, Zr_cm

    # ---------- old 4DOF IK ----------
    def solve_ik_3d_with_base(self, Xr_cm, Yr_cm, Zr_cm, elbow_up_mode: bool):
        # Exactly from your backup node:
        theta0 = math.atan2(Xr_cm, Zr_cm)

        r_target = math.sqrt(Xr_cm * Xr_cm + Zr_cm * Zr_cm)
        y_target = Yr_cm

        r_wrist = r_target - self.L3
        y_wrist = y_target

        d_sq = r_wrist * r_wrist + y_wrist * y_wrist
        L1_sq = self.L1 * self.L1
        L2_sq = self.L2 * self.L2

        cosTheta2 = (d_sq - L1_sq - L2_sq) / (2.0 * self.L1 * self.L2)
        cosTheta2 = clamp(cosTheta2, -1.0, 1.0)

        theta2 = math.acos(cosTheta2)
        if elbow_up_mode:
            theta2 = -theta2

        k1 = self.L1 + self.L2 * math.cos(theta2)
        k2 = self.L2 * math.sin(theta2)

        theta1 = math.atan2(y_wrist, r_wrist) - math.atan2(k2, k1)
        theta3 = -(theta1 + theta2)

        deg0 = math.degrees(theta0)
        deg1 = math.degrees(theta1)
        deg2 = math.degrees(theta2)
        deg3 = math.degrees(theta3)

        degBase = self.base_off + (deg0 * self.base_scale)
        degShoulder = self.sh_off - (deg1 * self.shoulder_scale)
        degElbow = self.el_off + (deg2 * self.elbow_scale)
        degWrist = self.wr_off - (deg3 * self.wrist_scale)

        return degBase, degShoulder, degElbow, degWrist

    # ---------- auto draw square ----------
    def _build_auto_square(self):
        self._auto_path = []
        half = clamp(self.auto_square_side_uv, 0.0, 1.0) * 0.5
        cx, cy = 0.5, 0.5
        n = max(2, int(self.auto_points_per_side))

        def lerp(a, b, t):
            return a + (b - a) * t

        # bottom
        for i in range(n):
            t = i / float(n - 1)
            self._auto_path.append((lerp(cx - half, cx + half, t), cy - half, True))
        # right
        for i in range(1, n):
            t = i / float(n - 1)
            self._auto_path.append((cx + half, lerp(cy - half, cy + half, t), True))
        # top
        for i in range(1, n):
            t = i / float(n - 1)
            self._auto_path.append((lerp(cx + half, cx - half, t), cy + half, True))
        # left
        for i in range(1, n - 1):
            t = i / float(n - 1)
            self._auto_path.append((cx - half, lerp(cy + half, cy - half, t), True))

        self._auto_idx = 0
        self._auto_next_t = time.monotonic()

    def _maybe_advance_auto(self, now_s: float):
        if not self._auto_path:
            return
        if now_s < self._auto_next_t:
            return

        u, v, pen = self._auto_path[self._auto_idx]
        self._auto_idx += 1
        if self._auto_idx >= len(self._auto_path):
            if self.auto_loop:
                self._auto_idx = 0
            else:
                self.auto_draw = False
                return

        self._last_u = float(u)
        self._last_v = float(v)
        self._last_pen_down = bool(pen)
        self._have_uv = True
        self._auto_next_t = now_s + float(self.auto_point_interval)

    # ---------- ROS callbacks ----------
    def _on_uv(self, msg: Point):
        # x=u, y=v, z=penDown (0/1)
        self._last_u = float(msg.x)
        self._last_v = float(msg.y)
        self._last_pen_down = (float(msg.z) >= 0.5)
        self._have_uv = True

    def _tick(self):
        now = time.monotonic()

        # Always enforce fixed channels (hold them every tick)
        for ch, deg in self.fixed_map.items():
            self.set_servo_deg(ch, deg)

        if self.auto_draw:
            self._maybe_advance_auto(now)

        if not self._have_uv:
            return

        if (now - self._last_send_time) < self.send_interval:
            return

        u = clamp(self._last_u, 0.0, 1.0)
        v = clamp(self._last_v, 0.0, 1.0)
        pen = bool(self._last_pen_down)

        Xr_cm, Yr_cm, Zr_cm = self.map_to_robot_space_3d(u, v)

        if self.pub_debug_target.get_subscription_count() > 0:
            dbg = Point()
            dbg.x = float(Xr_cm)
            dbg.y = float(Yr_cm)
            dbg.z = float(Zr_cm)
            self.pub_debug_target.publish(dbg)

        if not pen:
            self._last_send_time = now
            return

        # calc (old IK)
        degBase, degShoulder, degElbow, degWrist = self.solve_ik_3d_with_base(Xr_cm, Yr_cm, Zr_cm, self.elbow_up)

        # out (after sign+offset)
        out_base = self._apply_output_adjust(degBase, self.sign_base, self.offset_base)
        out_sh = self._apply_output_adjust(degShoulder, self.sign_sh, self.offset_sh)
        out_el = self._apply_output_adjust(degElbow, self.sign_el, self.offset_el)
        out_wr = self._apply_output_adjust(degWrist, self.sign_wr, self.offset_wr)

        # apply to hardware
        self.set_servo_deg(self.ch_base, out_base)
        sh_mirror = self._apply_shoulder_pair(out_sh)
        self.set_servo_deg(self.ch_elbow, out_el)
        self.set_servo_deg(self.ch_wrist, out_wr)

        # verbose log (your request)
        if self.verbose_log:
            fixed_str = ", ".join([f"CH{ch}={deg:.1f}" for ch, deg in self.fixed_map.items()]) if self.fixed_map else "(none)"
            sh_mirror_str = "{:.2f}".format(sh_mirror) if sh_mirror is not None else "disabled"

            self.get_logger().info(
                "target_cm=({:.2f},{:.2f},{:.2f}) | "
                "calc_deg base={:.2f} sh={:.2f} el={:.2f} wrist={:.2f} | "
                "out_deg base(CH{})={:.2f} sh(CH{})={:.2f} sh_mirror(CH{})={} "
                "el(CH{})={:.2f} wrist(CH{})={:.2f} | fixed: {}".format(
                    Xr_cm,
                    Yr_cm,
                    Zr_cm,
                    degBase,
                    degShoulder,
                    degElbow,
                    degWrist,
                    self.ch_base,
                    out_base,
                    self.ch_shoulder,
                    out_sh,
                    self.ch_shoulder_mirror,
                    sh_mirror_str,
                    self.ch_elbow,
                    out_el,
                    self.ch_wrist,
                    out_wr,
                    fixed_str,
                )
            )

        self._last_send_time = now


def main():
    rclpy.init()
    node = RoboArm4DOFStandalone()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pca.set_all_off()
            node.bus.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
