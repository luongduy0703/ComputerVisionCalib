import time
from adafruit_servokit import ServoKit

class ServoController:
    """
    Servo controller supporting:
    - 4DOF IK output (base/shoulder/elbow/wrist_pitch)
    - 6DOF/7-servo wiring by keeping extra channels fixed (or disabled)
    - optional shoulder mirror (second servo mechanically linked)

    Backward compatibility:
    - Existing code may call `ServoController(i2c_channel_obj, deadband=...)`
    - Newer code may call `ServoController()` with auto I2C + mux init
    """

    def __init__(
        self,
        i2c_channel_obj=None,
        address=0x40,
        deadband=0.1,
        *,
        channels=None,
        pulse_width_range=(600, 2400),
        fixed_channels=None,
        fixed_degs=None,
        off_channels=None,
        shoulder_mirror_enabled=False,
        shoulder_mirror_channel=None,
        shoulder_mirror_angle_max=180.0,
    ):
        self.deadband = float(deadband)
        self.address = int(address)
        self.channels = list(channels) if channels is not None else [0, 1, 2, 3]
        self.pulse_width_range = tuple(pulse_width_range)

        fixed_channels = [] if fixed_channels is None else list(fixed_channels)
        fixed_degs = [] if fixed_degs is None else list(fixed_degs)
        if len(fixed_channels) != len(fixed_degs):
            raise ValueError("fixed_channels and fixed_degs must have same length")
        self.fixed_map = {int(ch): float(deg) for ch, deg in zip(fixed_channels, fixed_degs)}

        self.off_channels = [] if off_channels is None else [int(x) for x in off_channels]

        self.shoulder_mirror_enabled = bool(shoulder_mirror_enabled)
        self.shoulder_mirror_channel = None if shoulder_mirror_channel is None else int(shoulder_mirror_channel)
        self.shoulder_mirror_angle_max = float(shoulder_mirror_angle_max)

        # Track last commanded degrees per channel (for deadband)
        self.current_angles = {int(ch): 90.0 for ch in set(self.channels) | set(self.fixed_map.keys())}
        if self.shoulder_mirror_enabled and self.shoulder_mirror_channel is not None:
            self.current_angles.setdefault(self.shoulder_mirror_channel, 90.0)

        # --- auto I2C init if no channel object provided ---
        if i2c_channel_obj is None:
            try:
                import yaml
                import os
                import board as _board
                import busio as _busio
                from adafruit_tca9548a import TCA9548A

                # Load mux + servo address from config if present
                current_dir = os.path.dirname(os.path.abspath(__file__))
                cfg_path = os.path.abspath(os.path.join(current_dir, "..", "config", "robot_config.yaml"))
                mux_ch = 2
                mux_addr = 0x70
                servo_addr = self.address
                if os.path.exists(cfg_path):
                    with open(cfg_path, "r") as f:
                        cfg = yaml.safe_load(f) or {}
                    i2c_cfg = (((cfg.get("sensors") or {}).get("i2c") or {}))
                    mux_addr = int(i2c_cfg.get("mux_address", mux_addr))
                    mux_ch = int(i2c_cfg.get("mux_channel", mux_ch))
                    servo_addr = int(i2c_cfg.get("servo_address", servo_addr))
                    self.address = servo_addr

                    # robot.servos.channels can override default [0,1,2,3]
                    robot_servos = (((cfg.get("robot") or {}).get("servos") or {}))
                    if channels is None and isinstance(robot_servos.get("channels"), list):
                        self.channels = [int(x) for x in robot_servos["channels"]]

                i2c = _busio.I2C(_board.SCL, _board.SDA)
                tca = TCA9548A(i2c, address=mux_addr)
                i2c_channel_obj = tca[mux_ch]
            except Exception as e:
                print(f"❌ I2C Auto-Init Error: {e}")
                i2c_channel_obj = None

        try:
            if i2c_channel_obj is None:
                raise RuntimeError("No I2C channel provided/initialized")

            self.kit = ServoKit(channels=16, i2c=i2c_channel_obj, address=self.address)
            
            # Configure pulse width for all used channels
            pw_min, pw_max = int(self.pulse_width_range[0]), int(self.pulse_width_range[1])
            for ch in set(self.channels) | set(self.fixed_map.keys()):
                try:
                    self.kit.servo[int(ch)].set_pulse_width_range(pw_min, pw_max)
                except Exception:
                    pass
            if self.shoulder_mirror_enabled and self.shoulder_mirror_channel is not None:
                try:
                    self.kit.servo[int(self.shoulder_mirror_channel)].set_pulse_width_range(pw_min, pw_max)
                except Exception:
                    pass

            # Apply OFF channels at startup
            for ch in self.off_channels:
                try:
                    self.kit.servo[int(ch)].angle = None
                except Exception:
                    pass

            # Apply fixed channels once at startup
            for ch, deg in self.fixed_map.items():
                self._set_servo_deg(int(ch), float(deg), force=True)

            print(
                "✅ Servo Driver Init OK "
                f"(Deadband={self.deadband}, addr=0x{self.address:02X}, channels={self.channels}, "
                f"fixed={self.fixed_map}, off={self.off_channels})"
            )
        except Exception as e:
            print(f"❌ Servo Driver Error: {e}")
            self.kit = None

    def _set_servo_deg(self, channel: int, deg: float, force: bool = False):
        """Set a single servo angle with deadband and clamping."""
        if not self.kit:
            return
        ch = int(channel)
        target = float(max(0.0, min(180.0, deg)))
        prev = float(self.current_angles.get(ch, 90.0))
        if force or abs(target - prev) > self.deadband:
            try:
                self.kit.servo[ch].angle = target
                self.current_angles[ch] = target
            except OSError:
                pass

    def apply_angles(self, angles):
        """
        Apply commanded joint angles.

        Supported inputs:
        - list/tuple: values mapped onto `self.channels` order
        - dict: {channel:int -> deg:float}

        This enables a 6DOF-wired arm to run a 4DOF IK by:
        - driving only base/shoulder/elbow/wrist_pitch channels
        - optionally mirroring shoulder
        - optionally keeping extra channels fixed (e.g., wrist_roll, pen)
        - optionally disabling channels (off_channels)
        """
        if not self.kit or angles is None:
            return

        # OFF channels (hold disabled)
        for ch in self.off_channels:
            try:
                self.kit.servo[int(ch)].angle = None
            except Exception:
                pass

        # Apply fixed channels first (always held)
        for ch, deg in self.fixed_map.items():
            self._set_servo_deg(int(ch), float(deg))

        # Normalize input to channel->deg commands
        cmd = {}
        if isinstance(angles, dict):
            cmd = {int(k): float(v) for k, v in angles.items()}
        else:
            seq = list(angles)
            for i, ch in enumerate(self.channels):
                if i >= len(seq):
                    break
                cmd[int(ch)] = float(seq[i])

        # Apply commanded channels
        for ch, deg in cmd.items():
            self._set_servo_deg(int(ch), float(deg))
            
        # Optional shoulder mirror: assumes shoulder is channel[1] unless explicitly set in cmd
        if self.shoulder_mirror_enabled and self.shoulder_mirror_channel is not None:
            # pick "main shoulder" angle from cmd if present; else from current_angles of channels[1]
            main_sh_ch = int(self.channels[1]) if len(self.channels) > 1 else None
            if main_sh_ch is not None:
                main_deg = float(cmd.get(main_sh_ch, self.current_angles.get(main_sh_ch, 90.0)))
                mirror_deg = float(self.shoulder_mirror_angle_max) - main_deg
                self._set_servo_deg(int(self.shoulder_mirror_channel), mirror_deg)