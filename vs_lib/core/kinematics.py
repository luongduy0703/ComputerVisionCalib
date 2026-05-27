import math
import yaml
import os  # <--- Quan trọng

class KinematicsSolver:
    def __init__(self, config_path=None):
        # --- FIX: SỬ DỤNG ĐƯỜNG DẪN TUYỆT ĐỐI ---
        if config_path is None:
            # Lấy vị trí của file kinematics.py này (trong thư mục core)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Đi ngược ra thư mục cha (..) rồi vào config
            config_path = os.path.join(current_dir, '..', 'config', 'robot_config.yaml')

        # Chuyển thành đường dẫn chuẩn (resolve ..)
        config_path = os.path.abspath(config_path)

        # Kiểm tra tồn tại
        if not os.path.exists(config_path):
            # In ra đường dẫn đầy đủ để debug
            raise FileNotFoundError(f"❌ Không tìm thấy Config tại: {config_path}")

        print(f"⚙️ Loading Config: {config_path}")
        
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)['robot']
        
        self.dims = cfg['dimensions']
        self.limits = cfg['limits']
        self.offsets = cfg['offsets']
        self.servos = cfg['servos']
        # Base angle convention: False = atan2(x,y) [arm 0° along +Y]; True = atan2(y,x) [0° along +X]
        self.base_angle_atan2_yx = bool(self.servos.get('base_angle_atan2_yx', False))
        
        # [MỚI] Load Manual Offset (Units: cm - must match IK input units)
        self.manual_x = self.offsets.get('vision_x', 0.0)
        self.manual_y = self.offsets.get('vision_y', 0.0)
        self.manual_z = self.offsets.get('vision_z', 0.0)
        
        # Load Z manual offset from config (Units: cm)
        self.z_manual_offset = self.offsets.get('z_manual_offset', 0.5)

        self.J4_REVERSE = True

    def _solve_2link(self, r, z):
        L1, L2 = self.dims['L1'], self.dims['L2']
        d_sq = r**2 + z**2
        dist = math.sqrt(d_sq)
        
        # --- Robust reach handling ---
        # Instead of giving up when the target is out of reach, we
        # project it back onto the closest point that the 2-link arm
        # CAN reach. This avoids IK_FAIL floods and ensures joints
        # 2–3 always move "as far as possible" toward the goal.
        max_reach = L1 + L2
        min_reach = abs(L1 - L2)
        if dist > max_reach or dist < min_reach:
            if dist < 1e-6:
                # Degenerate case: keep arm folded at min_reach straight up
                r = 0.0
                z = min_reach
            elif dist > max_reach:
                # Limit horizontal radius only; keep height z unchanged (avoid pen lift/plunge)
                r_sq_max = max_reach**2 - z**2
                if r_sq_max < 0:
                    r_sq_max = 0.0
                r = math.copysign(math.sqrt(r_sq_max), r) if abs(r) > 1e-9 else 0.0
                d_sq = r**2 + z**2
            else:
                # Point too close (dist < min_reach): scale (r, z) toward min_reach
                scale = min_reach / dist
                r *= scale
                z *= scale
                d_sq = r**2 + z**2
            
        cos_t2 = (d_sq - L1**2 - L2**2) / (2 * L1 * L2)
        cos_t2 = max(-1.0, min(1.0, cos_t2))
        theta2 = -math.acos(cos_t2)
        
        k1 = L1 + L2 * math.cos(theta2)
        k2 = L2 * math.sin(theta2)
        theta1 = math.atan2(z, r) - math.atan2(k2, k1)
        return theta1, theta2

    def solve_ik(self, x, y, z, tilt=-20.0):
        """
        Solve inverse kinematics for robot arm.
        
        Args:
            x, y, z: Target position in CENTIMETERS (cm)
            tilt: Wrist tilt angle in degrees
        
        Returns:
            List of servo angles [base, shoulder, elbow, wrist] in degrees
        
        Note: All dimensions (L0-L3) and limits (z_floor) are in cm.
        """
        SAFE_TILT_MAX = -30.0 
        SAFE_TILT_MIN = -90.0
        
        if tilt > SAFE_TILT_MAX: 
            print(f"⚠️ WARNING: Tilt {tilt} quá cao! Đã kẹp về {SAFE_TILT_MAX}")
            tilt = SAFE_TILT_MAX
            
        if tilt < SAFE_TILT_MIN:
            tilt = SAFE_TILT_MIN

        # [MỚI] Manual offset: apply as linear shift in base frame AFTER base rotation
        # so it does not distort atan2 (non-linear at edges). We apply offset to the
        # position used for r_total and z_target, not to the angle.
        x_eff = x + self.manual_x
        y_eff = y + self.manual_y
        z_eff = z + self.manual_z

        # 1. Check Floor (z_eff used below after z_target is set)
        
        # 2. Base Rotation (convention must match T_calib: atan2(x,y)=0° along +Y keeps base in range)
        if self.base_angle_atan2_yx:
            theta_base_rad = math.atan2(y_eff, x_eff)
        else:
            theta_base_rad = math.atan2(x_eff, y_eff)
        sv_base = 90.0 + math.degrees(theta_base_rad) * self.servos['scale_base']
        
        # 3. Wrist Position Calculation (use effective position including linear offset)
        # Dimensions L0-L3 are in cm (from config), x_eff/y_eff/z_eff are in cm - consistent ✓
        L0, L3 = self.dims['L0'], self.dims['L3']
        z_target = z_eff + self.z_manual_offset
        if z_target < self.limits['z_floor']: return None
        r_total = math.sqrt(x_eff**2 + y_eff**2)
        z_from_shoulder = z_target + L0
        
        tilt_rad = math.radians(tilt)
        r_wrist = r_total - L3 * math.cos(tilt_rad)
        z_wrist = z_from_shoulder - L3 * math.sin(tilt_rad)
        
        # 4. Solve 2-Link (Shoulder & Elbow)
        geom = self._solve_2link(r_wrist, z_wrist)
        if not geom: return None
        t1, t2 = geom
        
        # 5. Convert to Servo Angles
        scale = self.servos['scale_arm']
        sv_shoulder = 90.0 - math.degrees(t1) * scale
        sv_elbow = (180.0 - self.offsets['J3']) + math.degrees(t2) * scale
        
        # 6. Solve Wrist (J4)
        t3_local = tilt_rad - (t1 + t2)
        j4_deg = math.degrees(t3_local)
        
        if self.J4_REVERSE:
            sv_wrist = 90.0 + self.offsets['J4'] - j4_deg * scale
        else:
            sv_wrist = 90.0 + self.offsets['J4'] + j4_deg * scale
            
        return [sv_base, sv_shoulder, sv_elbow, sv_wrist]

    def solve_ik_4dof(self, x, y, z, tilt=-20.0):
        """
        Backward-compatible alias for the current IK (4 DOF: base, shoulder, elbow, wrist_pitch).

        Args:
            x, y, z: cm
            tilt: deg
        Returns:
            [base, shoulder, elbow, wrist_pitch] degrees or None
        """
        return self.solve_ik(x, y, z, tilt)

    @staticmethod
    def to_channel_map(angles_4dof, channels=(0, 1, 2, 3)):
        """
        Utility for 6DOF-wired setups that still use 4DOF IK:
        map [base, shoulder, elbow, wrist_pitch] -> {channel:deg}.
        """
        if angles_4dof is None:
            return None
        ch = list(channels)
        if len(ch) < 4:
            raise ValueError("channels must contain at least 4 entries for 4DOF mapping")
        return {int(ch[i]): float(angles_4dof[i]) for i in range(4)}