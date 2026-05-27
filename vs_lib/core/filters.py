import math
import time  # [MỚI] Cần thiết cho One Euro Filter

class EMASmoother:
    """
    Bộ lọc trung bình động hàm mũ (Exponential Moving Average).
    (Legacy) Giữ lại để tương thích ngược.
    """
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None

class KalmanFilter1D:
    """
    Bộ lọc Kalman 1 chiều.
    [CẬP NHẬT] Đã tinh chỉnh R/Q để xử lý rung động mạnh từ Drone.
    """
    def __init__(self, R=0.1, Q=0.01):
        self.R = R  # Nhiễu đo lường (Tăng R nếu Vision rung nhiều)
        self.Q = Q  # Nhiễu hệ thống
        self.x = None 
        self.p = 1.0

    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            self.p = 1.0
            return self.x

        # Dự đoán (Prediction)
        self.p = self.p + self.Q

        # Cập nhật (Correction)
        K = self.p / (self.p + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.p = (1 - K) * self.p
        
        return self.x

    def reset(self):
        self.x = None
        self.p = 1.0

class OutlierRejector:
    """
    Bộ lọc loại bỏ giá trị nhảy vọt (Spike).
    """
    def __init__(self, max_jump=5.0):
        self.max_jump = max_jump
        self.last_val = None

    def check(self, new_val):
        if self.last_val is None:
            self.last_val = new_val
            return new_val
            
        if abs(new_val - self.last_val) > self.max_jump:
            return self.last_val # Từ chối, dùng lại giá trị cũ
        else:
            self.last_val = new_val
            return new_val

    def reset(self):
        self.last_val = None

# ==========================================================
# [MỚI] ONE EURO FILTER - GIẢI PHÁP CHỐNG RUNG & GIẢM TRỄ
# ==========================================================
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        min_cutoff: Tần số cắt tối thiểu (Hz). Nhỏ = mượt hơn khi chậm.
        beta: Hệ số tốc độ. Lớn = giảm trễ khi nhanh.
        d_cutoff: Tần số cắt cho bộ lọc đạo hàm (thường giữ 1.0).
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def _exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def update(self, x, t=None):
        if t is None:
            t = time.time()
            
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        
        # Tránh lỗi chia cho 0 nếu update quá nhanh
        if t_e <= 0.0: return self.x_prev

        # Tính đạo hàm (tốc độ thay đổi) đã lọc
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        # Tính tần số cắt động dựa trên tốc độ
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Lọc tín hiệu chính
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat
    
    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None