import board
import busio
import adafruit_tca9548a
import adafruit_vl53l1x
import adafruit_vl53l0x
import time

class SensorManager:
    def __init__(self, tca_obj):
        self.base_sensor = None
        self.ee_sensor = None
        self.base_error_count = 0

        # --- KHỞI TẠO BASE SENSOR (VL53L1X - Tầm xa) ---
        try:
            self.base_sensor = adafruit_vl53l1x.VL53L1X(tca_obj[1])
            # Giảm timing_budget xuống thấp nhất (20ms hoặc 33ms) để đọc nhanh hơn
            self.base_sensor.timing_budget = 33 
            self.base_sensor.start_ranging()
            print("✅ Base Sensor Init OK")
        except Exception as e: 
            print(f"⚠️ Base Sensor Init Fail: {e}")
        
        # --- KHỞI TẠO EE SENSOR (VL53L0X - Tầm gần) ---
        try:
            self.ee_sensor = adafruit_vl53l0x.VL53L0X(tca_obj[0])
            print("✅ EE Sensor Init OK")
        except Exception as e: 
            print(f"⚠️ EE Sensor Init Fail: {e}")

    def get_data(self):
        """
        Trả về dữ liệu THÔ (Raw) đơn vị mét.
        Không lọc (Filtering) ở đây để tránh trễ pha (Phase Lag).
        """
        d_base = None
        d_ee = None
        
        # --- ĐỌC BASE SENSOR ---
        if self.base_sensor:
            try:
                if self.base_sensor.data_ready:
                    raw_mm = self.base_sensor.distance
                    self.base_sensor.clear_interrupt()
                    
                    if raw_mm is not None:
                        # Chỉ đổi đơn vị, không lọc
                        d_base = raw_mm / 1000.0
                        self.base_error_count = 0 
            except Exception:
                self.base_error_count += 1
                # Cơ chế tự hồi phục nếu treo sensor
                if self.base_error_count > 10:
                    self._reset_base_sensor()

        # --- ĐỌC EE SENSOR ---
        if self.ee_sensor:
            try:
                raw_mm = self.ee_sensor.range
                # Lọc sơ bộ giá trị rác (VL53L0X hay trả về 8190 khi quá xa)
                if raw_mm is not None and raw_mm < 8000:
                    d_ee = raw_mm / 1000.0
            except Exception: 
                pass
            
        return d_base, d_ee

    def _reset_base_sensor(self):
        """Hàm phụ trợ để khởi động lại sensor khi bị lỗi I2C"""
        try:
            self.base_sensor.stop_ranging()
            time.sleep(0.01)
            self.base_sensor.start_ranging()
            self.base_error_count = 0
            print("♻️ Base Sensor Restarted")
        except:
            pass