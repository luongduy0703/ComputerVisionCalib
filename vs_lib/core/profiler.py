import time
import csv
import numpy as np
import os

class SystemProfiler:
    def __init__(self, filename="pbvs_metrics.csv", output_dir=None):
        """
        Khởi tạo Profiler để ghi log hiệu năng hệ thống.
        
        Args:
            filename (str): Tên file log.
            output_dir (str, optional): Thư mục lưu file. Nếu None, dùng thư mục hiện tại.
        """
        # Xử lý đường dẫn file an toàn
        if output_dir is None:
            self.log_dir = os.getcwd()
        else:
            self.log_dir = output_dir

        # Đảm bảo thư mục tồn tại
        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir)
            except OSError as e:
                print(f"⚠️ [Profiler] Không thể tạo thư mục {self.log_dir}: {e}")
                self.log_dir = os.getcwd()

        self.log_file = os.path.join(self.log_dir, filename)

        # [CẬP NHẬT] Danh sách các cột dữ liệu (Headers) cho Benchmark Toàn Diện
        self.headers = [
            "Timestamp",          
            "Loop_Dt_ms",         # Thời gian vòng lặp (System Health)
            
            # --- VISION METRICS (New) ---
            "Vision_Detect_ms",   # Thời gian ArUco Detect
            "Vision_Solve_ms",    # Thời gian SolvePnP
            "Vision_Total_ms",    # Tổng thời gian xử lý Vision node
            "Vision_Latency_ms",  # Độ trễ từ lúc chụp ảnh đến lúc ra Pose
            
            # --- PROCESSING METRICS (New) ---
            "Queue_Get_ms",       # Thời gian chờ/lấy dữ liệu từ Queue
            "Filter_Update_ms",   # Thời gian chạy bộ lọc (Kalman + OneEuro)
            "Filter_Calc_ms",     # Thời gian tính IK
            "Servo_Write_ms",     # Thời gian gửi lệnh I2C
            
            # --- LATENCY ANALYSIS (New) ---
            "Phase_Delay_ms",     # Độ trễ tổng thể (Now - Image Timestamp)
            
            # --- ACCURACY METRICS ---
            "Tracking_Error_2D_cm", 
            "Tracking_Error_3D_cm",
            "Vision_Deviation_cm",
            
            # --- DATA POINTS ---
            "Raw_Vision_X", "Raw_Vision_Y", "Raw_Vision_Z",
            "Command_X", "Command_Y", "Command_Z",
            "Target_X", "Target_Y", "Target_Z",
            
            # --- IK DIAGNOSTICS ---
            "IK_Success",

            # --- WAYPOINT PIPELINE (Debug) ---
            # Target point in board frame (cm)
            "Target_Board_X_cm", "Target_Board_Y_cm", "Target_Board_Z_cm",
            # Target transformed into camera frame (cm)
            "Target_Cam_X_cm", "Target_Cam_Y_cm", "Target_Cam_Z_cm",
            # Target transformed into base frame BEFORE compensation (cm)
            "Target_Base_Raw_X_cm", "Target_Base_Raw_Y_cm", "Target_Base_Raw_Z_cm",
            # Applied position compensation (cm)
            "Comp_DX_cm", "Comp_DY_cm", "Comp_DZ_cm",
            # Target in base frame AFTER compensation (cm)
            "Target_Base_Comp_X_cm", "Target_Base_Comp_Y_cm", "Target_Base_Comp_Z_cm",
            # Filter outputs (cm)
            "Target_Base_Filt_X_cm", "Target_Base_Filt_Y_cm", "Target_Base_Filt_Z_cm",
            # Wrist tilt used by IK (deg)
            "Tilt_Fixed_deg", "Tilt_Comp_deg",
            # Estimated attitude (deg)
            "Drone_Roll_deg", "Drone_Pitch_deg", "Drone_Yaw_deg",
            # Prediction flags
            "Using_Extrapolation", "Pose_Predicted",
            # Interpolation context
            "Seg_Step", "Seg_Steps",
            
            # Legacy
            "Current_X", "Current_Y", "Current_Z"
        ]
        
        # [DEBUG] In đường dẫn để người dùng biết file nằm đâu
        print(f"📊 [Profiler] Initializing log at: {os.path.abspath(self.log_file)}")

        try:
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            print("   ✅ Log file created successfully.")
        except Exception as e:
            print(f"   ❌ ERROR creating log file: {e}")
            print("   👉 Hint: Close the CSV file if it is open in Excel.")
            
        self.timers = {}
        # [MỚI] Buffer để lưu dữ liệu cho báo cáo tóm tắt
        self.data_buffer = [] 

    def start_timer(self, key):
        self.timers[key] = time.perf_counter()

    def stop_timer(self, key):
        if key in self.timers:
            return (time.perf_counter() - self.timers[key]) * 1000.0 # đổi ra ms
        return 0.0

    def log_data(self, **kwargs):
        """Ghi một dòng dữ liệu vào CSV và lưu vào buffer"""
        row = []
        for h in self.headers:
            # Lấy giá trị từ kwargs, nếu không có thì để 0.0
            val = kwargs.get(h, 0.0)
            row.append(val)
        
        # Lưu vào buffer để tính summary (có thể giới hạn size nếu chạy quá lâu)
        if len(self.data_buffer) < 10000: # Giới hạn mẫu để tránh tràn RAM
            self.data_buffer.append(kwargs)

        try:
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception:
            pass # Silent fail để không làm gián đoạn robot

    def print_summary(self):
        """In báo cáo tóm tắt khi kết thúc chương trình"""
        if not self.data_buffer:
            print("\n⚠️ [Profiler] No data recorded to summarize.")
            return

        print("\n" + "="*50)
        print(f"📊 BENCHMARK SUMMARY ({len(self.data_buffer)} samples)")
        print("="*50)
        
        # Các chỉ số quan trọng cần báo cáo
        metrics_to_report = [
            "Loop_Dt_ms",
            "Vision_Detect_ms", 
            "Vision_Solve_ms", 
            "Vision_Total_ms",
            "Vision_Latency_ms",
            "Filter_Update_ms", 
            "Servo_Write_ms", 
            "Phase_Delay_ms",
            "Tracking_Error_3D_cm"
        ]

        print(f"{'METRIC':<25} | {'AVG (ms/cm)':<12} | {'MAX (ms/cm)':<12}")
        print("-" * 55)

        for m in metrics_to_report:
            # Lọc ra các giá trị > 0 để tính trung bình chính xác hơn (trừ error)
            values = [d.get(m, 0.0) for d in self.data_buffer if d.get(m, 0.0) != 0.0]
            
            if values:
                avg_val = sum(values) / len(values)
                max_val = max(values)
                print(f"🔹 {m:<22} | {avg_val:10.2f}   | {max_val:10.2f}")
            else:
                print(f"🔹 {m:<22} | {'N/A':>10}   | {'N/A':>10}")
        
        print("="*50 + "\n")