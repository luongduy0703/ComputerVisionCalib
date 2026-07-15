import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def _search_paths(filename):
    """Trả về danh sách các path có thể chứa file log."""
    return [
        filename,
        os.path.join("nodes", filename),
        os.path.join("core", filename),
        os.path.join("..", "nodes", filename),
    ]


def load_exec_log_file(filename):
    """
    Load log Executor (pbvs_metrics.csv) với header chuẩn.

    File này đã có dòng header đầy đủ:
      Timestamp, Loop_Dt_ms, Vision_Detect_ms, ...
    nên chỉ cần đọc bình thường.
    """
    for p in _search_paths(filename):
        if os.path.exists(p):
            print(f"✅ Loaded EXEC log: {p}")
            try:
                df = pd.read_csv(p)
                df.columns = df.columns.str.strip()
                # Coerce numeric columns used for plotting so charts are never blank
                numeric_cols = [
                    "Timestamp", "Loop_Dt_ms", "Phase_Delay_ms", "Filter_Update_ms",
                    "Tracking_Error_3D_cm", "Raw_Vision_X", "Raw_Vision_Y", "Raw_Vision_Z",
                    "Command_X", "Command_Y", "Command_Z", "Target_X", "Target_Y", "Target_Z",
                    "Pose_Predicted",
                ]
                for c in numeric_cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                if df.empty:
                    print(f"⚠️ EXEC log has 0 data rows (only header?).")
                return df
            except Exception as e:
                print(f"❌ Error reading EXEC log {p}: {e}")
                return None

    print(f"⚠️ Warning: Could not find EXEC log {filename}")
    return None


def load_vision_log_file(filename):
    """
    Load log Vision (vision_metrics.csv) KHÔNG có header.

    Dòng đầu tiên của file là dữ liệu số, không phải tên cột, ví dụ:
      1770282725.917816,0.0,8.87...,1.04...,14.69...,532.01,...

    Ta chỉ cần các cột:
      0: Timestamp
      2: Vision_Detect_ms
      3: Vision_Solve_ms
      4: Vision_Total_ms
      5: Vision_Latency_ms
    Các cột còn lại hiện tại đều là 0 và không dùng tới.
    """
    vision_cols = [
        "Timestamp",
        "Loop_Dt_ms",
        "Vision_Detect_ms",
        "Vision_Solve_ms",
        "Vision_Total_ms",
        "Vision_Latency_ms",
    ]

    for p in _search_paths(filename):
        if os.path.exists(p):
            print(f"✅ Loaded VISION log: {p}")
            try:
                df = pd.read_csv(
                    p,
                    header=None,          # không có header
                    names=vision_cols,    # ép tên cột
                    usecols=range(len(vision_cols)),  # chỉ lấy 6 cột đầu
                )
                # Robust: nếu file THỰC RA có header (vd: dòng đầu là "Timestamp,..."),
                # giá trị Timestamp sẽ là string và gây lỗi float(). Ta ép numeric và bỏ các dòng không hợp lệ.
                df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
                # Các cột timing cũng cần numeric để .mean() hoạt động ổn định
                for c in ("Loop_Dt_ms", "Vision_Detect_ms", "Vision_Solve_ms", "Vision_Total_ms", "Vision_Latency_ms"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["Timestamp"]).reset_index(drop=True)
                return df
            except Exception as e:
                print(f"❌ Error reading VISION log {p}: {e}")
                return None

    print(f"⚠️ Warning: Could not find VISION log {filename}")
    return None


def _set_ylim_positive(ax, df, columns):
    """Set y-axis limits to [0, max] so flat data is visible. Safe if df is None or columns missing."""
    if df is None or df.empty or not columns:
        ax.set_ylim(0, 1)
        return
    vals = []
    for c in columns:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if not v.empty:
                vals.extend(v.tolist())
    if vals:
        mx = max(vals)
        ax.set_ylim(0, max(mx * 1.05, 0.1))
    else:
        ax.set_ylim(0, 1)


def _set_ylim_symmetric(ax, df, columns):
    """Set y-axis to symmetric range around 0 so position data is visible."""
    if df is None or df.empty or not columns:
        ax.set_ylim(-1, 1)
        return
    vals = []
    for c in columns:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if not v.empty:
                vals.extend(v.tolist())
    if vals:
        mx = max(abs(x) for x in vals)
        margin = max(mx * 1.1, 0.1)
        ax.set_ylim(-margin, margin)
    else:
        ax.set_ylim(-1, 1)


# ---------------------------------------------------------------------------
# BẢNG THAM CHIẾU CỘT DỮ LIỆU DÙNG CHO PHÂN TÍCH
# ---------------------------------------------------------------------------
LOG_COLUMNS_REFERENCE = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ LOG FILE: pbvs_metrics.csv (Executor)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ COLUMN                  │ USED IN ANALYSIS                                   │
├─────────────────────────┼────────────────────────────────────────────────────┤
│ Timestamp               │ Time sync (t0), Time_Rel, phase delay            │
│ Time_Rel                │ Derived: Timestamp - t0 (x-axis all time plots)   │
│ Loop_Dt_ms              │ [1] Execution stats (avg loop time)               │
│ Phase_Delay_ms           │ [1] Stats; Subplot 2 (latency)                    │
│ Filter_Update_ms        │ [1] Execution stats                                │
│ Tracking_Error_3D_cm    │ [1] Stats; Subplot 4 (tracking error)             │
│ Raw_Vision_X,Y,Z        │ Subplot 3 (stabilization); Histogram (no-filter)  │
│ Command_X,Y,Z           │ Subplot 3 (smoothed); Histogram (with-filter)      │
│ Target_X,Y,Z            │ Subplot 3 (ideal); Histogram (error vs target)    │
│ Pose_Predicted          │ Histogram: split With/No predictor                │
│ Using_Extrapolation      │ (Available for future use)                       │
└─────────────────────────┴────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ LOG FILE: vision_metrics.csv (Vision node)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ COLUMN                  │ USED IN ANALYSIS                                   │
├─────────────────────────┼────────────────────────────────────────────────────┤
│ Timestamp               │ Time sync (t0), Time_Rel                           │
│ Time_Rel                │ Derived (x-axis)                                   │
│ Vision_Detect_ms        │ [2] Stats; Subplot 1 (Detect ArUco)               │
│ Vision_Solve_ms         │ [2] Stats                                          │
│ Vision_Total_ms         │ [2] Stats; Subplot 1 (Total Vision Loop)          │
│ Vision_Latency_ms       │ Subplot 2 (Camera Hardware Latency)               │
└─────────────────────────┴────────────────────────────────────────────────────┘

Histogram: requires Command_X/Y/Z, Raw_Vision_X/Y/Z, Target_X/Y/Z, Pose_Predicted (and Target_Base_Raw_* when available).
"""

# ---------------------------------------------------------------------------
# FOUR STATES: Comparative histogram (tracking error vs target)
# ---------------------------------------------------------------------------
# State 1 — No Filter & No Predictor (Raw System)
#   Data: Raw_Vision_X/Y/Z (raw coordinates from Vision node before processing).
#   Characteristics: High jitter (motor vibration), large phase lag vs target.
#
# State 2 — With Filter & No Predictor
#   Data: Command_X/Y/Z where Pose_Predicted < 0.5 (filtered, no V·Δt_delay).
#   Characteristics: Smooth (no jitter) but "behind" target (no delay compensation).
#
# State 3 — No Filter & With Predictor
#   Data: Raw + velocity prediction (V·Δt); if not logged, Raw_Vision is used and noted.
#   Characteristics: Follows target in time (no lag) but spiky/jerky (coarse noise).
#
# State 4 — With Filter & With Predictor (Complete System)
#   Data: Command_X/Y/Z where Pose_Predicted ≥ 0.5 (Smoothed Command / full pipeline).
#   Characteristics: Smooth and free of phase lag — project target.
# ---------------------------------------------------------------------------

FOUR_STATES_SUMMARY = """
  State 1: No Filter & No Predictor (Raw)     → Raw_Vision_X/Y/Z
  State 2: With Filter & No Predictor         → Command_X/Y/Z where Pose_Predicted < 0.5
  State 3: No Filter & With Predictor         → Raw + V·Δt (or Raw if not logged)
  State 4: With Filter & With Predictor       → Command_X/Y/Z where Pose_Predicted ≥ 0.5
"""


def print_columns_used(df_exec, df_vis):
    """In bảng cột có trong file vs cột cần cho phân tích."""
    print("\n" + "="*70)
    print("📋 DATA COLUMNS USED IN LOG FILES FOR ANALYSIS")
    print("="*70)
    print(LOG_COLUMNS_REFERENCE)
    print("Columns present in loaded data:")
    if df_exec is not None and not df_exec.empty:
        exec_cols = list(df_exec.columns)
        required_hist = [
            "Command_X", "Command_Y", "Command_Z",
            "Raw_Vision_X", "Raw_Vision_Y", "Raw_Vision_Z",
            "Target_X", "Target_Y", "Target_Z", "Pose_Predicted",
        ]
        missing = [c for c in required_hist if c not in exec_cols]
        print(f"  pbvs_metrics.csv: {len(exec_cols)} columns")
        if missing:
            print(f"  ⚠️  Missing for histogram: {missing}")
        else:
            print(f"  ✅ All columns required for histogram are present.")
    else:
        print("  pbvs_metrics.csv: (not loaded)")
    if df_vis is not None and not df_vis.empty:
        print(f"  vision_metrics.csv: {list(df_vis.columns)}")
    else:
        print("  vision_metrics.csv: (not loaded or empty)")
    print("Four-state histogram (data sources):")
    print(FOUR_STATES_SUMMARY)
    print("="*70 + "\n")


def analyze_dual_logs(exec_file="pbvs_metrics.csv", vision_file="vision_metrics.csv"):
    print("="*60)
    print("📊 PHÂN TÍCH HIỆU NĂNG HỆ THỐNG (DUAL LOGS)")
    print("="*60)

    df_exec = load_exec_log_file(exec_file)
    df_vis = load_vision_log_file(vision_file)

    if df_exec is None:
        print("❌ Critical: Không tìm thấy log Executor. Dừng phân tích.")
        return

    # In bảng tham chiếu cột và kiểm tra cột tồn tại
    print_columns_used(df_exec, df_vis)

    # --- 1. ĐỒNG BỘ THỜI GIAN (TIME SYNC) ---
    # Lấy thời gian bắt đầu sớm nhất làm T0
    t0 = float(df_exec["Timestamp"].iloc[0])
    if df_vis is not None and not df_vis.empty:
        t0 = min(t0, float(df_vis["Timestamp"].iloc[0]))

    df_exec["Time_Rel"] = df_exec["Timestamp"] - t0
    if df_vis is not None and not df_vis.empty:
        df_vis["Time_Rel"] = df_vis["Timestamp"] - t0

    # --- 2. THỐNG KÊ SỐ LIỆU ---
    print("\n[1] EXECUTION STATISTICS (50Hz Loop):")
    print(f" - Avg Loop Time:      {df_exec['Loop_Dt_ms'].mean():.2f} ms")
    print(f" - Avg Phase Delay:    {df_exec['Phase_Delay_ms'].mean():.2f} ms (Target: <100ms)")
    print(f" - Avg Filter Time:    {df_exec['Filter_Update_ms'].mean():.2f} ms")
    print(f" - Avg Tracking Error: {df_exec['Tracking_Error_3D_cm'].mean():.4f} cm")

    if df_vis is not None and not df_vis.empty:
        print("\n[2] VISION STATISTICS (~30-60Hz):")
        print(f" - Avg Detection Time: {df_vis['Vision_Detect_ms'].mean():.2f} ms")
        print(f" - Avg SolvePnP Time:  {df_vis['Vision_Solve_ms'].mean():.2f} ms")
        print(f" - Avg Camera Latency: {df_vis['Vision_Latency_ms'].mean():.2f} ms (Physical)")

    # --- 3. VẼ BIỂU ĐỒ ---
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # --- SUBPLOT 1: VISION PROCESSING PERFORMANCE ---
    if df_vis is not None and not df_vis.empty:
        t_vis = df_vis['Time_Rel'].dropna()
        d1 = df_vis['Vision_Detect_ms'].dropna()
        d2 = df_vis['Vision_Total_ms'].dropna()
        if not t_vis.empty and (not d1.empty or not d2.empty):
            if not d1.empty and len(t_vis) == len(d1):
                axs[0].plot(t_vis, d1, label='Detect (ArUco)', color='blue', alpha=0.6)
            if not d2.empty and len(t_vis) == len(d2):
                axs[0].plot(t_vis, d2, label='Total Vision Loop', color='red', linewidth=1.5)
                axs[0].fill_between(t_vis, 0, d2, color='red', alpha=0.1)
    _set_ylim_positive(axs[0], df_vis, ['Vision_Detect_ms', 'Vision_Total_ms'] if df_vis is not None else [])
    axs[0].set_ylabel('Processing Time (ms)')
    axs[0].set_title('Biểu đồ 1: Hiệu năng Xử lý Vision (Camera Node)', fontweight='bold')
    axs[0].legend(loc='upper right', fontsize=8)
    axs[0].grid(True, alpha=0.3)

    # --- SUBPLOT 2: SYSTEM LATENCY CHAIN ---
    if df_vis is not None and not df_vis.empty:
        t_vis = df_vis['Time_Rel'].dropna()
        lat = df_vis['Vision_Latency_ms'].dropna()
        if not t_vis.empty and not lat.empty and len(t_vis) == len(lat):
            axs[1].plot(t_vis, lat, label='Camera Hardware Latency', color='orange', linestyle='--')
    t_exec = df_exec['Time_Rel'].dropna()
    phase = df_exec['Phase_Delay_ms'].dropna()
    if not t_exec.empty and not phase.empty:
        axs[1].plot(t_exec, phase, label='Total System Phase Delay', color='purple', linewidth=2)
    axs[1].axhline(y=100, color='red', linestyle=':', label='Safety Limit (100ms)')
    _set_ylim_positive(axs[1], df_exec, ['Phase_Delay_ms'])
    axs[1].set_ylabel('Latency (ms)')
    axs[1].set_title('Biểu đồ 2: Phân tích Độ trễ Hệ thống (Pipeline Latency)', fontweight='bold')
    axs[1].legend(loc='upper right', fontsize=8)
    axs[1].grid(True, alpha=0.3)

    # --- SUBPLOT 3: STABILIZATION QUALITY (X-AXIS) ---
    if 'Raw_Vision_X' in df_exec.columns and 'Command_X' in df_exec.columns:
        t = df_exec['Time_Rel'].values
        raw_x = df_exec['Raw_Vision_X'].fillna(0).values
        cmd_x = df_exec['Command_X'].fillna(0).values
        tgt_x = df_exec['Target_X'].fillna(0).values if 'Target_X' in df_exec.columns else np.zeros_like(t)
        mask = np.isfinite(raw_x) & np.isfinite(cmd_x)
        if np.any(mask):
            axs[2].plot(df_exec['Time_Rel'], raw_x, 'r.', markersize=2, alpha=0.3, label='Raw Input (Vibration)')
            axs[2].plot(df_exec['Time_Rel'], cmd_x, 'b-', linewidth=1.5, label='Smoothed Command')
            axs[2].plot(df_exec['Time_Rel'], tgt_x, 'g--', linewidth=2.0, label='Ideal Target')
    _set_ylim_symmetric(axs[2], df_exec, ['Raw_Vision_X', 'Command_X', 'Target_X'])
    axs[2].set_ylabel('Position X (cm)')
    axs[2].set_title('Biểu đồ 3: Hiệu quả Chống rung (Input vs Output)', fontweight='bold')
    axs[2].legend(loc='upper right', fontsize=8)
    axs[2].grid(True, alpha=0.3)

    # --- SUBPLOT 4: TRACKING ERROR ---
    err = df_exec['Tracking_Error_3D_cm'].dropna() if 'Tracking_Error_3D_cm' in df_exec.columns else pd.Series(dtype=float)
    t_err = df_exec['Time_Rel'].loc[err.index] if not err.empty else pd.Series(dtype=float)
    if not t_err.empty and not err.empty:
        axs[3].plot(t_err, err, color='green', label='Tracking Error (3D)')
        axs[3].fill_between(t_err, 0, err, color='green', alpha=0.2)
    _set_ylim_positive(axs[3], df_exec, ['Tracking_Error_3D_cm'])
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Error (cm)')
    axs[3].set_title('Biểu đồ 4: Sai số Bám quỹ đạo (Accuracy)', fontweight='bold')
    axs[3].legend(loc='upper right', fontsize=8)
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout(pad=0.8)
    fig.canvas.draw()
    save_path = "benchmark_report.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    print(f"\n✅ Đã lưu biểu đồ phân tích vào: {os.path.abspath(save_path)}")
    plt.show()

    # --- 4. BIỂU ĐỒ ĐƯỜNG: TRACKING ERROR SO VỚI GROUND TRUTH ---
    plot_error_line(df_exec, t0)

    # --- 5. BIỂU ĐỒ ĐƯỜNG: STATE 3 (No filter & With predictor) ---
    plot_state3_prediction_offset(df_exec, t0)

    return df_exec, df_vis


def plot_error_line(df_exec, t0):
    """
    Vẽ line graph để thể hiện tracking error so với ground truth:

    - Ground truth line: 0 cm error (theo lý tưởng).
    - State 1 (nếu có): No filter & No predictor (Base_NoPred vs Target_Base_Raw).
    - State 2: With filter & No predictor (hoặc With filter & With predictor nếu prediction=true),
      dùng Command vs Target_Base_Raw.
    """
    if df_exec is None or df_exec.empty:
        return

    if "Time_Rel" not in df_exec.columns:
        df = df_exec.copy()
        df["Time_Rel"] = df["Timestamp"] - t0
    else:
        df = df_exec

    err_state1 = None
    err_state2 = None

    if "Target_Base_Raw_X_cm" in df.columns:
        # Dùng target trong base frame để tính sai số tracking thật
        tx = pd.to_numeric(df["Target_Base_Raw_X_cm"], errors="coerce").fillna(0)
        ty = pd.to_numeric(df["Target_Base_Raw_Y_cm"], errors="coerce").fillna(0)
        tz = pd.to_numeric(df["Target_Base_Raw_Z_cm"], errors="coerce").fillna(0)

        # State 2 (With filter): Command vs target
        cx = pd.to_numeric(df["Command_X"], errors="coerce").fillna(0)
        cy = pd.to_numeric(df["Command_Y"], errors="coerce").fillna(0)
        cz = pd.to_numeric(df["Command_Z"], errors="coerce").fillna(0)
        err_state2 = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2 + (cz - tz) ** 2)

        # State 1 (No filter & No predictor) nếu đã log Base_NoPred
        if all(c in df.columns for c in ["Base_NoPred_X_cm", "Base_NoPred_Y_cm", "Base_NoPred_Z_cm"]):
            bnx = pd.to_numeric(df["Base_NoPred_X_cm"], errors="coerce").fillna(0)
            bny = pd.to_numeric(df["Base_NoPred_Y_cm"], errors="coerce").fillna(0)
            bnz = pd.to_numeric(df["Base_NoPred_Z_cm"], errors="coerce").fillna(0)
            err_state1 = np.sqrt((bnx - tx) ** 2 + (bny - ty) ** 2 + (bnz - tz) ** 2)
    elif "Tracking_Error_3D_cm" in df.columns:
        # Fallback: chỉ có một đường error tổng hợp
        err_state2 = pd.to_numeric(df["Tracking_Error_3D_cm"], errors="coerce").fillna(0)
    else:
        print("⚠️ Không tìm thấy cột Tracking_Error_3D_cm hoặc Target_Base_Raw_*; bỏ qua line graph.")
        return

    t = pd.to_numeric(df["Time_Rel"], errors="coerce")
    if err_state2 is None:
        print("⚠️ Không có dữ liệu State 2 để vẽ line graph.")
        return

    mask = t.notna() & np.isfinite(err_state2)
    t = t[mask]
    err2 = err_state2[mask]
    if err_state1 is not None:
        err1 = err_state1[mask]
    else:
        err1 = None

    if t.empty:
        print("⚠️ Không có dữ liệu hợp lệ để vẽ line graph tracking error.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, np.zeros_like(err2), "k--", linewidth=1.0, label="Ground truth: 0 cm error")

    if err1 is not None:
        ax.plot(t, err1, color="crimson", linewidth=1.0, alpha=0.9,
                label="State 1: No filter & No predictor")

    ax.plot(t, err2, color="darkgreen", linewidth=1.5,
            label="State 2: With filter (prediction off in this run)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (cm)")
    ax.set_title("States 1 & 2: Tracking error vs ground truth (line graph)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout(pad=0.8)
    fig.canvas.draw()
    save_path = "benchmark_error_line.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    print(f"✅ Đã lưu line graph tracking error vào: {os.path.abspath(save_path)}")
    plt.show()


def plot_state3_prediction_offset(df_exec, t0):
    """
    State 3: No Filter & With Predictor.
    Vẽ line graph độ lệch giữa pose dự đoán (Target_Base_Raw_cm) và pose không dự đoán (Base_NoPred_cm)
    khi predictor đang bật (Pose_Predicted >= 0.5):

        err3(t) = || Target_Base_Raw_cm(t) - Base_NoPred_cm(t) ||_2
    """
    if df_exec is None or df_exec.empty:
        return

    if "Time_Rel" not in df_exec.columns:
        df = df_exec.copy()
        df["Time_Rel"] = df["Timestamp"] - t0
    else:
        df = df_exec

    needed = [
        "Base_NoPred_X_cm", "Base_NoPred_Y_cm", "Base_NoPred_Z_cm",
        "Target_Base_Raw_X_cm", "Target_Base_Raw_Y_cm", "Target_Base_Raw_Z_cm",
        "Pose_Predicted",
    ]
    if not all(c in df.columns for c in needed):
        print("⚠️ Thiếu cột cho State 3 (Base_NoPred_* hoặc Target_Base_Raw_*); bỏ qua line graph State 3.")
        return

    bnx = pd.to_numeric(df["Base_NoPred_X_cm"], errors="coerce")
    bny = pd.to_numeric(df["Base_NoPred_Y_cm"], errors="coerce")
    bnz = pd.to_numeric(df["Base_NoPred_Z_cm"], errors="coerce")
    brx = pd.to_numeric(df["Target_Base_Raw_X_cm"], errors="coerce")
    bry = pd.to_numeric(df["Target_Base_Raw_Y_cm"], errors="coerce")
    brz = pd.to_numeric(df["Target_Base_Raw_Z_cm"], errors="coerce")
    pose_pred = pd.to_numeric(df["Pose_Predicted"], errors="coerce").fillna(0)

    mask_pred = pose_pred >= 0.5
    t = pd.to_numeric(df["Time_Rel"], errors="coerce")[mask_pred]
    err3 = np.sqrt(
        (bnx[mask_pred] - brx[mask_pred]) ** 2 +
        (bny[mask_pred] - bry[mask_pred]) ** 2 +
        (bnz[mask_pred] - brz[mask_pred]) ** 2
    )

    valid = t.notna() & np.isfinite(err3)
    t = t[valid]
    err3 = err3[valid]

    if t.empty:
        print("⚠️ Không có mẫu nào với Pose_Predicted>=0.5 để vẽ State 3.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, np.zeros_like(err3), "k--", linewidth=1.0, label="Baseline: no prediction (offset = 0)")
    ax.plot(t, err3, color="darkorange", linewidth=1.5, label="State 3: |Predicted - NoPred| (cm)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Offset (cm)")
    ax.set_title("State 3: No Filter & With Predictor – prediction offset vs no-pred", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout(pad=0.8)
    fig.canvas.draw()
    save_path = "benchmark_state3_prediction_offset.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    print(f"✅ Đã lưu line graph State 3 vào: {os.path.abspath(save_path)}")
    plt.show()


def plot_comparative_histogram(df_exec, t0, exec_file_label="pbvs_metrics.csv"):
    """
    Four-state comparative histogram (tracking error vs target):
      State 1: No Filter & No Predictor (Raw)     — Raw_Vision vs target
      State 2: With Filter & No Predictor         — Command (Pose_Predicted<0.5) vs target
      State 3: No Filter & With Predictor         — Raw+V·Δt vs target (or Raw if not logged)
      State 4: With Filter & With Predictor       — Command (Pose_Predicted≥0.5) vs target
    In this pipeline Raw = Target_Base_Raw, so State 1 (and State 3 when using Raw) error = 0.
    """
    if df_exec is None or df_exec.empty:
        return

    # Time_Rel nếu chưa có (để hiển thị trong title)
    if "Time_Rel" not in df_exec.columns:
        df_exec = df_exec.copy()
        df_exec["Time_Rel"] = df_exec["Timestamp"] - t0

    # Dùng toàn bộ bảng dữ liệu (full log)
    df = df_exec.copy()
    t_min = float(df["Time_Rel"].min())
    t_max = float(df["Time_Rel"].max())

    # Cột bắt buộc
    required = [
        "Command_X", "Command_Y", "Command_Z",
        "Raw_Vision_X", "Raw_Vision_Y", "Raw_Vision_Z",
        "Target_X", "Target_Y", "Target_Z",
        "Pose_Predicted",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"⚠️ Thiếu cột: {missing}. Bỏ qua histogram.")
        return

    # Ưu tiên target trong base frame (cùng frame với Command/Raw) để sai số là tracking error thật
    base_target_cols = ["Target_Base_Raw_X_cm", "Target_Base_Raw_Y_cm", "Target_Base_Raw_Z_cm"]
    use_base_target = all(c in df.columns for c in base_target_cols)
    if use_base_target:
        for c in base_target_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        tgt_x, tgt_y, tgt_z = df["Target_Base_Raw_X_cm"], df["Target_Base_Raw_Y_cm"], df["Target_Base_Raw_Z_cm"]
        frame_note = " (base frame)"
    else:
        tgt_x = pd.to_numeric(df["Target_X"], errors="coerce").fillna(0)
        tgt_y = pd.to_numeric(df["Target_Y"], errors="coerce").fillna(0)
        tgt_z = pd.to_numeric(df["Target_Z"], errors="coerce").fillna(0)
        frame_note = " (board vs base — large offset)"

    # Ép kiểu số; tránh xóa hết hàng
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Pose_Predicted"] = df["Pose_Predicted"].fillna(0)
    for k in ["Command_X", "Command_Y", "Command_Z", "Raw_Vision_X", "Raw_Vision_Y", "Raw_Vision_Z"]:
        if k in df.columns:
            df[k] = df[k].fillna(0)

    # With filter: tracking error = distance(Command, target)
    df["Err_With_Filter"] = np.sqrt(
        (df["Command_X"] - tgt_x) ** 2 + (df["Command_Y"] - tgt_y) ** 2 + (df["Command_Z"] - tgt_z) ** 2
    )
    # No filter: tracking error = distance(Raw, target). In this pipeline Raw = Target_Base_Raw, so this is 0.
    df["Err_No_Filter"] = np.sqrt(
        (df["Raw_Vision_X"] - tgt_x) ** 2 + (df["Raw_Vision_Y"] - tgt_y) ** 2 + (df["Raw_Vision_Z"] - tgt_z) ** 2
    )
    # Chỉ bỏ hàng không có giá trị error hợp lệ (inf/nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Err_With_Filter", "Err_No_Filter"]).reset_index(drop=True)

    # Four states: split by filter (Command vs Raw) and predictor (Pose_Predicted ≥ 0.5)
    has_pred = (df["Pose_Predicted"] >= 0.5).astype(bool)
    # State 1: No filter, no predictor → Raw vs target
    state1_err = df.loc[~has_pred, "Err_No_Filter"].dropna()
    # State 2: With filter, no predictor → Command vs target
    state2_err = df.loc[~has_pred, "Err_With_Filter"].dropna()
    # State 3: No filter, with predictor → Raw vs target (Raw+V·Δt not in log; use Raw)
    state3_err = df.loc[has_pred, "Err_No_Filter"].dropna()
    # State 4: With filter, with predictor → Command vs target
    state4_err = df.loc[has_pred, "Err_With_Filter"].dropna()

    # Bin chung cho tất cả histogram
    all_vals = pd.concat([state1_err, state2_err, state3_err, state4_err])
    if all_vals.empty:
        bins = np.linspace(0, 20, 40)
        print("⚠️ Không có dữ liệu hợp lệ để vẽ histogram; vẽ khung trống với thông báo.")
    else:
        data_max = float(all_vals.max())
        bin_max = max(data_max * 1.05, 0.1)  # cover full data range; cap only for huge outliers
        bin_max = min(bin_max, 200)  # 200 cm max axis
        bins = np.linspace(0, bin_max, 40)

    no_pred_count = (~has_pred).sum()
    if no_pred_count == 0:
        print("ℹ️  Log has only predictor-ON samples (Pose_Predicted≥0.5); State 2 & State 1 (no-pred) panels show N/A.")
    no_pred_note = " (No predictor-OFF samples)" if no_pred_count == 0 else ""
    print("📊 Four-state histogram data sources:" + FOUR_STATES_SUMMARY)

    fig2, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig2.suptitle(
        f"Four states: Tracking error (3D) vs target{frame_note} | Full log ({t_min:.1f}–{t_max:.1f} s, n={len(df)}){no_pred_note}",
        fontweight="bold",
        fontsize=12,
    )

    x_max = float(bins[-1]) if len(bins) else 20
    empty_no_pred_msg = "No predictor-OFF samples\nin this log"
    raw_equals_target_msg = "Raw = Target (base) in this pipeline\n→ tracking error = 0"

    def _draw_hist(ax, series, color, title, xlabel=False, empty_msg="No data", all_zero_msg=None):
        if series is not None and not series.empty:
            if all_zero_msg is not None and float(series.max()) < 1e-6:
                ax.text(0.5, 0.5, all_zero_msg, ha="center", va="center", transform=ax.transAxes, fontsize=9, wrap=True)
                ax.set_xlim(0, x_max)
                ax.set_ylim(0, 1)
            else:
                ax.hist(series, bins=bins, color=color, alpha=0.7, edgecolor="black", density=True)
                ax.axvline(series.mean(), color="red", linestyle="--", label=f"Mean={series.mean():.3f} cm")
                ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, empty_msg, ha="center", va="center", transform=ax.transAxes, fontsize=10, wrap=True)
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_ylabel("Density")
        if xlabel:
            ax.set_xlabel("Error (cm)")
        ax.grid(True, alpha=0.3)

    # State 1: No Filter & No Predictor (Raw System)
    _draw_hist(axs[0, 0], state1_err, "crimson", "State 1: No Filter & No Predictor (Raw System)", xlabel=False, empty_msg=empty_no_pred_msg, all_zero_msg=raw_equals_target_msg)
    # State 2: With Filter & No Predictor
    _draw_hist(axs[0, 1], state2_err, "seagreen", "State 2: With Filter & No Predictor", xlabel=False, empty_msg=empty_no_pred_msg)
    # State 3: No Filter & With Predictor
    _draw_hist(axs[1, 0], state3_err, "darkorange", "State 3: No Filter & With Predictor", xlabel=True, all_zero_msg=raw_equals_target_msg)
    # State 4: With Filter & With Predictor (Complete System)
    _draw_hist(axs[1, 1], state4_err, "steelblue", "State 4: With Filter & With Predictor (Complete)", xlabel=True)

    plt.tight_layout(pad=0.8)
    fig2.canvas.draw()
    save_hist = "benchmark_histogram_comparison.png"
    plt.savefig(save_hist, bbox_inches='tight', dpi=100)
    print(f"✅ Đã lưu histogram so sánh vào: {os.path.abspath(save_hist)}")
    plt.show()

    # Overlay: all four states on one plot
    fig3, ax = plt.subplots(1, 1, figsize=(8, 5))
    for series, label, color in [
        (state1_err, "State 1: Raw (No filter, No pred)", "crimson"),
        (state2_err, "State 2: Filter, No pred", "seagreen"),
        (state3_err, "State 3: No filter, With pred", "darkorange"),
        (state4_err, "State 4: Complete (Filter + Pred)", "steelblue"),
    ]:
        if series is not None and not series.empty and float(series.max()) >= 1e-6:
            ax.hist(series, bins=bins, alpha=0.5, label=label, color=color, density=True)
    ax.set_xlabel("Tracking error vs target (cm)")
    ax.set_ylabel("Density")
    ax.set_title(f"Four states (overlay){frame_note} | Full log ({t_min:.1f}–{t_max:.1f} s, n={len(df)})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    if all_vals.empty:
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 1)
    plt.tight_layout(pad=0.8)
    fig3.canvas.draw()
    save_overlay = "benchmark_histogram_overlay.png"
    plt.savefig(save_overlay, bbox_inches='tight', dpi=100)
    print(f"✅ Đã lưu histogram overlay vào: {os.path.abspath(save_overlay)}")
    plt.show()


if __name__ == "__main__":
    # Tự động tìm file nếu có tham số dòng lệnh
    exec_path = sys.argv[1] if len(sys.argv) > 1 else "pbvs_metrics.csv"
    vis_path = sys.argv[2] if len(sys.argv) > 2 else "vision_metrics.csv"
    
    analyze_dual_logs(exec_path, vis_path)