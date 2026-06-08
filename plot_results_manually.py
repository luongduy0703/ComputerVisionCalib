import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Đường dẫn tới file csv kết quả vừa train xong
    csv_path = '/home/luongduy/AeroScript_Vision/runs/pose/Aero_Models/pen_pose_v4-3/results.csv'
    output_dir = os.path.dirname(csv_path)
    
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file kết quả tại: {csv_path}")
        return
        
    print(f"📊 Đang đọc dữ liệu từ: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Làm sạch tên cột (loại bỏ khoảng trắng thừa)
    df.columns = [c.strip() for c in df.columns]
    
    # Tạo đồ thị
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AeroScript Pose V4 — Training Metrics', fontsize=16, fontweight='bold')
    
    # 1. Vẽ Box Loss (Train vs Val)
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue')
    if 'val/box_loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='orange', linestyle='--')
    axes[0, 0].set_title('Bounding Box Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Vẽ Pose Loss (Train vs Val)
    axes[0, 1].plot(df['epoch'], df['train/pose_loss'], label='Train Pose Loss', color='green')
    if 'val/pose_loss' in df.columns:
        axes[0, 1].plot(df['epoch'], df['val/pose_loss'], label='Val Pose Loss', color='red', linestyle='--')
    axes[0, 1].set_title('Keypoint Pose Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Vẽ Class Loss & DFL Loss
    axes[1, 0].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', color='purple')
    axes[1, 0].plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss', color='brown')
    axes[1, 0].set_title('Classification & DFL Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Vẽ Độ chính xác mAP (Box vs Pose)
    if 'metrics/mAP50-95(B)' in df.columns:
        axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95 (Box)', color='darkblue')
    if 'metrics/mAP50-95(Pose)' in df.columns:
        axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(Pose)'], label='mAP50-95 (Pose)', color='darkred')
    axes[1, 1].set_title('Validation mAP Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('mAP')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Lưu file ảnh biểu đồ
    save_path = os.path.join(output_dir, 'custom_results.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ Đã vẽ và lưu biểu đồ thành công tại: {save_path}")

if __name__ == '__main__':
    main()
