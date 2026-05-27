"""
==============================================================================
  AeroScript Vision - Script Huấn luyện YOLOv8 tùy chỉnh
  Phát hiện bút vẽ (painting_pen) cho hệ thống UAV
==============================================================================

  Tác giả : AeroScript Team
  GPU     : NVIDIA RTX 4060 (8GB VRAM)
  Framework: Ultralytics YOLOv8

  Hướng dẫn:
    1. Chỉnh sửa biến DATASET_ROOT_DIR bên dưới nếu cần.
    2. Dataset gốc có cấu trúc:
         data/
         ├── images/    (tất cả ảnh .jpg)
         ├── labels/    (tất cả label .txt, có prefix hash)
         └── classes.txt
    3. Script sẽ TỰ ĐỘNG:
       - Rename label cho khớp tên ảnh (loại bỏ prefix hash)
       - Chia train/val (80/20)
       - Tạo file YAML
       - Bắt đầu huấn luyện
    4. Chạy: python Computervision/train_custom_yolo.py
==============================================================================
"""

import os
import sys
import shutil
import random
from pathlib import Path

import yaml

# ============================================================================
#  CẤU HÌNH - CHỈNH SỬA TẠI ĐÂY
# ============================================================================

# Đường dẫn tuyệt đối tới thư mục gốc chứa dataset (chứa images/ và labels/)
DATASET_ROOT_DIR = "/home/luongduy/AeroScript_Vision/data"

# Tỷ lệ chia train/val (0.8 = 80% train, 20% val)
TRAIN_RATIO = 0.8

# Trọng số pretrained YOLOv8
PRETRAINED_WEIGHTS = "yolov8n.pt"

# Tham số huấn luyện tối ưu cho RTX 4060 8GB VRAM
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16          # Tăng lên 32 nếu VRAM còn dư, giảm về 8 nếu OOM
DEVICE = 0               # 0 = GPU đầu tiên, "cpu" nếu không có GPU
WORKERS = 4              # Số worker DataLoader (giảm nếu gặp lỗi shared memory)
PROJECT_NAME = "runs/detect"
RUN_NAME = "aeroscript_pen_model"

# Seed cho reproducibility
RANDOM_SEED = 42

# Thông tin class
NUM_CLASSES = 1
CLASS_NAMES = ["painting_pen"]

# ============================================================================


def prepare_dataset(dataset_dir: str) -> str:
    """
    Chuẩn bị dataset: rename label cho khớp ảnh và chia train/val.

    Dataset gốc có label dạng: <hash>-<tên_ảnh>.txt
    Ảnh có dạng: <tên_ảnh>.jpg
    Script sẽ tạo cấu trúc chuẩn YOLO:
        dataset_dir/
        ├── train/
        │   ├── images/
        │   └── labels/
        └── val/
            ├── images/
            └── labels/

    Args:
        dataset_dir: Đường dẫn thư mục gốc dataset.

    Returns:
        Đường dẫn thư mục dataset đã chuẩn bị.
    """
    print("=" * 60)
    print("  [1/4] ĐANG CHUẨN BỊ VÀ CHIA DATASET...")
    print("=" * 60)

    dataset_dir = os.path.abspath(dataset_dir)

    # Kiểm tra thư mục dataset tồn tại
    if not os.path.isdir(dataset_dir):
        print(f"\n  ❌ LỖI: Không tìm thấy thư mục dataset!")
        print(f"     Đường dẫn: {dataset_dir}")
        print(f"     Hãy chỉnh sửa biến DATASET_ROOT_DIR ở đầu file.\n")
        sys.exit(1)

    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"\n  ❌ LỖI: Thiếu thư mục images/ hoặc labels/ trong dataset!")
        sys.exit(1)

    # Kiểm tra nếu đã chia train/val rồi thì bỏ qua
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    if (os.path.isdir(os.path.join(train_dir, "images")) and
        os.path.isdir(os.path.join(val_dir, "images"))):
        train_count = len([f for f in os.listdir(os.path.join(train_dir, "images"))
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        val_count = len([f for f in os.listdir(os.path.join(val_dir, "images"))
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if train_count > 0 and val_count > 0:
            print(f"\n  ✅ Dataset đã được chia trước đó!")
            print(f"     🖼️  Train: {train_count} ảnh")
            print(f"     🖼️  Val  : {val_count} ảnh")
            print(f"     ⏭️  Bỏ qua bước chia dataset.\n")
            return dataset_dir

    # Bước 1: Xây dựng mapping label -> ảnh
    print(f"\n  📂 Đang quét thư mục ảnh: {images_dir}")
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ])
    print(f"     Tìm thấy {len(image_files)} ảnh.")

    print(f"  📂 Đang quét thư mục label: {labels_dir}")
    label_files = sorted([
        f for f in os.listdir(labels_dir)
        if f.lower().endswith('.txt')
    ])
    print(f"     Tìm thấy {len(label_files)} label.")

    # Bước 2: Mapping label có prefix hash -> tên ảnh gốc
    # Label format: <hash>-<tên_ảnh_gốc>.txt -> ảnh: <tên_ảnh_gốc>.jpg
    print(f"\n  🔗 Đang mapping label với ảnh...")

    paired = []  # Danh sách (image_filename, label_filename)
    unmatched_labels = []

    # Tạo set các tên ảnh (không extension) để tra cứu nhanh
    image_stems = {}
    for img_file in image_files:
        stem = os.path.splitext(img_file)[0]
        image_stems[stem] = img_file

    for label_file in label_files:
        label_stem = os.path.splitext(label_file)[0]  # e.g., "00bad068-frame_00674"

        # Thử tách hash prefix: tìm dấu '-' đầu tiên
        if '-' in label_stem:
            # Lấy phần sau hash: "frame_00674" hoặc "IMG_2267"
            parts = label_stem.split('-', 1)
            image_stem = parts[1]  # e.g., "frame_00674"
        else:
            image_stem = label_stem

        # Xử lý trường hợp đặc biệt: IMG_22971 -> IMG_2297(1)
        if image_stem == "IMG_22971":
            image_stem = "IMG_2297(1)"

        if image_stem in image_stems:
            paired.append((image_stems[image_stem], label_file, image_stem))
        else:
            unmatched_labels.append(label_file)

    print(f"     ✅ Đã ghép thành công: {len(paired)} cặp ảnh-label")
    if unmatched_labels:
        print(f"     ⚠️  Label không tìm thấy ảnh tương ứng: {len(unmatched_labels)}")

    if len(paired) == 0:
        print(f"\n  ❌ LỖI: Không ghép được cặp ảnh-label nào!")
        sys.exit(1)

    # Bước 3: Chia train/val
    random.seed(RANDOM_SEED)
    random.shuffle(paired)

    split_idx = int(len(paired) * TRAIN_RATIO)
    train_pairs = paired[:split_idx]
    val_pairs = paired[split_idx:]

    print(f"\n  ✂️  Chia dataset (tỷ lệ {TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%}):")
    print(f"     📁 Train: {len(train_pairs)} ảnh")
    print(f"     📁 Val  : {len(val_pairs)} ảnh")

    # Bước 4: Tạo cấu trúc thư mục và copy file
    for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:
        split_img_dir = os.path.join(dataset_dir, split_name, "images")
        split_lbl_dir = os.path.join(dataset_dir, split_name, "labels")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)

        for img_file, lbl_file, img_stem in pairs:
            # Copy ảnh (giữ nguyên tên)
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(split_img_dir, img_file)
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)

            # Copy label (đổi tên cho khớp với ảnh, bỏ hash prefix)
            src_lbl = os.path.join(labels_dir, lbl_file)
            # Tên label mới = tên ảnh nhưng đuôi .txt
            new_lbl_name = os.path.splitext(img_file)[0] + ".txt"
            dst_lbl = os.path.join(split_lbl_dir, new_lbl_name)
            if not os.path.exists(dst_lbl):
                shutil.copy2(src_lbl, dst_lbl)

    print(f"\n  ✅ Đã chia dataset xong!")
    print(f"     📂 Train: {os.path.join(dataset_dir, 'train')}")
    print(f"     📂 Val  : {os.path.join(dataset_dir, 'val')}")
    print()

    return dataset_dir


def create_yaml(dataset_dir: str, output_path: str) -> str:
    """
    Tự động tạo file cấu hình YAML cho Ultralytics YOLOv8.

    Args:
        dataset_dir : Đường dẫn tuyệt đối tới thư mục gốc dataset.
        output_path : Đường dẫn file YAML sẽ được ghi ra.

    Returns:
        Đường dẫn tuyệt đối tới file YAML đã tạo.
    """
    print("=" * 60)
    print("  [2/4] ĐANG TẠO FILE CẤU HÌNH YAML...")
    print("=" * 60)

    dataset_dir = os.path.abspath(dataset_dir)

    # Kiểm tra cấu trúc thư mục con
    train_images = os.path.join(dataset_dir, "train", "images")
    val_images = os.path.join(dataset_dir, "val", "images")

    for folder in [train_images, val_images]:
        if not os.path.isdir(folder):
            print(f"\n  ❌ LỖI: Thiếu thư mục: {folder}")
            sys.exit(1)

    # Đếm số lượng ảnh
    train_count = len([
        f for f in os.listdir(train_images)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ])
    val_count = len([
        f for f in os.listdir(val_images)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ])

    # Tạo nội dung YAML
    yaml_content = {
        "path": dataset_dir,
        "train": "train/images",
        "val": "val/images",
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
    }

    # Ghi file YAML
    output_path = os.path.abspath(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            yaml_content,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    print(f"\n  ✅ Đã tạo xong file YAML!")
    print(f"     📄 File : {output_path}")
    print(f"     📂 Dataset : {dataset_dir}")
    print(f"     🖼️  Ảnh train : {train_count}")
    print(f"     🖼️  Ảnh val   : {val_count}")
    print(f"     🏷️  Classes  : {CLASS_NAMES}")
    print()

    return output_path


def train_model(yaml_path: str) -> None:
    """
    Khởi tạo và huấn luyện mô hình YOLOv8 với các tham số tối ưu cho RTX 4060.

    Args:
        yaml_path: Đường dẫn tuyệt đối tới file YAML cấu hình dataset.
    """
    print("=" * 60)
    print("  [3/4] ĐANG KHỞI TẠO MÔ HÌNH YOLOv8...")
    print("=" * 60)

    # Import Ultralytics (đặt ở đây để kiểm tra cài đặt sớm)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n  ❌ LỖI: Chưa cài đặt thư viện Ultralytics!")
        print("     Chạy lệnh: pip install ultralytics")
        sys.exit(1)

    # Kiểm tra GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"\n  🚀 GPU phát hiện: {gpu_name}")
            print(f"     💾 VRAM      : {vram:.1f} GB")
        else:
            print("\n  ⚠️  CẢNH BÁO: Không phát hiện GPU CUDA!")
            print("     Huấn luyện sẽ rất chậm trên CPU.")
            print("     Kiểm tra: nvidia-smi và cài đặt CUDA toolkit.\n")
    except ImportError:
        print("\n  ⚠️  Không thể kiểm tra GPU (thiếu PyTorch).")

    # Tải mô hình pretrained
    print(f"\n  📦 Đang tải trọng số pretrained: {PRETRAINED_WEIGHTS}")
    model = YOLO(PRETRAINED_WEIGHTS)
    print(f"  ✅ Đã tải xong mô hình YOLOv8 Nano!\n")

    # Bắt đầu huấn luyện
    print("=" * 60)
    print("  [4/4] BẮT ĐẦU KHỞI ĐỘNG GPU VÀ HUẤN LUYỆN...")
    print("=" * 60)
    print(f"\n  ⚙️  Cấu hình huấn luyện:")
    print(f"     - Epochs    : {EPOCHS}")
    print(f"     - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"     - Batch size: {BATCH_SIZE}")
    print(f"     - Device    : GPU {DEVICE}")
    print(f"     - Workers   : {WORKERS}")
    print(f"     - Project   : {PROJECT_NAME}")
    print(f"     - Run name  : {RUN_NAME}")
    print(f"\n  ⏳ Quá trình huấn luyện bắt đầu... Vui lòng chờ.\n")

    # Huấn luyện mô hình
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=RANDOM_SEED,
        deterministic=True,
        val=True,
        plots=True,
    )

    # Thông báo hoàn thành
    print("\n" + "=" * 60)
    print("  🎉 HUẤN LUYỆN HOÀN TẤT!")
    print("=" * 60)

    best_model = os.path.join(PROJECT_NAME, RUN_NAME, "weights", "best.pt")
    last_model = os.path.join(PROJECT_NAME, RUN_NAME, "weights", "last.pt")

    print(f"\n  📊 Kết quả được lưu tại:")
    print(f"     📂 Thư mục : {os.path.join(PROJECT_NAME, RUN_NAME)}/")
    print(f"     🏆 Best    : {best_model}")
    print(f"     💾 Last    : {last_model}")
    print(f"\n  💡 Sử dụng mô hình:")
    print(f'     model = YOLO("{best_model}")')
    print(f'     results = model.predict(source="image.jpg")')
    print()


# ============================================================================
#  ĐIỂM KHỞI CHẠY CHÍNH
#  Bắt buộc bọc trong if __name__ == '__main__' để tránh lỗi
#  multiprocessing trên Linux/Ubuntu khi PyTorch tạo DataLoader workers.
# ============================================================================

if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║   AeroScript Vision - Huấn luyện YOLOv8 tùy chỉnh      ║")
    print("║   Phát hiện bút vẽ (painting_pen) cho hệ thống UAV      ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # Xác định đường dẫn file YAML (cùng thư mục với script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_output = os.path.join(script_dir, "aeroscript_data.yaml")

    # Bước 1: Chuẩn bị dataset (rename label + chia train/val)
    dataset_path = prepare_dataset(DATASET_ROOT_DIR)

    # Bước 2: Tạo file YAML cấu hình dataset
    yaml_path = create_yaml(dataset_path, yaml_output)

    # Bước 3+4: Khởi tạo mô hình và huấn luyện
    train_model(yaml_path)

    print("  👋 Script kết thúc. Chúc bạn huấn luyện thành công!\n")
