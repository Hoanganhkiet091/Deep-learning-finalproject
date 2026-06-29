import os
import numpy as np
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# =================================================================
# 1. CẤU HÌNH & HUẤN LUYỆN YOLOv11 CLASSIFICATION
# =================================================================
if __name__ == "__main__":
    model = YOLO("yolo11n-cls.pt") 

    print("--- BẮT ĐẦU HUẤN LUYỆN YOLOv11 ---")
    # Truyền chính xác các siêu tham số tăng cường (Augmentation) theo yêu cầu bài báo
    model.train(
        data="data",            # Đường dẫn đến thư mục gốc chứa 'train' và 'val'
        epochs=50,              # Số lượng epochs huấn luyện
        imgsz=640,              # Kích thước ảnh 640x640 (YOLO tự động áp dụng letterbox)
        batch=16,               # Kích thước batch size (tùy thuộc vào VRAM card đồ họa)
        device=0,               # Chạy trên GPU 0 (thay bằng 'cpu' nếu không có card rời)
        
        # --- BỘ TĂNG CƯỜNG DỮ LIỆU THEO YÊU CẦU ---
        mosaic=1.0,             # Mosaic augmentation
        fliplr=0.5,             # Horizontal flips (p=0.5)
        flipud=0.5,             # Vertical flips (p=0.5)
        degrees=10,             # Rotation (+/- 10 degrees)
        hsv_h=0.015,            # HSV Jitter: Hue
        hsv_s=0.7,              # HSV Jitter: Saturation
        hsv_v=0.4,              # HSV Jitter: Value (Brightness)
        mixup=0.1,              # Mixup (p=0.1)
        copy_paste=0.1,         # Copy-Paste activation
        erasing=0.2,            # Random Erasing (p=0.2)
        
        # Lưu vết kết quả
        project="runs/classify",
        name="yolo11_train",
        rect=False              # Đảm bảo không ép ảnh chữ nhật thành vuông sai tỉ lệ, kết hợp với imgsz tạo ra letterbox
    )

    # =================================================================
    # 2. ĐÁNH GIÁ CHỈ SỐ TRÊN TẬP TEST ĐỘC LẬP (PRINT OUT METRICS)
    # =================================================================
    print("\n--- HUẤN LUYỆN HOÀN TẤT! TIẾN HÀNH ĐÁNH GIÁ TRÊN TẬP TEST ---")
    
    # Tải lại trọng số tốt nhất đã đạt được từ quá trình Train
    best_model_path = "runs/classify/yolo11_train/weights/best.pt"
    if os.path.exists(best_model_path):
        model = YOLO(best_model_path)
    
    # Lấy ánh xạ nhãn tự động từ mô hình đã học (đảm bảo đồng bộ thứ tự lớp)
    yolo_names = model.names  # Ví dụ: {0: 'class_0', 1: 'class_1', ...}
    name_to_idx = {name: idx for idx, name in yolo_names.items()}
    
    test_dir = "data/test"
    all_preds = []
    all_labels = []
    
    # Duyệt qua các thư mục con trong tập dữ liệu Test
    for class_name, class_idx in name_to_idx.items():
        class_path = os.path.join(test_dir, class_name)
        if not os.path.exists(class_path):
            continue
            
        # Thu thập toàn bộ ảnh trong thư mục lớp hiện tại
        img_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            img_paths.extend(glob(os.path.join(class_path, ext)))
            
        # Dự đoán nhãn cho từng ảnh
        for img_path in tqdm(img_paths, desc=f"Testing class: {class_name}"):
            results = model(img_path, verbose=False)
            
            # Trích xuất vị trí của lớp có xác suất cao nhất
            pred_idx = results[0].probs.top1 
            
            all_preds.append(pred_idx)
            all_labels.append(class_idx)
            
    # Chuyển đổi sang mảng numpy để tính chỉ số
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Tính toán 5 chỉ số phân loại theo yêu cầu
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(yolo_names.keys()))
    
    # In kết quả chuẩn ra màn hình Terminal của VS Code
    print("\n" + "="*30 + " KẾT QUẢ ĐÁNH GIÁ YOLOv11 " + "="*30)
    print(f"1. Accuracy (ACC)      : {acc:.4f}")
    print(f"2. Precision (PRE)     : {precision:.4f} (Macro-average)")
    print(f"3. Recall (REC)        : {recall:.4f} (Macro-average)")
    print(f"4. F1-Score (F1)       : {f1:.4f} (Macro-average)")
    print("\n5. Confusion Matrix (Ma trận nhầm lẫn):")
    print(cm)
    print("="*86 + "\n")