import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ (CONFIGURATION)
# ==========================================
NUM_CLASSES = 10  
IMG_SIZE = 224     
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thống kê ImageNet chuẩn
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==========================================
# 2. HÀM TIỀN XỬ LÝ GỐC (EXIF CORRECTION)
# ==========================================
def correct_exif_orientation(image_path):
    """Sửa hướng ảnh tự động dựa trên thông tin EXIF"""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    return np.array(img)

# ==========================================
# 3. ĐỊNH NGHĨA DATASET PHÂN LOẠI (CLASSIFICATION)
# ==========================================
class DermoscopyClassificationDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        df: DataFrame chứa 'image_path' và 'label' (ID lớp từ 0 đến 9)
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # (1) EXIF auto-orientation correction
        image = correct_exif_orientation(row['image_path'])
        label = int(row['label'])
        
        # Áp dụng bộ transform (Bao gồm CLAHE, Resizing, Augmentation, Normalization)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

# ==========================================
# 4. PIPELINE TĂNG CƯỜNG DỮ LIỆU ĐẶC THÙ RESNET50
# ==========================================
train_transform = A.Compose([
    # Random resized crop (scale 0.75–1.0) đưa về kích thước 224x224
    A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.75, 1.0), p=1.0),
    
    # (3) CLAHE từ pipeline gốc tăng tương phản vùng ảnh da
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    
    # --- BỘ TĂNG CƯỜNG RIÊNG CỦA RESNET50 ---
    A.HorizontalFlip(p=0.5),                                      # Horizontal flip (p=0.5)
    A.VerticalFlip(p=0.3),                                        # Vertical flip (p=0.3)
    A.RandomRotate90(p=0.5),                                      # 90° rotations
    A.Rotate(limit=20, p=0.5),                                    # Rotation (+/- 20°)
    A.Affine(shear={'x': (-10, 10), 'y': (-10, 10)}, p=0.5),       # Affine shear (10°)
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=0.5), # ColorJitter
    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.5),              # Gaussian blur (sigma=0.1-2.0)
    A.GaussNoise(var_limit=(0.0149 * 255, 0.0149 * 255), p=0.5),  # Additive noise (0.0149)
    A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.25), # Random Erasing (p=0.25)
    
    # (2) Channel-wise normalization (ImageNet)
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# ==========================================
# 5. KHỞI TẠO MÔ HÌNH RESNET50 TRUYỀN THỐNG
# ==========================================
def get_resnet50_model(num_classes=10):
    # Sử dụng weights mặc định mới nhất của PyTorch thay thế cho biến pretrained cũ
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Thay thế lớp Fully Connected cuối cùng phù hợp với 10 lớp bệnh
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# ==========================================
# 6. HÀM ĐÁNH GIÁ CHỈ SỐ PHÂN LOẠI (METRICS)
# ==========================================
def evaluate_classification(model, data_loader, desc="Test"):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Evaluating {desc}"):
            images = images.to(DEVICE)
            outputs = model(images)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Tính toán các chỉ số phân loại theo yêu cầu
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    
    print(f"\n============= KẾT QUẢ PHÂN LOẠI TRÊN TẬP {desc.upper()} =============")
    print(f"1. Accuracy (ACC)      : {acc:.4f}")
    print(f"2. Precision (PRE)     : {precision:.4f} (Macro-average)")
    print(f"3. Recall (REC)        : {recall:.4f} (Macro-average)")
    print(f"4. F1-Score (F1)       : {f1:.4f} (Macro-average)")
    print("\n5. Confusion Matrix :")
    print(cm)
    print("=================================================================\n")
    return acc

# ==========================================
# 7. VÒNG LẶP HUẤN LUYỆN
# ==========================================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Kết thúc - Average Loss: {epoch_loss:.4f}")
        
        # Đánh giá trên tập Validation sau mỗi Epoch
        val_acc = evaluate_classification(model, val_loader, desc="Validation")
        
        # Lưu lại checkpoint tốt nhất dựa trên accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_resnet50_model.pth")

# ==========================================
# 8. LUỒNG CHẠY CHÍNH (MAIN EXECUTION)
# ==========================================
if __name__ == "__main__":
    # Giả lập lại DataFrame chứa 20,752 ảnh cấu trúc tương tự bài báo đề cập
    # Thật tế bạn thay thế bằng: df = pd.read_csv("path_to_your_dataset.csv")
    data_mock = {
        'image_path': [f"dummy_img_{i}.jpg" for i in range(200)], 
        'label': np.random.randint(0, 10, size=200) 
    }
    df = pd.DataFrame(data_mock)
    
    # Tách Stratified Sampling theo đúng tỷ lệ 80% / 15% / 5% dựa vào cột nhãn bệnh 'label'
    train_df, val_test_df = train_test_split(
        df, test_size=0.20, stratify=df['label'], random_state=42
    )
    val_df, test_df = train_test_split(
        val_test_df, test_size=0.25, stratify=val_test_df['label'], random_state=42
    )
    
    print(f"Phân chia tập dữ liệu thành công:")
    print(f"-> Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}\n")

    # Khởi tạo các Dataset và Dataloader
    train_dataset = DermoscopyClassificationDataset(train_df, transform=train_transform)
    val_dataset = DermoscopyClassificationDataset(val_df, transform=val_test_transform)
    test_dataset = DermoscopyClassificationDataset(test_df, transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Khởi tạo mô hình mạng ResNet50
    model = get_resnet50_model(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Định nghĩa Hàm Loss (CrossEntropyLoss dành cho bài toán phân loại đa lớp) & Bộ tối ưu
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Thực hiện Train 50 Epochs
    print("Bắt đầu quá trình huấn luyện ResNet50...")
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS)
    
    # Tải lại trọng số tốt nhất để chạy đánh giá cuối cùng trên tập Test độc lập (5%)
    print("Huấn luyện hoàn tất! Đang tải checkpoint tốt nhất để đánh giá trên tập Test...")
    if os.path.exists("best_resnet50_model.pth"):
        model.load_state_dict(torch.load("best_resnet50_model.pth"))
    
    # In toàn bộ 5 chỉ số phân loại cuối cùng
    evaluate_classification(model, test_loader, desc="Test")