import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ (CONFIGURATION)
# ==========================================
NUM_CLASSES = 10  
IMG_SIZE = 384     
BATCH_SIZE = 8    
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==========================================
# 2. HÀM TIỀN XỬ LÝ GỐC (EXIF CORRECTION)
# ==========================================
def correct_exif_orientation(image_path):
    """Sửa hướng ảnh tự động dựa trên thông tin EXIF (Bước 1 của pipeline)"""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    return np.array(img)

# ==========================================
# 3. ĐỊNH NGHĨA DATASET PHÂN LOẠI 
# ==========================================
class DermoscopyViTDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # (1) EXIF auto-orientation correction
        image = correct_exif_orientation(row['image_path'])
        label = int(row['label'])
        
        # Áp dụng Albumentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

# ==========================================
# 4. PIPELINE TĂNG CƯỜNG DỮ LIỆU ĐẶC THÙ ViT
# ==========================================
train_transform = A.Compose([
    # Random crop (0–30%) và đưa về kích thước 384x384 bằng nội suy Bilinear (cv2.INTER_LINEAR)
    A.RandomResizedCrop(
        height=IMG_SIZE, 
        width=IMG_SIZE, 
        scale=(0.70, 1.0), 
        interpolation=cv2.INTER_LINEAR, 
        p=1.0
    ),
    
    # (3) CLAHE từ pipeline tổng thể nhằm tăng tương phản cục bộ
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    
    A.HorizontalFlip(p=0.5),                                      # Horizontal flip
    A.VerticalFlip(p=0.5),                                        # Vertical flip
    A.RandomRotate90(p=0.5),                                      # 90° rotations
    A.Rotate(limit=15, p=0.5),                                    # Free rotation (+/- 15°)
    A.Affine(shear={'x': (-10, 10), 'y': (-10, 10)}, p=0.5),       # Shear (+/- 10°)
    A.ColorJitter(brightness=0.15, contrast=0, saturation=0, hue=0, p=0.5), # Brightness (+/- 15%)
    A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.5), p=0.5), # Gaussian blur (<= 2.5 px)
    A.GaussNoise(var_limit=(0.0, 0.0149 * 255), p=0.5),           # Noise (<= 1.49%)
    
    # (2) Channel-wise normalization (ImageNet)
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_test_transform = A.Compose([
    # Sử dụng Bilinear interpolation để đồng bộ tính nhất quán không gian ảnh
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# ==========================================
# 5. KHỞI TẠO MÔ HÌNH ViT-BASE/16
# ==========================================
def get_vit_model(num_classes=10):
    # Sử dụng trọng số IMAGENET1K_SWAG_LINEAR_V1 hỗ trợ mặc định kích thước ảnh chuẩn 384x384
    weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    model = vit_b_16(weights=weights)
    
    # Thay thế lớp phân loại cuối cùng (Heads) cho 10 lớp bệnh lý
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model

# ==========================================
# 6. HÀM ĐÁNH GIÁ CHỬ SỐ PHÂN LOẠI (METRICS)
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
    
    # Tính toán chính xác các chỉ số phân loại theo yêu cầu
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    
    print(f"\n============= KẾT QUẢ ViT TRÊN TẬP {desc.upper()} =============")
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
        print(f"Epoch {epoch+1} - Average Loss: {epoch_loss:.4f}")
        
        # Đánh giá trên tập Validation sau mỗi Epoch
        val_acc = evaluate_classification(model, val_loader, desc="Validation")
        
        # Lưu lại mô hình tối ưu nhất
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_vit_model.pth")

# ==========================================
# 8. LUỒNG CHẠY CHÍNH (MAIN EXECUTION)
# ==========================================
if __name__ == "__main__":
    # Giả lập tập dữ liệu đầu vào gồm 20,752 ảnh (thực tế thay bằng tập csv của bạn)
    # df = pd.read_csv("path_to_your_dataset.csv")
    data_mock = {
        'image_path': [f"dummy_img_{i}.jpg" for i in range(120)], 
        'label': np.random.randint(0, 10, size=120) 
    }
    df = pd.DataFrame(data_mock)
    
    # Chia Stratified Sampling theo đúng tỷ lệ đề bài: 80% Train / 15% Val / 5% Test
    train_df, val_test_df = train_test_split(
        df, test_size=0.20, stratify=df['label'], random_state=42
    )
    val_df, test_df = train_test_split(
        val_test_df, test_size=0.25, stratify=val_test_df['label'], random_state=42
    )
    
    print(f"Dữ liệu đã phân mảnh hoàn tất:")
    print(f"-> Tập Train: {len(train_df)} | Tập Validation: {len(val_df)} | Tập Test: {len(test_df)}\n")

    # Khởi tạo Datasets & Dataloaders
    train_dataset = DermoscopyViTDataset(train_df, transform=train_transform)
    val_dataset = DermoscopyViTDataset(val_df, transform=val_test_transform)
    test_dataset = DermoscopyViTDataset(test_df, transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Tải cấu trúc mô hình ViT với độ phân giải 384x384
    model = get_vit_model(num_classes=NUM_CLASSES).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    # Thường ViT hội tụ tốt hơn với bộ tối ưu AdamW cùng tốc độ học thấp (lr=3e-5 hoặc 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)

    # Thực hiện huấn luyện trong 50 Epochs
    print("Bắt đầu huấn luyện mạng Vision Transformer (ViT)...")
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS)
    
    # Tải lại trọng số tốt nhất để tiến hành nghiệm thu trên tập Test độc lập (5%)
    print("Quá trình huấn luyện kết thúc! Đang chạy thực nghiệm tập Test...")
    if os.path.exists("best_vit_model.pth"):
        model.load_state_dict(torch.load("best_vit_model.pth"))
    
    # Trích xuất toàn bộ 5 thông số phân loại cuối cùng
    evaluate_classification(model, test_loader, desc="Test")