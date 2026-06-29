import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ (CONFIGURATION)
# ==========================================
NUM_CLASSES = 10  # 10 lớp bệnh theo bài báo
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. HÀM TIỀN XỬ LÝ (EXIF CORRECTION)
# ==========================================
def correct_exif_orientation(image_path):
    """Sửa hướng ảnh tự động dựa trên thông tin EXIF"""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    return np.array(img)

# ==========================================
# 3. ĐỊNH NGHĨA DATASET
# ==========================================
class DermoscopyDataset(Dataset):
    def __init__(self, df, transform=None, is_train=True):
        """
        df: DataFrame chứa các cột: 'image_path', 'mask_path', 'label' (nhãn lớp bệnh để phân tầng)
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # (1) EXIF auto-orientation correction
        image = correct_exif_orientation(row['image_path'])
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
        
        # Đảm bảo ảnh mask cùng kích thước gốc với ảnh gốc trước khi transform
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Áp dụng Albumentations (Bao gồm CLAHE, Resizing, Augmentation, Normalization)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()

        # (4) One-hot encoding cho ground-truth masks [0, 1]
        # Chuyển đổi mask từ dạng (H, W) sang (Num_Classes, H, W) dưới dạng một chuỗi các kênh binary
        mask_one_hot = torch.nn.functional.one_hot(mask, num_classes=NUM_CLASSES)
        mask_one_hot = mask_one_hot.permute(2, 0, 1).float()  # Cắt về dạng (C, H, W) giá trị [0, 1]

        return image, mask_one_hot

# ==========================================
# 4. PIPELINE TĂNG CƯỜNG DỮ LIỆU (ALBUMENTATIONS)
# ==========================================
# Thống kê ImageNet chuẩn
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = A.Compose([
    # (1) Kích thước U-Net: 256 x 256
    A.Resize(IMG_SIZE, IMG_SIZE),
    
    # (3) CLAHE để tăng cường độ tương phản cục bộ
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    
    # --- DATA AUGMENTATION NHƯ BÀI BÁO ---
    A.HorizontalFlip(p=0.5), # Lật ngang
    A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.0, 0.24), rotate_limit=0, p=0.5), # Zoom 0-24%
    A.ColorJitter(
        brightness=0.10,       # Brightness (+/- 10%)
        contrast=0.06,         # Exposure (+/- 6%)
        saturation=0.28,       # Saturation (+/- 28%)
        hue=0, p=0.5
    ),
    A.GaussNoise(var_limit=(0.0, 0.0022 * 255 * 255), p=0.5), # Gaussian noise (0.22%)
    A.CoarseDropout(max_holes=6, max_height=int(IMG_SIZE*0.02), max_width=int(IMG_SIZE*0.02), 
                    min_holes=6, min_height=int(IMG_SIZE*0.02), min_width=int(IMG_SIZE*0.02), 
                    fill_value=0, mask_fill_value=0, p=0.5), # Cutout (6 patches, 2% each)
    
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
# 5. KIẾN TRÚC MÔ HÌNH U-NET TRUYỀN THỐNG
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=10):
        super().__init__()
        self.downs = nn.ModuleList([
            DoubleConv(in_channels, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512)
        ])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            DoubleConv(1024, 512),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            DoubleConv(128, 64)
        ])
        self.final_conv = nn.Conv2d(64, out_channels, kernel_index=1 if hasattr(nn, 'Identity') else 1) # 1x1 conv
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_x)
            
        return self.final_conv(x)

# ==========================================
# 6. HÀM TÍNH TOÁN CÁC CHỈ SỐ ĐÁNH GIÁ (METRICS)
# ==========================================
def calculate_metrics(outputs, targets, num_classes=10):
    """
    outputs: (B, C, H, W) -> Dự đoán từ Model (Logits)
    targets: (B, C, H, W) -> Ground truth dạng One-hot
    """
    preds = torch.argmax(outputs, dim=1).flatten().cpu().numpy()
    targets_flat = torch.argmax(targets, dim=1).flatten().cpu().numpy()
    
    # 1. Confusion Matrix
    cm = confusion_matrix(targets_flat, preds, labels=list(range(num_classes)))
    
    # Tính toán Dice, IoU từng Class rồi lấy trung bình (Macro-average)
    dice_list, iou_list = [], []
    
    # Tránh chia cho 0
    smooth = 1e-6
    
    for c in range(num_classes):
        p_c = (preds == c)
        t_c = (targets_flat == c)
        
        intersection = np.sum(p_c & t_c)
        union = np.sum(p_c | t_c)
        
        iou = (intersection + smooth) / (union + smooth)
        dice = (2.0 * intersection + smooth) / (np.sum(p_c) + np.sum(t_c) + smooth)
        
        iou_list.append(iou)
        dice_list.append(dice)
        
    # 2. Pixel Accuracy
    pixel_acc = np.sum(preds == targets_flat) / len(targets_flat)
    
    return cm, np.mean(dice_list), np.mean(iou_list), pixel_acc

# ==========================================
# 7. VÒNG LẶP HUẤN LUYỆN VÀ ĐÁNH GIÁ
# ==========================================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Khớp định dạng: BCEWithLogitsLoss nhận (B, C, H, W) từ cả hai phía
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1} - Loss: {train_loss/len(train_loader):.4f}")
        
        # Kiểm tra nhanh trên tập Validation sau mỗi epoch (tùy chọn)
        validate_or_test(model, val_loader, desc="Validation")

def validate_or_test(model, data_loader, desc="Test"):
    model.eval()
    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES))
    total_dice, total_iou, total_acc = [], [], []
    
    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc=f"Evaluating {desc}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            
            cm, dice, iou, acc = calculate_metrics(outputs, masks, NUM_CLASSES)
            total_cm += cm
            total_dice.append(dice)
            total_iou.append(iou)
            total_acc.append(acc)
            
    print(f"\n============= KẾT QUẢ TRÊN TẬP {desc.upper()} =============")
    print(f"1. Dice Coefficient: {np.mean(total_dice):.4f}")
    print(f"2. Intersection over Union (IoU): {np.mean(total_iou):.4f}")
    print(f"3. Pixel Accuracy: {np.mean(total_acc):.4f}")
    print("\n4. Confusion Matrix:")
    print(total_cm)
    print("======================================================\n")

# ==========================================
# 8. HÀM CHẠY CHÍNH (MAIN EXECUTION)
# ==========================================
if __name__ == "__main__":
    # --- MÔ PHỎNG DỮ LIỆU ĐẦU VÀO (Bạn thay thế bằng dữ liệu thật của bạn nhé) ---
    # Giả sử bạn có 20,752 ảnh như bài báo, kèm cột 'label' chứa 10 loại bệnh để chia Stratified
    # df = pd.read_csv("path_to_your_dataset.csv")
    
    # Tạo dữ liệu giả lập cấu trúc để code chạy demo được:
    data_mock = {
        'image_path': [f"dummy_img_{i}.jpg" for i in range(100)], # Thay bằng đường dẫn thực tế
        'mask_path': [f"dummy_mask_{i}.png" for i in range(100)], # Thay bằng đường dẫn thực tế
        'label': np.random.randint(0, 10, size=100) # Nhãn phân lớp bệnh [0-9] để Stratified split
    }
    df = pd.DataFrame(data_mock)
    
    # --- CHIA TẬP DỮ LIỆU: Phân tầng (Stratified Sampling) theo tỷ lệ Tỷ lệ: 80% / 15% / 5% ---
    # Bước 1: Tách 80% Train, 20% còn lại (Val + Test)
    train_df, val_test_df = train_test_split(
        df, test_size=0.20, stratify=df['label'], random_state=42
    )
    # Bước 2: Tách 20% còn lại thành 15% Val và 5% Test (Tỷ lệ tương đương 15/20 = 75% cho Val)
    val_df, test_df = train_test_split(
        val_test_df, test_size=0.25, stratify=val_test_df['label'], random_state=42
    )
    
    print(f"Tổng số mẫu: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Khởi tạo các Dataset & Dataloader
    train_dataset = DermoscopyDataset(train_df, transform=train_transform)
    val_dataset = DermoscopyDataset(val_df, transform=val_test_transform)
    test_dataset = DermoscopyDataset(test_df, transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Khởi tạo mô hình, hàm Loss và Bộ tối ưu
    model = UNet(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss() # Thích hợp cho mặt nạ đã mã hóa One-hot dạng [0, 1]
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Thực hiện Train 50 Epochs
    print("Bắt đầu huấn luyện...")
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS)
    
    # Thực hiện Test cuối cùng và in ra đầy đủ Confusion Matrix + 3 chỉ số bài báo yêu cầu
    print("Huấn luyện hoàn tất. Đang đánh giá trên tập Test...")
    validate_or_test(model, test_loader, desc="Test")