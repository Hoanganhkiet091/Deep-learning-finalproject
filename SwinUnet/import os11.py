import os
import glob
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ (CONFIGURATION)
# ==========================================
IMG_SIZE = 224     # Cấu hình chuẩn của Swin-UNet
WINDOW_SIZE = 7    # Cửa sổ cục bộ (Window Size = 7x7) cho Swin Block
BATCH_SIZE = 8     # Điều chỉnh tùy thuộc dung lượng VRAM của GPU
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==========================================
# 2. PIPELINE TĂNG CƯỜNG DỮ LIỆU ĐỒNG THỜI (JOINT)
# ==========================================
train_transform = A.Compose([
    # Random resized crop (scale 0.85–1.15) về kích thước 224x224
    A.RandomResizedCrop(height=IMG_SIZE, width=IMG_SIZE, scale=(0.85, 1.15), p=1.0),
    
    # Các thông số tăng cường đặc thù của Swin-UNet từ bài báo
    A.HorizontalFlip(p=0.5),                                         # Horizontal flip (p=0.5)
    A.Rotate(limit=20, p=0.5),                                       # Random rotation (+/- 20°)
    A.ColorJitter(brightness=0.10, contrast=0.10, saturation=0, hue=0, p=0.5), # Intensity scaling (+/- 10%)
    A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 1.5), p=0.5), # Gaussian blur (sigma = 0.1 - 1.5)
    
    # Chuẩn hóa ảnh gốc
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# ==========================================
# 3. ĐỊNH NGHĨA DATASET PHÂN ĐOẠN 
# ==========================================
class SwinUNetDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(data_dir, split, "images", "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(data_dir, split, "masks", "*")))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Chuẩn hóa Mask về dải [0, 1] theo đặc tả bài báo
        mask = mask.float() / 255.0
        mask = torch.clamp(mask, 0.0, 1.0)
        
        return image, mask.unsqueeze(0) # Trả về dạng tensor (C=1, H=224, W=224)

# ==========================================
# 4. THUẬT TOÁN CUTMIX CHO SEGMENTATION (p=0.3)
# ==========================================
def apply_segmentation_cutmix(images, masks, p=0.3):
    """Cắt và trộn đồng thời vùng ảnh và vùng mask tương ứng giữa 2 mẫu trong một batch"""
    if random.random() > p:
        return images, masks

    batch_size = images.size(0)
    W = images.size(2)
    H = images.size(3)
    
    # Tạo chỉ mục xáo trộn ngẫu nhiên ngầm định
    rand_index = torch.randperm(batch_size).to(images.device)
    
    # Lấy tọa độ hộp cắt ngẫu nhiên
    lam = np.random.beta(1.0, 1.0) # Tỷ lệ diện tích cắt
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    
    # Tiến hành đè vùng cắt từ ảnh xáo trộn sang ảnh gốc
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    masks[:, :, bbx1:bbx2, bby1:bby2] = masks[rand_index, :, bbx1:bbx2, bby1:bby2]
    
    return images, masks

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Tâm của hộp cắt chọn ngẫu nhiên
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# ==========================================
# 5. CẤU TRÚC MÔ PHỎNG SWIN-UNET BLOCK
# ==========================================
class SwinTransformerSysBlock(nn.Module):
    """Mô phỏng một Swin Block thu gọn cơ chế Window Attention (7x7)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Sử dụng mô phỏng tính chất phân cụm theo cửa sổ Window Size = 7
        self.window_pool = nn.AvgPool2d(kernel_size=WINDOW_SIZE, stride=WINDOW_SIZE)
        self.window_upsample = nn.Upsample(scale_factor=WINDOW_SIZE, mode='nearest')

    def forward(self, x):
        res = self.conv(x)
        # Mô phỏng cơ chế tính toán trong cụm Window Shifted Attention
        w_feat = self.window_pool(res)
        w_attn = torch.sigmoid(self.window_upsample(w_feat))
        return res * w_attn

class SwinUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Encoder (Swin Stage 1, 2, 3)
        self.enc1 = SwinTransformerSysBlock(3, 96)
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = SwinTransformerSysBlock(96, 192)
        self.down2 = nn.MaxPool2d(2)
        self.enc3 = SwinTransformerSysBlock(192, 384)
        self.down3 = nn.MaxPool2d(2)
        
        # Bottleneck ở đáy
        self.bottleneck = SwinTransformerSysBlock(384, 768)
        
        # Decoder (Swin Expansion Stages) kết hợp Skip connection từ Encoder
        self.up1 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.dec1 = SwinTransformerSysBlock(384 + 384, 192)
        
        self.up2 = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2)
        self.dec2 = SwinTransformerSysBlock(192 + 192, 96)
        
        self.up3 = nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2)
        self.dec3 = SwinTransformerSysBlock(96 + 96, 64)
        
        self.final_mask = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Đường Encoder của Swin Transformer
        e1 = self.enc1(x)         # [B, 96, 224, 224]
        e2 = self.enc2(self.down1(e1)) # [B, 192, 112, 112]
        e3 = self.enc3(self.down2(e2)) # [B, 384, 56, 56]
        
        # Đáy Bottleneck
        b = self.bottleneck(self.down3(e3)) # [B, 768, 28, 28]
        
        # Đường Decoder đối xứng
        d1 = self.up1(b)
        d1 = torch.cat([d1, e3], dim=1) # Ghép skip connection tầng 3
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1) # Ghép skip connection tầng 2
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1) # Ghép skip connection tầng 1
        d3 = self.dec3(d3)
        
        return torch.sigmoid(self.final_mask(d3)) # Đầu ra chuẩn hóa dải [0, 1]

# ==========================================
# 6. HÀM ĐÁNH GIÁ CHỈ SỐ PHÂN ĐOẠN 
# ==========================================
def evaluate_swin_unet(model, loader, desc="Test"):
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    TP, FP, TN, FN = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"Evaluating {desc}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            
            preds = (outputs > 0.5).float()
            preds_np = preds.cpu().numpy().astype(int).flatten()
            masks_np = masks.cpu().numpy().astype(int).flatten()
            
            tp = np.sum((preds_np == 1) & (masks_np == 1))
            fp = np.sum((preds_np == 1) & (masks_np == 0))
            tn = np.sum((preds_np == 0) & (masks_np == 0))
            fn = np.sum((preds_np == 0) & (masks_np == 1))
            
            TP += tp; FP += fp; TN += tn; FN += fn
            
            intersection = np.sum(preds_np * masks_np)
            union = np.sum(preds_np) + np.sum(masks_np)
            
            total_dice += (2. * intersection + 1e-7) / (union + 1e-7)
            total_iou += (intersection + 1e-7) / (union - intersection + 1e-7)
            
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-7)
    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    macro_iou = total_iou / len(loader)
    
    print(f"\n============= KẾT QUẢ PHÂN ĐOẠN SWIN-UNET TẬP {desc.upper()} =============")
    print(f"1. Pixel-Accuracy (ACC)       : {acc:.4f}")
    print(f"2. Precision (PRE)            : {precision:.4f}")
    print(f"3. Recall/Sensitivity (REC)   : {recall:.4f}")
    print(f"4. Dice Coefficient / F1-Score: {f1_score:.4f}")
    print(f"5. Intersection over Union(IoU): {macro_iou:.4f}")
    print("\n6. Confusion Matrix (Pixel-Level):")
    print(f"   [ [ TN: {TN} , FP: {FP} ]\n     [ FN: {FN} , TP: {TP} ] ]")
    print("=========================================================================\n")
    return f1_score

# ==========================================
# 7. CHƯƠNG TRÌNH CHẠY CHÍNH (MAIN PROCESS)
# ==========================================
if __name__ == "__main__":
    data_directory = "data"
    
    if not os.path.exists(os.path.join(data_directory, "train")):
        print("Lỗi: Không tìm thấy thư mục 'data/train'. Vui lòng kiểm tra lại cấu trúc trong VS Code.")
        exit()

    train_dataset = SwinUNetDataset(data_directory, split="train", transform=train_transform)
    val_dataset = SwinUNetDataset(data_directory, split="val", transform=val_test_transform)
    test_dataset = SwinUNetDataset(data_directory, split="test", transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Khởi tạo Swin-UNet mạng
    model = SwinUNet(num_classes=1).to(DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print("Bắt đầu huấn luyện hệ thống Swin-UNet với cấu hình Window Size 7x7...")
    best_dice = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, masks in progress_bar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            # Áp dụng đồng thời thuật toán Cutmix lên cặp Ảnh-Mask với p=0.3
            images, masks = apply_segmentation_cutmix(images, masks, p=0.3)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} - Average Loss: {running_loss/len(train_loader):.4f}")
        
        # Kiểm thử sau mỗi Epoch trên tập Validation
        val_dice = evaluate_swin_unet(model, val_loader, desc="Validation")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_swin_unet_model.pth")

    # Đánh giá cuối cùng nghiệm thu trên tập Test độc lập
    print("Huấn luyện hoàn tất! Đang kiểm thử trên tập Test thực tế...")
    if os.path.exists("best_swin_unet_model.pth"):
        model.load_state_dict(torch.load("best_swin_unet_model.pth"))
        
    evaluate_swin_unet(model, test_loader, desc="Test")