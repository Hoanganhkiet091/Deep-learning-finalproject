import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import log_loss

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ (CONFIGURATION)
# ==========================================
IMG_SIZE = 224     # Độ phân giải yêu cầu của TransUNet
BATCH_SIZE = 8     # Điều chỉnh tùy thuộc vào dung lượng VRAM GPU
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==========================================
# 2. PIPELINE TĂNG CƯỜNG DỮ LIỆU TRANSUNET
# ==========================================
# Đối với Segmentation, các phép biến đổi hình học phải áp dụng ĐỒNG THỜI lên cả Image và Mask
train_transform = A.Compose([
    # Random resized crop (scale 0.8–1.2) đưa về kích thước 224x224
    A.RandomResizedCrop(height=IMG_SIZE, width=IMG_SIZE, scale=(0.8, 1.2), p=1.0),
    
    # Các phép tăng cường theo yêu cầu bài báo
    A.HorizontalFlip(p=0.5),                                      # Horizontal flip (p=0.5)
    A.Rotate(limit=20, p=0.5),                                    # Random rotation (+/- 20°)
    A.GaussNoise(var_limit=(0.0, (0.1**2) * 255), p=0.5),         # Additive Gaussian noise (sigma <= 0.1)
    A.ColorJitter(brightness=0.1, contrast=0, saturation=0, hue=0, p=0.5), # Brightness (+/- 10%)
    
    # Elastic deformation (alpha = 10, sigma = 3) mô phỏng biến dạng mô tế bào
    A.ElasticTransform(alpha=10, sigma=3, alpha_affine=10, p=0.5),
    
    # Chuẩn hóa ảnh gốc theo ImageNet
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# ==========================================
# 3. ĐỊNH NGHĨA DATASET PHÂN ĐOẠN (SEGMENTATION)
# ==========================================
class MedicalSegmentationDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(data_dir, split, "images", "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(data_dir, split, "masks", "*")))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Đọc ảnh gốc (RGB) và ảnh mask (Grayscale)
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Chuẩn hóa Mask về dải [0, 1] theo yêu cầu bài báo
        mask = mask.float() / 255.0
        mask = torch.clamp(mask, 0.0, 1.0)
        
        return image, mask.unsqueeze(0) # Trả về dạng (Channel=1, H, W)

# ==========================================
# 4. KIẾN TRÚC MÔ HÌNH TRANSUNET (MINI VERSION)
# ==========================================
class EncoderBottleneck(nn.Module):
    """ Mô phỏng ViT Transformer Encoder bẻ nhỏ ảnh thành các Patch 16x16 """
    def __init__(self, in_channels=512, out_channels=512):
        super().__init__()
        # Kích thước patch 16x16 từ ảnh 224x224 tạo ra 14x14 = 196 tokens
        self.patch_embed = nn.Conv2d(in_channels, out_channels, kernel_size=14, stride=14)
        self.ln = nn.LayerNorm(out_channels)
        # Khối Self-Attention thu nhỏ
        self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=8, batch_first=True)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=14, stride=14)

    def forward(self, x):
        # x: [B, 512, 14, 14]
        b, c, h, w = x.shape
        feat = self.patch_embed(x) # [B, 512, 1, 1] -> Token hóa
        feat = feat.flatten(2).transpose(1, 2) # [B, 1, 512]
        
        # Áp dụng cơ chế Attention mô phỏng ViT
        attn_out, _ = self.attn(feat, feat, feat)
        feat = feat + attn_out
        
        feat = feat.transpose(1, 2).view(b, c, 1, 1)
        out = self.up(feat) # Khôi phục lại kích thước không gian [B, 512, 14, 14]
        return out

class DecoderBlock(nn.Module):
    """ Khối Decoder của UNet kết hợp Skip Connection """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1) # Ghép nối cấu trúc Skip-connection
        return self.conv(x)

class TransUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # CNN Feature Extractor (Giảm độ phân giải đồng thời lấy các lớp Skip Connection)
        self.inc = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.down1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        
        # ViT Transformer Encoder nằm ở đáy (Bottleneck) tạo ra 196 tokens
        self.vit_bottleneck = EncoderBottleneck(512, 512)
        
        # UNet Decoder khôi phục mặt nạ ảnh
        self.dec1 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec3 = DecoderBlock(128, 64, 64)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # CNN Front-end
        s1 = self.inc(x)         # [B, 64, 224, 224]
        s2 = self.down1(s1)      # [B, 128, 112, 112]
        s3 = self.down2(s2)      # [B, 256, 56, 56]
        s4 = self.down3(s3)      # [B, 512, 28, 28] -> Pool tiếp sẽ xuống 14x14 (196 patches)
        
        # Đáy ViT Encoder
        s4_pool = nn.functional.max_pool2d(s4, 2) # [B, 512, 14, 14]
        b = self.vit_bottleneck(s4_pool)         # [B, 512, 14, 14]
        b = nn.functional.upsample(b, scale_factor=2, mode='bilinear', align_corners=True) # [B, 512, 28, 28]
        
        # UNet Decoder kết hợp các nhánh s3, s2, s1
        d1 = self.dec1(b, s3)    # [B, 256, 56, 56]
        d2 = self.dec2(d1, s2)   # [B, 128, 112, 112]
        d3 = self.dec3(d2, s1)   # [B, 64, 224, 224]
        
        return torch.sigmoid(self.out_conv(d3)) # Đầu ra nhị phân thuộc dải [0, 1]

# ==========================================
# 5. HÀM ĐÁNH GIÁ CÁC CHỈ SỐ PHÂN ĐOẠN (METRICS)
# ==========================================
def evaluate_segmentation(model, loader, desc="Test"):
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    
    # Các biến tính chỉ số Confusion Matrix cấp độ Pixel
    TP, FP, TN, FN = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"Evaluating {desc}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            
            # Ngưỡng hóa (Thresholding) đưa về nhãn nhị phân 0 hoặc 1
            preds = (outputs > 0.5).float()
            
            # Chuyển thành dạng phẳng mảng 1 chiều tính toán Scikit-Learn chỉ số
            preds_np = preds.cpu().numpy().astype(int).flatten()
            masks_np = masks.cpu().numpy().astype(int).flatten()
            
            # Tính toán ma trận nhầm lẫn pixel-wise
            tp = np.sum((preds_np == 1) & (masks_np == 1))
            fp = np.sum((preds_np == 1) & (masks_np == 0))
            tn = np.sum((preds_np == 0) & (masks_np == 0))
            fn = np.sum((preds_np == 0) & (masks_np == 1))
            
            TP += tp; FP += fp; TN += tn; FN += fn
            
            # Tính toán cục bộ Dice và IoU từng cặp ảnh
            intersection = np.sum(preds_np * masks_np)
            union = np.sum(preds_np) + np.sum(masks_np)
            
            dice = (2. * intersection + 1e-7) / (union + 1e-7)
            iou = (intersection + 1e-7) / (union - intersection + 1e-7)
            
            total_dice += dice
            total_iou += iou
            
    # Tính toán toàn cục hệ thống chỉ số phân loại nhị phân trên từng pixel
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-7)
    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    macro_iou = total_iou / len(loader)
    
    # Dựng cấu trúc Confusion Matrix hiển thị trực quan
    cm = np.array([[TN, FP], [FN, TP]])
    
    print(f"\n============= KẾT QUẢ PHÂN ĐOẠN TRANSUNET TẬP {desc.upper()} =============")
    print(f"1. Pixel-Accuracy (ACC)       : {acc:.4f}")
    print(f"2. Precision (PRE)            : {precision:.4f}")
    print(f"3. Recall/Sensitivity (REC)   : {recall:.4f}")
    print(f"4. Dice Coefficient / F1-Score: {f1_score:.4f} (Trùng với F1-Score)")
    print(f"5. Intersection over Union(IoU): {macro_iou:.4f}")
    print("\n6. Confusion Matrix (Pixel-Level):")
    print(f"   [ [ TN: {TN} , FP: {FP} ]")
    print(f"     [ FN: {FN} , TP: {TP} ] ]")
    print("=========================================================================\n")
    return f1_score

# ==========================================
# 6. VÒNG LẶP HUẤN LUYỆN CHÍNH (MAIN LOOP)
# ==========================================
if __name__ == "__main__":
    # Đọc cấu trúc thư mục data/
    data_directory = "data"
    
    # Kiểm tra xem thư mục dữ liệu thực tế có tồn tại hay không
    if not os.path.exists(os.path.join(data_directory, "train")):
        print(f"Lỗi: Không tìm thấy thư mục 'data/train'. Hãy tạo cấu trúc thư mục như hướng dẫn.")
        exit()

    # Khởi tạo DataLoader
    train_dataset = MedicalSegmentationDataset(data_directory, split="train", transform=train_transform)
    val_dataset = MedicalSegmentationDataset(data_directory, split="val", transform=val_test_transform)
    test_dataset = MedicalSegmentationDataset(data_directory, split="test", transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Khởi tạo mô hình mạng
    model = TransUNet(num_classes=1).to(DEVICE)
    
    # Hàm Loss Phân đoạn (Sử dụng BCELoss cho nhãn dải [0,1])
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print("Bắt đầu huấn luyện hệ thống TransUNet...")
    best_dice = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, masks in progress_bar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} - Average Loss: {running_loss/len(train_loader):.4f}")
        
        # Đánh giá sau mỗi epoch trên tập Validation
        val_dice = evaluate_segmentation(model, val_loader, desc="Validation")
        
        # Lưu checkpoint mô hình dựa theo chỉ số Dice tối ưu nhất
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_transunet_model.pth")

    # ==========================================
    # 7. ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST
    # ==========================================
    print("Huấn luyện hoàn tất! Tiến hành tải trọng số tốt nhất chạy tập Test thực nghiệm...")
    if os.path.exists("best_transunet_model.pth"):
        model.load_state_dict(torch.load("best_transunet_model.pth"))
        
    evaluate_segmentation(model, test_loader, desc="Test")