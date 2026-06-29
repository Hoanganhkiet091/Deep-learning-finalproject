import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import timm
from timm.data import Mixup
from timm.data.utils import RASampler
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ (CONFIGURATION)
# ==========================================
NUM_CLASSES = 10  
IMG_SIZE = 224     # Độ phân giải chuẩn của DeiT-Base
BATCH_SIZE = 16    # Điều chỉnh tùy theo dung lượng VRAM GPU của bạn
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thống kê chuẩn ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==========================================
# 2. PIPELINE TĂNG CƯỜNG DỮ LIỆU ĐẶC THÙ DeiT
# ==========================================

train_transform = transforms.Compose([
    # Random resized crop (scale 0.08–1.0, ratio 0.75–1.33)
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
    # Lật ảnh ngang và dọc
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    # RandAugment (magnitude 9, num_ops 2)
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    # Chuẩn hóa ImageNet
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    # Random Erasing (p=0.25)
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
train_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"

train_dataset = ImageFolder(root=train_dir, transform=train_transform)
val_dataset = ImageFolder(root=val_dir, transform=val_test_transform)
test_dataset = ImageFolder(root=test_dir, transform=val_test_transform)

# Áp dụng Repeated Augmentation (ra_sampler=2) bằng cách sử dụng RASampler từ timm
train_sampler = RASampler(train_dataset, num_repeats=2, shuffle=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Khởi tạo Mixup (alpha=0.8), CutMix (alpha=1.0) và Label Smoothing (epsilon=0.1)
mixup_fn = Mixup(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,               # Kích hoạt thực thi Mixup/CutMix trên từng batch
    switch_prob=0.5,        # Tỷ lệ hoán đổi 50/50 giữa Mixup và CutMix
    mode='batch',
    label_smoothing=0.1,    # Khớp nhãn mịn (𝜖 = 0.1)
    num_classes=NUM_CLASSES
)

print(f"Khởi tạo dữ liệu thành công! Số lượng mẫu Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# ==========================================
# 4. KHỞI TẠO MÔ HÌNH DeiT-BASE
# ==========================================
# Khởi tạo kiến trúc deit_base_patch16_224 tự động chia ảnh thành 196 patches (14x14 grid)
model = timm.create_model('deit_base_patch16_224', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# Khi sử dụng Mixup/CutMix, nhãn đích trở thành dạng Soft-target (vector xác suất thay vì số nguyên đơn lẻ)
criterion_train = SoftTargetCrossEntropy() 
criterion_val = nn.CrossEntropyLoss()      # Dành cho validation và test không dùng Mixup

optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)

# ==========================================
# 5. HÀM ĐÁNH GIÁ PHÂN LOẠI (METRICS)
# ==========================================
def evaluate_deit(model, data_loader, desc="Test"):
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
    
    # Tính toán chính xác 5 chỉ số phân loại theo yêu cầu
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    
    print(f"\n============= KẾT QUẢ DeiT TRÊN TẬP {desc.upper()} =============")
    print(f"1. Accuracy (ACC)      : {acc:.4f}")
    print(f"2. Precision (PRE)     : {precision:.4f} (Macro-average)")
    print(f"3. Recall (REC)        : {recall:.4f} (Macro-average)")
    print(f"4. F1-Score (F1)       : {f1:.4f} (Macro-average)")
    print("\n5. Confusion Matrix :")
    print(cm)
    print("=================================================================\n")
    return acc

# ==========================================
# 6. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
# ==========================================
if __name__ == "__main__":
    print("Bắt đầu huấn luyện DeiT-Base...")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        # Trong kiến trúc RASampler, việc set epoch ở mỗi đầu vòng lặp là bắt buộc để thay đổi cách lấy mẫu ngẫu nhiên
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Áp dụng cơ chế trộn ảnh Mixup / CutMix lên batch hiện tại
            images_mixed, targets_mixed = mixup_fn(images, labels)
            
            optimizer.zero_grad()
            outputs = model(images_mixed)
            loss = criterion_train(outputs, targets_mixed)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Kết Thúc - Average Loss: {epoch_loss:.4f}")
        
        # Đánh giá hiệu năng Validation sau mỗi epoch
        val_acc = evaluate_deit(model, val_loader, desc="Validation")
        
        # Lưu lại mô hình đạt kết quả chính xác cao nhất
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_deit_model.pth")

    # ==========================================
    # 7. NGHIỆM THU CUỐI CÙNG TRÊN TẬP TEST
    # ==========================================
    print("Huấn luyện hoàn tất! Tiến hành tải trọng số tối ưu nhất để kiểm thử tập Test...")
    if os.path.exists("best_deit_model.pth"):
        model.load_state_dict(torch.load("best_deit_model.pth"))
        
    evaluate_deit(model, test_loader, desc="Test")