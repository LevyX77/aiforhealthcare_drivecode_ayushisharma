import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, RandRotate90d, 
    RandFlipd, ScaleIntensityd, Resized, RandAffined, RandGaussianNoised
)

# --- CONFIG TASK 3 (MULTI CLASS) ---
HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "Alzheimer_Project", "data", "final")
CSV_PATH = os.path.join(HOME, "Alzheimer_Project", "data", "metadata.csv")
MODEL_SAVE_PATH = os.path.join(HOME, "Alzheimer_Project", "models", "task3_model.pth")

EPOCHS = 80
BATCH_SIZE = 4
LR = 0.0005
SPATIAL_SIZE = (96, 96, 96)

# --- DATASET ---
class ADNI_Dataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        data = {"image": self.files[idx]}
        if self.transform:
            data = self.transform(data)
        return data["image"], torch.tensor(self.labels[idx], dtype=torch.long)

# --- RESNET BLOCKS ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# --- MODEL (MULTI-CLASS OUTPUT) ---
class MediumResNet(nn.Module):
    def __init__(self):
        super(MediumResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.layer4 = self._make_layer(128, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3) # Important: 3 classes (CN, MCI, AD)
        )

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

# --- TRAIN ---
def train():
    print("STARTING TASK 3 (CN vs MCI vs AD)...")
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Data
    df = pd.read_csv(CSV_PATH)
    mapping = {'CN': 0, 'MCI': 1, 'AD': 2} 
    df = df[df['Group'].isin(mapping.keys())]
    id_to_label = dict(zip(df['ImageID'], df['Group'].map(mapping)))
    
    all_files = glob.glob(os.path.join(DATA_DIR, "*.nii.gz"))
    valid_files, valid_labels = [], []
    for f in all_files:
        img_id = os.path.basename(f).replace(".nii.gz", "")
        if img_id in id_to_label:
            valid_files.append(f)
            valid_labels.append(int(id_to_label[img_id]))

    print(f"Images found: {len(valid_files)}")

    X_train, X_val, y_train, y_val = train_test_split(
        valid_files, valid_labels, test_size=0.2, random_state=42, stratify=valid_labels
    )
    
    # Weights for 3 classes
    counts = [y_train.count(i) for i in [0, 1, 2]]
    weights = torch.tensor([1.0/c for c in counts], dtype=torch.float).cuda()
    weights = weights / weights.sum() * 3

    train_transforms = Compose([
        LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=SPATIAL_SIZE),
        ScaleIntensityd(keys=["image"]),
        RandRotate90d(keys=["image"], prob=0.6, spatial_axes=(0, 2)), 
        RandFlipd(keys=["image"], prob=0.6),
        RandAffined(keys=["image"], prob=0.4, rotate_range=(0.15), scale_range=(0.15)),
        RandGaussianNoised(keys=["image"], prob=0.3, std=0.02)
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=SPATIAL_SIZE),
        ScaleIntensityd(keys=["image"])
    ])

    train_ds = ADNI_Dataset(X_train, y_train, transform=train_transforms)
    val_ds = ADNI_Dataset(X_val, y_val, transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    device = torch.device("cuda")
    model = MediumResNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        train_loss = 0.0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            _, pred = torch.max(outputs, 1)
            total += lbls.size(0)
            correct += (pred == lbls).sum().item()
            train_loss += loss.item()

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
                v_total += lbls.size(0)
                v_correct += (pred == lbls).sum().item()

        val_acc = 100*v_correct/v_total
        scheduler.step(epoch + val_acc/100)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   NEW RECORD: {best_acc:.2f}%")

    print(f"Final Score: {best_acc:.2f}%")

if __name__ == "__main__": train()
