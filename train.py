import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import monai
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, RandRotate90d, RandFlipd, ScaleIntensityd

# --- CONFIGURATION ---
HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "Alzheimer_Project", "data", "final")
CSV_PATH = os.path.join(HOME, "Alzheimer_Project", "data", "metadata.csv")
MODEL_SAVE_PATH = os.path.join(HOME, "Alzheimer_Project", "models", "alzheimer_cnn.pth")

# Training Settings
EPOCHS = 25
BATCH_SIZE = 4
LR = 0.0001

# --- 1. DATASET CLASS ---
class ADNI_Dataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]
        
        data = {"image": img_path}
        
        if self.transform:
            data = self.transform(data)
        
        return data["image"], torch.tensor(label, dtype=torch.long)

# --- 2. 3D CNN MODEL ---
class AlzheimerNet(nn.Module):
    def __init__(self):
        super(AlzheimerNet, self).__init__()
        
        # Feature Extraction (Convolutional Layers)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Block 2
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Block 3
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Block 4
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        
        # Classification Layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2) # 0 for CN, 1 for AD
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 3. DATA PREPARATION ---
def prepare_data():
    print("Loading Data...")
    
    # Read CSV
    df = pd.read_csv(CSV_PATH)
    id_to_group = dict(zip(df['ImageID'], df['Group']))
    
    # List files
    all_files = glob.glob(os.path.join(DATA_DIR, "*.nii.gz"))
    
    valid_files = []
    valid_labels = []
    
    for f in all_files:
        filename = os.path.basename(f)
        img_id = filename.replace(".nii.gz", "")
        
        if img_id in id_to_group:
            group = id_to_group[img_id]
            if group == 'CN':
                valid_files.append(f)
                valid_labels.append(0) # 0 = Normal
            elif group == 'AD':
                valid_files.append(f)
                valid_labels.append(1) # 1 = Alzheimer

    print(f"Found {len(valid_files)} images (CN: {valid_labels.count(0)}, AD: {valid_labels.count(1)})")
    
    return train_test_split(valid_files, valid_labels, test_size=0.2, random_state=42, stratify=valid_labels)

# --- 4. TRAINING LOOP ---
def train():
    if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH))

    X_train, X_val, y_train, y_val = prepare_data()
    
    # Data Augmentation (MONAI)
    train_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 2)),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"])
    ])

    # Loaders
    train_ds = ADNI_Dataset(X_train, y_train, transform=train_transforms)
    val_ds = ADNI_Dataset(X_val, y_val, transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")
    
    model = AlzheimerNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    print("Training Started...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {running_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()