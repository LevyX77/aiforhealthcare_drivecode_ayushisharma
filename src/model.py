import torch
import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(Simple3DCNN, self).__init__()
        
        # Bloc 1 : Extraction de formes simples
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # Bloc 2 : Extraction de formes complexes
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool3d(2, 2)
        
        # Bloc 3 : Classification
        self.flatten = nn.Flatten()
        
        # Calcul de la taille après les convolutions (pour 128x128x128 en entrée)
        # Après 2 poolings (divisé par 4 au total) : 128 -> 64 -> 32
        # Taille finale : 64 canaux * 32 * 32 * 32
        self.fc1 = nn.Linear(64 * 32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes) # Sortie : 3 classes

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x