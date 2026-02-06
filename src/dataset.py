import os
import torch
import pandas as pd
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from src.config import PROCESSED_DIR, METADATA_PATH

class AlzheimerDataset(Dataset):
    def __init__(self, transform=None):
        """
        Prépare la liste des fichiers et des labels.
        """
        self.transform = transform
        
        # 1. Charger le CSV
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"CSV introuvable : {METADATA_PATH}")
            
        self.meta = pd.read_csv(METADATA_PATH)
        
        # 2. Dictionnaire de conversion (Texte -> Chiffre)
        # CN=0 (Sain), MCI=1 (Intermédiaire), AD=2 (Malade)
        self.label_map = {'CN': 0, 'MCI': 1, 'AD': 2}
        
        # Filtrer : On ne garde que les patients qui ont bien une image traitée
        self.valid_samples = []
        for idx, row in self.meta.iterrows():
            # Le nom du fichier traité doit commencer par "proc_"
            filename = f"proc_{row['Subject']}.nii" # Adapte si tes fichiers ont un autre nom !
            filepath = os.path.join(PROCESSED_DIR, filename)
            
            if os.path.exists(filepath):
                self.valid_samples.append((filepath, self.label_map[row['Group']]))
            
        print(f"✅ Dataset chargé : {len(self.valid_samples)} images prêtes pour l'entraînement.")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        path, label = self.valid_samples[idx]
        
        # Charger l'image Nifti
        img = nib.load(path)
        data = img.get_fdata()
        
        # Convertir en Tensor PyTorch (Ajout d'une dimension Channel : 1, 128, 128, 128)
        data_tensor = torch.from_numpy(data).float().unsqueeze(0)
        label_tensor = torch.tensor(label).long()
        
        return data_tensor, label_tensor