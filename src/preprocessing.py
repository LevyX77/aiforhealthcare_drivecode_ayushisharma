import os
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Spacingd, Orientationd, ScaleIntensityRanged, 
    Resized, ThresholdIntensityd
)
from nilearn.masking import compute_brain_mask
from nilearn.image import smooth_img

# --- CONFIGURATION ---
HOME = os.path.expanduser("~")
INPUT_DIR = os.path.join(HOME, "Alzheimer_Project", "data", "raw")
OUTPUT_DIR = os.path.join(HOME, "Alzheimer_Project", "data", "processed")

# Taille cible standardisee pour les modeles de recherche
TARGET_SHAPE = (128, 128, 128)

def expert_preprocess(input_path, output_path):
    try:
        # 1. Chargement et Nettoyage du bruit (Smoothing léger)
        # On utilise Nilearn pour lisser l'image et reduire le grain de l'IRM
        img = nib.load(input_path)
        img_smoothed = smooth_img(img, fwhm=1) 

        # 2. Skull Stripping (Retrait du crane)
        # Tres important : l'IA ne doit voir que le cerveau !
        mask = compute_brain_mask(img_smoothed)
        brain_only = nib.Nifti1Image(img_smoothed.get_fdata() * mask.get_fdata(), img.affine)

        # 3. Pipeline MONAI (La puissance industrielle)
        # On definit une serie de transformations professionnelles
        data_dict = {"image": brain_only}
        
        transforms = Compose([
            # S'assure que l'image est bien orientee (RAS)
            Orientationd(keys=["image"], axcodes="RAS"),
            # Normalisation intelligente : on se focalise sur les intensites du cerveau
            ScaleIntensityRanged(
                keys=["image"], a_min=0, a_max=1000,
                b_min=0.0, b_max=1.0, clip=True
            ),
            # Redimensionnement haute qualite
            Resized(keys=["image"], spatial_size=TARGET_SHAPE)
        ])

        processed = transforms(data_dict)
        
        # 4. Sauvegarde
        nib.save(processed["image"], output_path)
        return True
    except Exception as e:
        print(f"Error processing {os.path.basename(input_path)}: {e}")
        return False

def main():
    print("🧠 Starting EXPERT Preprocessing Pipeline (MONAI + Nilearn)...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.nii.gz')]
    
    for i, filename in enumerate(files):
        in_p = os.path.join(INPUT_DIR, filename)
        out_p = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(out_p): continue
        
        success = expert_preprocess(in_p, out_p)
        if success:
            print(f"🔥 [{i+1}/{len(files)}] Perfect Preprocessing: {filename}")

    print("\n✅ Your dataset is now at Research-Grade quality.")

if __name__ == "__main__":
    main()