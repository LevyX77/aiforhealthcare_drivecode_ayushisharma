import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# --- CONFIGURATION DES CHEMINS ---
HOME = os.path.expanduser("~")
INPUT_DIR = os.path.join(HOME, "Alzheimer_Project", "data", "raw")
OUTPUT_DIR = os.path.join(HOME, "Alzheimer_Project", "data", "processed")

# Taille cible pour que l'IA puisse traiter les images sans planter (96x96x96)
TARGET_SHAPE = (96, 96, 96)

def process_image(file_path):
    try:
        # 1. Charger l'image NIfTI
        img = nib.load(file_path)
        data = img.get_fdata()

        # 2. Normalisation (Min-Max) : transforme les valeurs en 0.0 - 1.0
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min != 0:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros(data.shape)

        # 3. Redimensionnement (Resizing) : pour que toutes les images fassent 96x96x96
        zoom_factors = [t / s for t, s in zip(TARGET_SHAPE, data.shape)]
        # order=1 signifie une interpolation bilinéaire (rapide et propre)
        data_resized = zoom(data, zoom_factors, order=1)

        return data_resized, img.affine
    except Exception as e:
        print(f"❌ Erreur sur {os.path.basename(file_path)} : {e}")
        return None, None

def main():
    print("🚀 Démarrage du pré-traitement...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.nii.gz')]
    total = len(files)
    print(f"📦 {total} images trouvées. Préparation en cours...")

    for i, filename in enumerate(files):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        # On saute l'image si elle est déjà traitée
        if os.path.exists(output_path):
            continue

        processed_data, affine = process_image(input_path)

        if processed_data is not None:
            # Sauvegarder la nouvelle image traitée
            new_img = nib.Nifti1Image(processed_data, affine)
            nib.save(new_img, output_path)
            
            # Afficher la progression toutes les 5 images
            if i % 5 == 0 or i == total - 1:
                print(f"✅ [{i+1}/{total}] Traité : {filename}")

    print(f"\n🎉 Terminé ! Tes images propres sont ici : {OUTPUT_DIR}")

if __name__ == "__main__":
    main()