import os
import pandas as pd
import dicom2nifti
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRangePercentilesd, Resized, SaveImaged
)
import glob
import shutil

# --- CONFIGURATION ---
HOME = os.path.expanduser("~")
CSV_PATH = os.path.join(HOME, "Alzheimer_Project", "data", "metadata.csv")
RAW_DICOM_DIR = os.path.join(HOME, "TRAIN_1", "MRI")
OUTPUT_DIR = os.path.join(HOME, "Alzheimer_Project", "data", "final")
TEMP_DIR = os.path.join(HOME, "Alzheimer_Project", "data", "temp_conversion")

TARGET_SHAPE = (96, 96, 96)

def get_labels_from_csv():
    print("📊 Lecture du CSV...")
    df = pd.read_csv(CSV_PATH)
    # On ne garde que CN (Normal) et AD (Alzheimer)
    df = df[df['Group'].isin(['CN', 'AD'])]
    # On crée un dictionnaire: I12345 -> AD
    labels = dict(zip(df['ImageID'], df['Group']))
    print(f"✅ {len(labels)} patients CN/AD trouvés dans le CSV.")
    return labels

def preprocess_monai(temp_file, final_output_path):
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image"], spatial_size=TARGET_SHAPE),
        SaveImaged(keys=["image"], output_dir=os.path.dirname(final_output_path), 
                   output_postfix="", resample=False, print_log=False)
    ])
    
    # MONAI sauvegarde avec un nom complexe, on va gérer ça
    data = {"image": temp_file}
    transforms(data)
    
    # Renommer le fichier créé par MONAI pour qu'il ait le bon nom exact
    created_files = glob.glob(os.path.join(os.path.dirname(final_output_path), "*_trans.nii.gz"))
    for f in created_files:
        # On le déplace vers le nom final (Ixxxxx.nii.gz)
        shutil.move(f, final_output_path)

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)

    labels = get_labels_from_csv()
    
    print("🚀 Démarrage du Pipeline : Conversion + Preprocessing...")
    
    count = 0
    # On parcourt récursivement pour trouver les dossiers Ixxxxx
    for root, dirs, files in os.walk(RAW_DICOM_DIR):
        folder_name = os.path.basename(root)
        
        # Si le dossier est un ImageID valide (ex: I41168) et qu'il est dans notre liste CN/AD
        if folder_name in labels:
            label = labels[folder_name]
            final_path = os.path.join(OUTPUT_DIR, f"{folder_name}.nii.gz")
            
            if os.path.exists(final_path):
                continue
                
            print(f"🔄 Traitement : {folder_name} ({label})")
            
            try:
                # 1. Conversion DICOM -> NIfTI (Temporaire)
                temp_nifti_dir = os.path.join(TEMP_DIR, folder_name)
                if not os.path.exists(temp_nifti_dir): os.makedirs(temp_nifti_dir)
                dicom2nifti.convert_directory(root, temp_nifti_dir, compression=True, reorient=True)
                
                # Trouver le fichier généré
                nii_files = glob.glob(os.path.join(temp_nifti_dir, "*.nii.gz"))
                if nii_files:
                    # 2. Preprocessing MONAI
                    preprocess_monai(nii_files[0], final_path)
                    count += 1
                
                # Nettoyage temporaire
                shutil.rmtree(temp_nifti_dir)
                
            except Exception as e:
                print(f"⚠️ Erreur sur {folder_name}: {e}")

    # Nettoyage final
    shutil.rmtree(TEMP_DIR)
    print(f"\n🎉 Terminé ! {count} images prêtes dans {OUTPUT_DIR}")

if __name__ == "__main__":
    main()