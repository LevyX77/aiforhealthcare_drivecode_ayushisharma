import os
import dicom2nifti

# --- PATHS ---
INPUT_DIR = "/home/vyakti3/TRAIN_1/MRI"
OUTPUT_DIR = "/home/vyakti3/Alzheimer_Project/data/raw"

def convert_all():
    print("Starting conversion...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    count = 0
    for root, dirs, files in os.walk(INPUT_DIR):
        if any(f.endswith('.dcm') for f in files):
            try:
                folder_name = os.path.basename(root)
                print(f"Processing: {folder_name}")
                
                dicom2nifti.convert_directory(root, OUTPUT_DIR, compression=True, reorient=True)
                count += 1
            except Exception as e:
                print(f"Error: {e}")

    print(f"Finished! {count} folders converted.")

if __name__ == "__main__":
    convert_all()