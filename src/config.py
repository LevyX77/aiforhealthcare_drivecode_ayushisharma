import os

# Calcule le chemin racine du projet (C:\Users\...\Alzheimer_Project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Définition des dossiers
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.csv')

# Dossier des templates (pour le fichier MNI152)
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, 'MNI152_T1_1mm.nii.gz')

# Paramètres globaux
IMG_SIZE = (128, 128, 128)