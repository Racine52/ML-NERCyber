import os

# Chemins des dossiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Chemin pour sauvegarder le modèle
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'NER_model_1.h5')