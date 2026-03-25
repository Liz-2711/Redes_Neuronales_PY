# model_loader.py
import json
import numpy as np

def load_model_weights(file_path):
   
    try:
        with open(file_path, 'r') as f:
            model_config = json.load(f)
        print(f"Configuración del modelo cargada desde {file_path}.")
        return model_config['layers']
    except FileNotFoundError:
        print(f"Error: El archivo del modelo no se encontró en {file_path}")
        return None