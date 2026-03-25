# data_loader.py
import numpy as np

def load_dataset(file_path):
  
    try:
        data = np.load(file_path)
        images = data["images"]
        labels = data["labels"]
        print(f"Dataset cargado desde {file_path}.")
        return images, labels
    except FileNotFoundError:
        print(f"Error: El archivo de datos no se encontró en {file_path}")
        return None, None
    except KeyError:
        print(f"Error: El archivo {file_path} no tiene las llaves 'images' y 'labels'.")
        return None, None