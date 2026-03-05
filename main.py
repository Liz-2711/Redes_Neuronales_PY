# main.py
import numpy as np
from data_loader import load_dataset
from load_modelo import load_model_weights
from inference import Forward_Red

# --- CONFIGURACIÓN SIMPLE ---

DATA_PATH = "data/mnist_test.npz"
MODEL_PATH = "models/mnist_mlp_pretty.json"
# ---------------------------

def main():
    """
    Función principal para evaluar la precisión de la red neuronal.
    """
    print("=" * 50)
    print("  EVALUADOR DE RED NEURONAL MNIST")
    print("=" * 50)

    # 1. Cargar los datos de prueba y el modelo entrenado
    print("Paso 1: Cargando datos y modelo...")
    images, labels = load_dataset(DATA_PATH)
    model_config = load_model_weights(MODEL_PATH)
    
    # Si algo falló al cargar, detenemos el programa
    if images is None or model_config is None:
        print("Error: No se pudieron cargar los archivos. Revisa las rutas en el script.")
        return

    # 2. Construir la red neuronal con la configuración cargada
    print("Paso 2: Construyendo la red neuronal...")
    network = Forward_Red(model_config)

    # 3. Evaluar el modelo imagen por imagen
    print("Paso 3: Evaluando el modelo...")
    correct_predictions = 0
    total_images = len(images)
    
    # Recorremos todas las imágenes del dataset de prueba
    for i in range(total_images):
        image = images[i]
        true_label = labels[i]
        
        # La red predice la clase
        predicted_class = network.predict_class(image)
        
        # Si la predicción es correcta, sumamos al contador
        if predicted_class == true_label:
            correct_predictions += 1
            
        # Mostramos el progreso cada 1000 imágenes para no saturar la consola
        if (i + 1) % 1000 == 0:
            print(f"  Procesadas {i + 1}/{total_images} imágenes...")

    # 4. Calcular y mostrar los resultados finales
    accuracy = (correct_predictions / total_images) * 100
    
    print("\n" + "=" * 50)
    print("         RESULTADOS FINALES")
    print("=" * 50)
    print(f"Total de imágenes evaluadas: {total_images}")
    print(f"Aciertos: {correct_predictions}")
    print(f"Errores: {total_images - correct_predictions}")
    print(f"La precisión de la red es: {accuracy:.2f}%")
    print("=" * 50)

# Este bloque asegura que la función main() se ejecute cuando corres el script
if __name__ == "__main__":
    main()