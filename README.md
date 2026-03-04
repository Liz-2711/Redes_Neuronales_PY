proyecto_inferencia/

-> main.py            # Punto de entrada, recibe parámetros de línea de comando
-> data_loader.py     # Carga datasets (.npz) y devuelve images + labels
-> model_loader.py    # Carga pesos de la red desde JSON
-> layers.py          # Contiene clases: DenseLayer, Activation_ReLU, Activation_Softmax
-> inference.py       # Ejecuta forward pass imagen por imagen
-> evaluator.py       # Compara predicciones vs labels, cuenta aciertos/errores
-> visualizer.py      # Muestra imágenes con matplotlib, muestra errores

-> requirements.txt   # Lista de dependencias (numpy, matplotlib, etc.)


-> data/                      # Carpeta para datasets
   -> mnist_test.npz         # Dataset de prueba (imágenes + labels)
   ->mnist_train.npz        # (Opcional) Dataset de entrenamiento

 models/                    # Carpeta para modelos entrenados
    -> mnist_mlp_pretty.json  # Pesos y bias de la red ya entrenada