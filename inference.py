# inference.py
import numpy as np
from Redes_Neuronales import Dense_Layer, Activation_ReLU, Activation_Softmax


class Forward_Red:
    def __init__(self, layers_config):
      
        self.layers = []
        self._build_network(layers_config)

    def _build_network(self, layers_config):
        for i, layer_cfg in enumerate(layers_config):
            # Pesos y bias desde el JSON
            weights = np.array(layer_cfg["W"])
            biases = np.array(layer_cfg["b"])

            n_input, n_neurons = weights.shape

            # Crear capa densa con las dimensiones correctas
            dense = Dense_Layer(n_input, n_neurons)
            dense.cargar_dimensiones(weights, biases)
            self.layers.append(dense)

            # Capas de activación: ReLU en las ocultas, Softmax en la final
            if i < len(layers_config) - 1:
                self.layers.append(Activation_ReLU())
            else:
                self.layers.append(Activation_Softmax())

    def predict(self, image):
        # Busacar porque con flattened = image.reshape(-1) / 255.0 me da error ??
       
        flattened = image.reshape(1, -1).astype(np.float32) / 255.0

        current_output = flattened
        for layer in self.layers:
            current_output = layer.forward(current_output)

        return current_output

    def predict_class(self, image):
       
        probabilities = self.predict(image)
        # np.argmax sobre eje de columnas, tomamos el primer (y único) elemento del batch
        return int(np.argmax(probabilities, axis=1)[0])