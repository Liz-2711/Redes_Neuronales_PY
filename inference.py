# inference.py
import numpy as np
from Redes_Neuronales import DenseLayer, Activation_ReLU, Activation_Softmax



def predict_single_image(image, model_layers):
    
    input= image

    reshaped_image=input.shape(-1)
    imagen_norm = reshaped_image/255

    for layer in model_layers:
        current_output = layer.forward(imagen_norm)
    
   
    return current_output


class Forward_Red:
    def __int__(self, layers):
        self.layer=[]
        self.Capa_Red(layers)
    def Capa_Red(self, layers):
        for i, lay in enumerate(layers):
            W = lay["W"]
            b = lay["b"]

            n_input, n_neurons = W.shape
            
            # Crear capa densa
            dense = DenseLayer(n_input, n_neurons)
            dense.set_weights(W, b)
            self.layers.append(dense)
            
            # Agregar activación - Softmax para salida
            if i < len(layers) - 1:
                self.layers.append(Activation_ReLU())
            else:
                self.layers.append(Activation_Softmax())



def predicciones(self,imagen):
    # Preprocesar imagen
        output = predict_single_image(imagen)
        
        # Pasar por todas las capas
        for layer in self.layers:
            output = layer.forward(output)
        
        return output

def predict_class(self, imagen):
        
        probabilities = self.predict(imagen)
        return np.argmax(probabilities)