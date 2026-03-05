# inference.py
import numpy as np

#Agregar ----> NOrmalizar las vainas 
def predict_single_image(image, model_layers):
    
    input= image

    reshaped_image=input.shape(-1)
    imagen_norm = reshaped_image/255

    for layer in model_layers:
        current_output = layer.forward(imagen_norm)
    
   
    return current_output


