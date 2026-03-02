import json


def cargar_modelo_desde_json(filepath):
   
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    return config

modelo = cargar_modelo_desde_json("mnist_mlp_pretty.json")

print(modelo)