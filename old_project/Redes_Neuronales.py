import numpy as np
import random 
class Dense_Layer:
    def __init__(self, n_input,n_neu):
        self.pesos = np.random.rand(n_input,n_neu)
        self.bias = np.zeros(n_neu)
        self.n_input = n_input
        self.n_neurons = n_neu
        print(self.pesos, self.bias)
    def forward(self, inputs):
        matrix_f = np.dot(inputs,self.pesos)+self.bias
        print(matrix_f.shape)
        return matrix_f
    def cargar_dimensiones(self, pesos, bias):
        if pesos.shape != self.pesos.shape or bias.shape != self.bias.shape:
            raise ValueError("Las dimensiones de los pesos no coincide ")
        self.pesos = pesos
        self.bias = bias

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output

class Activation_Softmax:
    def forward(self, inputs):
        safe_batch = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(safe_batch)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output


entradas = np.random.rand(6, 4)
print("Entradas:", entradas.shape)

Test_Layer = Dense_Layer(4, 3)
salidas = Test_Layer.forward(entradas)

capa_relu = Activation_ReLU()
capa_softmax = Activation_Softmax()

print("RELU ----> ", capa_relu.forward(salidas))
print("Softmax ----> ", capa_softmax.forward(salidas))


