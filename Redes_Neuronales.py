import numpy as np
import random 
class Dense_Layer:
    def __init__(self, n_input,n_neu):
        self.pesos = np.random.rand(n_input,n_neu)
        self.bias = np.zeros(n_neu)
        print(self.pesos, self.bias)
    def forward(self, inputs):
        matrix_f = np.dot(inputs,self.pesos)+self.bias
        print(matrix_f.shape)
        return matrix_f

def Relu (x):
    return np.maximum(0,x)

def Softmax(x):
    safe_batch = x - np.max(x)
    exp_x = np.exp(safe_batch)
    return exp_x / np.sum(exp_x)


    

entradas = np.random.rand(6,4)
print("ENtradas" , entradas)
Test_Layer= Dense_Layer(4,3)
salidas = Test_Layer.forward(entradas)
print("RELU ----> ", Relu(salidas))
print("Softmax ----> ", Softmax(salidas))







