import numpy as np
from activation_funcs import sigmoide

class Perceptron:
  def __init__(self, inputs, tasa_aperendizaje):
    # guardamos las entradas, la ultima es el bias
    self.data_trn = np.copy(inputs)
    self.data_trn[:,-1] = -1
    # creamos los pesos de la neurona, con valores aleatoreos
    self.gamma = tasa_aperendizaje
    self.weights = np.random.rand(inputs.shape[1])-0.5
    # guardamos las salidas
    self.yd = np.copy(inputs[:,-1])
    
  def eval(self,it,func=sigmoide):
    return func(np.dot(self.weights,self.data_trn[it]))
  
  def check(self,x,func=sigmoide):
    x = np.copy(x)
    x[-1] = -1
    return func(np.dot(self.weights,x))

  def learning(self, it, y):
    error = (self.yd[it]-y)
    self.weights += (self.gamma*error*self.data_trn[it])
    return error