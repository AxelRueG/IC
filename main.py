from dbus import Interface
import numpy as np
import matplotlib.pyplot as plt

def sigmoide(z):
  x = np.exp(-z)
  return (1-x)/(1+x)

class Perceptron:
  def __init__(self, inputs, tasa_aperendizaje):
    # guardamos las entradas, la ultima es el bias
    self.data_trn = inputs
    self.data_trn[:,-1] = -1
    # creamos los pesos de la neurona, con valores aleatoreos
    self.gamma = tasa_aperendizaje
    self.weights = np.random.rand(inputs.shape[1])-0.5
    # guardamos las salidas
    self.outputs = inputs[:,-1]
    
  def eval(self, it):
    return sigmoide(np.dot(self.weights,self.data_trn[it]))
  
  def check(self, x):
    x = np.copy(x)
    x[-1] = -1
    return np.dot(self.weights,x)

  def learning(self, it, y):
    error = (self.outputs[it]-y)
    self.weights += self.gamma*error*self.data_trn[it]
    return error

def ejer1():
  data = np.genfromtxt('./data/gtp1/OR_trn.csv',delimiter=',')
  Neurona = Perceptron(data,0.2)

  max_num_epoc = 10
  error_minimo = 0.2

  epoc = 0
  mse = 1
  while(epoc<max_num_epoc and error_minimo<mse):
    # mse = 0
    epoc+=1
    for i in range(data.shape[0]):
      y = Neurona.eval(i)
      Neurona.learning(i,y)
    
    print(f'el ECM en la epoca {epoc} es del {mse}')
  
  # chequeamos que el clasificador funcione
  recta = lambda x: (-Neurona.weights[0]*x+Neurona.weights[2])/Neurona.weights[1]

  data2 = np.genfromtxt('./data/gtp1/OR_tst.csv',delimiter=',')
  fig, ax = plt.subplots()
  for i in range(data2.shape[0]):
    y = Neurona.check(data2[i])
    color = 'ob' if y>0 else 'or'
    # ax.plot(data2[i,0],data2[i,1],color,markersize=2)
  # ax.plot(np.array([-1,1]),np.array([recta(-1),recta(1)]))
  print(np.array([-1,1]),np.array([recta(-1),recta(1)]))
  # plt.show()
  


if __name__=='__main__':
  # ejer1()
  plt.plot(np.arange(-15,15,0.2),sigmoide(np.arange(-15,15,0.2)))
  plt.show()