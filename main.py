import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron
from activation_funcs import escalon

def ejer1():
  # cargamos archivos y valores principlaes
  data = np.genfromtxt('./data/gtp1/XOR_trn.csv',delimiter=',')
  Neurona = Perceptron(data,0.05)
  max_num_epoc = 50
  sencitivity_min = 0.8

  # epocas de entrenamiento 
  epoc = 0
  sencitivity = 0
  while(epoc<max_num_epoc and sencitivity<sencitivity_min):
    sencitivity = 0
    epoc+=1
    for i in range(Neurona.data_trn.shape[0]):
      y = Neurona.eval(i)
      Neurona.learning(i,y)
      if escalon(y) == data[i,-1]:
        sencitivity+=1
    sencitivity /= data.shape[0]
    print(f'epoca [{epoc}] sencivilidad del {sencitivity}%')
  
  # chequeamos que el clasificador funcione
  recta = lambda x: (-Neurona.weights[0]*x+Neurona.weights[2])/Neurona.weights[1]

  data2 = np.genfromtxt('./data/gtp1/XOR_tst.csv',delimiter=',')
  fig, ax = plt.subplots()
  t_pos = 0
  f_neg = 0
  t_neg = 0
  f_pos = 0
  for i in range(data2.shape[0]):
    y = Neurona.check(data2[i])
    if y < 0 and data2[i,-1] == -1:
      t_neg += 1
    elif y < 0 and data2[i,-1] != -1:
      f_neg += 1
    elif y > 0 and data2[i,-1] == 1:
      t_pos += 1
    else:
      f_pos += 1

    color = 'ob' if y==data2[i,-1] else 'or'
    ax.plot(data2[i,0],data2[i,1],color,markersize=2)
  ax.plot(np.array([-1,1]),np.array([recta(-1),recta(1)]))
  # print(np.array([-1,1]),np.array([recta(-1),recta(1)]))
  plt.show()
  print(f'datos [tp:{t_pos}] [fp:{f_pos}] [tn:{t_neg}] [fn:{f_neg}]')
  


if __name__=='__main__':
  ejer1()
