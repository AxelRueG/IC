import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron
from activation_funcs import escalon

def ejer1():
  # cargamos archivos y valores principlaes
  data = np.genfromtxt('./data/gtp1/OR_trn.csv',delimiter=',')
  Neurona = Perceptron(data,0.05)

  ## ENTRENAMIENTO -------------------------------------------------------------
  max_num_epoc = 50
  epoc = 0
  umbral_error = 0.15
  N = Neurona.data_trn.shape[0]

  while(epoc<max_num_epoc):
    e_trn = 0
    epoc+=1
    # entrenamos la neurona
    for i in range(N):
      y = Neurona.eval(i,escalon)
      Neurona.learning(i,y)
    
    for i in range(N):
      y = Neurona.eval(i,escalon)
      if (y != Neurona.yd[i]): e_trn+=1

    e_trn/=N
    print(f'el error en epoca [{epoc}] es del {e_trn}%')
    if e_trn<umbral_error: break
  
  ## COMPROBACION --------------------------------------------------------------
  # funcion para graficar recta
  recta = lambda x: (-Neurona.weights[0]*x+Neurona.weights[2])/Neurona.weights[1]

  # cargamos dataset de test
  data2 = np.genfromtxt('./data/gtp1/OR_tst.csv',delimiter=',')
  e_tst = 0
  for i in range(data2.shape[0]):
    y = Neurona.check(data2[i],escalon)
    if y != Neurona.yd[i]: e_tst+=1
  e_tst /= data2.shape[0]
  print(f'el error con dataset test es del {e_tst}%')
    
  # fig, ax = plt.subplots()
  # ax.plot(np.array([-1,1]),np.array([recta(-1),recta(1)]))
  # ax.plot(np.array([-1,1]),np.array([recta(-1),recta(1)]))
  # plt.show()
  

if __name__=='__main__':
  ejer1()
