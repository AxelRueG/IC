import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Perceptron import Perceptron
from activation_funcs import escalon

def ejer1():
  # cargamos archivos y valores principlaes
  data = np.genfromtxt('./data/gtp1/OR_trn.csv',delimiter=',')
  Neurona = Perceptron(data,0.05)

  # funcion para graficar recta
  recta = lambda x: (-Neurona.weights[0]*x+Neurona.weights[2])/Neurona.weights[1]
  # frames = np.zeros([N,2])

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
      # frames[i] = np.array([recta(-1.0),recta(1.0)])
      Neurona.learning(i,y)
    
    for i in range(N):
      y = Neurona.eval(i,escalon)
      if (y != Neurona.yd[i]): e_trn+=1
    
    e_trn/=N
    print(f'el error en epoca [{epoc}] es del {e_trn}%')
    if e_trn<umbral_error: break
  
  ## COMPROBACION --------------------------------------------------------------
  # cargamos dataset de test
  data2 = np.genfromtxt('./data/gtp1/OR_tst.csv',delimiter=',')
  e_tst = 0
  for i in range(data2.shape[0]):
    y = Neurona.check(data2[i],escalon)
    if y != Neurona.yd[i]: e_tst+=1
  e_tst /= data2.shape[0]
  print(f'el error con dataset test es del {e_tst}%')
    
  ## -- ANIMACION DE LA PRIMERA EPOCA ------------------------------------------
  # def f_animacion(x):
  #   ax.clear()
  #   ax.plot(np.array([-1.0,1.0]),x)
  #   ax.set_xlim(-1.0,1.0)
  #   ax.set_ylim(-1.0,1.0)
  #   plt.show()
  
  # fig, ax = plt.subplots()
  ## definimos la animacion
  # animator = FuncAnimation(fig,func=f_animacion,frames=frames, interval=5)
  ## grafica de la recta de los pesos
  # ax.plot(np.array([-1.0,1.0]),np.array([recta(-1.0),recta(1.0)]))
  # plt.show()
  

if __name__=='__main__':
  ejer1()
