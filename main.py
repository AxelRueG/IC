import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
from MLP import MultiLayerPerceptron
from Perceptron import Perceptron
from activation_funcs import escalon, signo
import time

def entrenamiento(data_trn, data_tst):
  Neurona = Perceptron(data_trn,0.05)
  N = data_trn.shape[0]

  # funcion para graficar recta
  recta = lambda x: (-Neurona.weights[0]*x+Neurona.weights[2])/Neurona.weights[1]
  frames = np.zeros([N,2])

  ## ENTRENAMIENTO -------------------------------------------------------------
  max_num_epoc = 50
  epoc = 0
  umbral_error = 0.15

  while(epoc<max_num_epoc):
    e_trn = 0
    epoc+=1
    # entrenamos la neurona
    for i in range(N):
      y = Neurona.eval(i,escalon)
      frames[i] = np.array([recta(-1.0),recta(1.0)])
      Neurona.learning(i,y)
    
    for i in range(N):
      y = Neurona.eval(i,escalon)
      if (y != Neurona.yd[i]): e_trn+=1
    e_trn/=N
    # print(f'el error en epoca [{epoc}] es del {e_trn}%')
    if e_trn<umbral_error: break
  
  ## COMPROBACION --------------------------------------------------------------
  e_tst = 0
  for i in range(data_tst.shape[0]):
    y = Neurona.check(data_tst[i],escalon)
    if y != data_tst[i,-1]: e_tst+=1
  e_tst /= data_tst.shape[0]
  # print(f'el error con dataset test es del {e_tst}%')

  return epoc,np.array([e_trn,e_tst])

  ## -- ANIMACION DE LA PRIMERA EPOCA ------------------------------------------
  # def f_animacion(x):
  #   ax.clear()
  #   ax.plot(np.array([-1.0,1.0]),x)
  #   ax.set_xlim(-1.0,1.0)
  #   ax.set_ylim(-1.0,1.0)
  #   plt.show()
  
  # fig, ax = plt.subplots()
  # # definimos la animacion
  # animator = FuncAnimation(fig,func=f_animacion,frames=frames, interval=5)
  # # grafica de la recta de los pesos
  # ax.plot(np.array([-1.0,1.0]),np.array([recta(-1.0),recta(1.0)]))
  # plt.show()

def ejer1():
  # cargamos archivos y valores principlaes
  data_trn = np.genfromtxt('./data/gtp1/OR_trn.csv',delimiter=',')
  data_tst = np.genfromtxt('./data/gtp1/OR_tst.csv',delimiter=',')
  entrenamiento(data_trn, data_tst)
  

def ejer2(num_part, porcentage):
  data = np.genfromtxt('./data/gtp1/spheres2d10.csv',delimiter=',')
  N = data.shape[0]
  blend_index = np.zeros([N,num_part])
  n = int(N*porcentage)    

  for i in range(num_part):
    index = np.random.permutation(np.arange(N)) # vector de indices permutados
    blend_index[:,i] = index.copy()
    epoc, e = entrenamiento(data[index[0:n]],data[index[n::]])
    print(f'particion {i} trn {epoc}:\t{e[0]}, porcentaje error_tst \t{e[1]}%')

def ejer3(file):
  data = np.genfromtxt(file, delimiter=',')
  index = np.random.permutation(np.arange(len(data)))
  to = int(len(data)*0.9)
  data_trn = data[index[0:to]].copy()
  data_tst = data[index[to::]].copy()
  # data_trn = data
  MLP = MultiLayerPerceptron([3,1],data_trn,mu=0.15)

  # training
  start = time.time()
  le = []
  for i in range(500):
    e = MLP.training_epoc()
    le.append(e)
    if e<0.2: break
  end = time.time()
  print(f'tiempo {end-start}')

  print(f'epocas recorridas {len(le)}')
  # test
  fig, ax = plt.subplots(2)
  ax[0].plot(np.array(le))
  for i in range(data_tst.shape[0]):
    y = MLP.eval(data_tst[i,0:-1])[-1]
  #   print(f'y: {y} d: {data_tst[i,-1]}')
    if y>0:
      if data_tst[i,-1] == 1:
        ax[1].plot(data_tst[i,0],data_tst[i,1],'*k')
      else:
        ax[1].plot(data_tst[i,0],data_tst[i,1],'*r')
    else:
      if data_tst[i,-1] == -1:
        ax[1].plot(data_tst[i,0],data_tst[i,1],'.b')
      else:
        ax[1].plot(data_tst[i,0],data_tst[i,1],'.r')
  plt.show()
        


if __name__=='__main__':
  # ejer2(10,0.8)
  ejer3('./data/gtp1/concentlite.csv')
