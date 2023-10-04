from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath
# import time

class SOM:

    def __init__(self, n_input, map_dim=(10,10)):
        self.n_input = n_input              # Dimensión de los datos de entrada
        self.map_dim = map_dim              # Dimensión del mapa SOM (defoult 10x10)
        self.pesos = np.random.rand(map_dim[0], map_dim[1], n_input)-0.5

        # self.etapas_titulo = ['ajuste topologico',]

    # --- nos devuelve el indice para la matriz ------------------------------------------
    def encontrar_neurona_ganadora(self, x):
        distancias = np.linalg.norm(self.pesos - x, axis=2)      # distancia euclidea
        return np.unravel_index(np.argmin(distancias), distancias.shape)

    # --- Actualizaciones de los pesos ---------------------------------------------------
    def actualizar_vecindad_cuadrada(self, index, x, k, coef_learn):
        for i in range(max(0,index[0]-k), min(index[0]+k+1, self.pesos.shape[0])):
            for j in range(max(0,index[1]-k), min(index[1]+k+1, self.pesos.shape[1])):
                self.pesos[i,j] += (coef_learn * ( x - self.pesos[index] ))

    def actualizar_vecindad(self, index, x, k, coef_learn):
        for i in range(-k,k+1):
            for j in range(-k,k+1):
                if abs(i)+abs(j) <= k:
                    try: 
                        self.pesos[index[0]+i,index[1]+j] += (coef_learn * ( x - self.pesos[index] ))
                    except:
                        pass

    # --- Entrenamiento ------------------------------------------------------------------
    def trn(self, data_set, vecindad = 5, epocs = [1000,1000,500], coef_learn=[0.1, 0.01]):        
        coef_learn.insert(1,0)
        vecindad = [vecindad, 0, 0]

        # peso_history = [np.copy(self.pesos)]

        for etapa in range(3):
            for epoc in range(epocs[etapa]):
                if etapa == 1:
                    coefs = np.linspace(coef_learn[0],coef_learn[1],epocs[1])
                    vecindades = np.linspace(vecindad[etapa], 0, epocs[1]).astype(int)
                else:
                    coefs = coef_learn[etapa] * np.ones(epocs[etapa])
                    vecindades = vecindad[etapa] * np.ones(epocs[etapa]).astype(int)

                print(epoc, vecindades[epoc], coefs[epoc])
                for x in data_set:
                    # Encontrar la neurona ganadora
                    ganadora = self.encontrar_neurona_ganadora(x)
                    # actulizar ganadora y vecidad
                    # self.actualizar_vecindad(ganadora, x, vecindades[epoc], coefs[epoc])
                    self.actualizar_vecindad_cuadrada(ganadora, x, vecindades[epoc], coefs[epoc])
                    
        #     peso_history.append(np.copy(self.pesos))

        
        # return peso_history

    def plot_pesos(self, pesos):
        for i in range(pesos.shape[0]):
            for j in range(pesos.shape[1]):
                plt.plot(pesos[i,j,0], pesos[i,j,1], '*k')
                # Graficar conexiones
                if i+1 < pesos.shape[0]:
                    plt.plot(pesos[i:i+2,j,0], pesos[i:i+2,j,1], '-k')    
                if j+1 < pesos.shape[1]:
                    plt.plot(pesos[i,j:j+2,0], pesos[i,j:j+2,1], '-k')

    def animaccion(self, frame, pesos, x):
        plt.clf()
        plt.title(f'Frame {frame}')
        plt.plot(x[:,0], x[:,1], '.r')
        self.plot_pesos(pesos[frame])





if __name__=='__main__':
    # arch_name_trn = abspath('data/Guia4/circulo.csv')
    arch_name_trn = abspath('data/Guia4/te.csv')
    data = np.genfromtxt(arch_name_trn, delimiter=',')
    x = data[:, 0:2]

    som = SOM(2)

    # som.plot_pesos(som.pesos)
    # plt.show()
    
    pesos = som.trn(x, 10, epocs = [100,100,100], coef_learn=[0.9, 0.1])
    # print(len(pesos))

    fig = plt.figure()
    anim = FuncAnimation(fig, som.animaccion, frames=len(pesos), interval=10, fargs=(pesos, x))
    plt.show()
