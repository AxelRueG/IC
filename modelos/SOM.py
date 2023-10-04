from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath
import time

np.random.seed = 10

class SOM:

    def __init__(self, n_input, map_dim=(10,10)):
        self.n_input = n_input              # Dimensión de los datos de entrada
        self.map_dim = map_dim              # Dimensión del mapa SOM (defoult 10x10)
        self.pesos = np.random.rand(map_dim[0], map_dim[1], n_input)-0.5

    # --- nos devuelve el indice para la matriz ------------------------------------------
    def encontrar_neurona_ganadora(self, x):
        distancias = np.linalg.norm(self.pesos - x, axis=2)      # distancia euclidea
        return np.unravel_index(np.argmin(distancias), distancias.shape)

    # --- Actualizaciones de los pesos ---------------------------------------------------
    def actualizar_vecindad_cuadrada(self, index, x, k, coef_learn):
        for i in range(max(0,index[0]-k), min(index[0]+k+1, self.pesos.shape[0])):
            for j in range(max(0,index[1]-k), min(index[1]+k+1, self.pesos.shape[1])):
                self.pesos[i,j] += (coef_learn * ( x - self.pesos[i,j] ))

    def actualizar_vecindad_rombo(self, index, x, k, coef_learn):
        for i in range(-k,k+1):
            for j in range(-k,k+1):
                if abs(i)+abs(j) <= k:
                    try: 
                        self.pesos[index[0]+i,index[1]+j] += (coef_learn * ( x - self.pesos[index[0]+i,index[1]+j] ))
                    except:
                        pass

    # --- Entrenamiento ------------------------------------------------------------------
    def trn(self, data_set, vecindad = 5, epocs = 1000, coef_learn=[0.1, 0.01]):        

        ajuste_topologico = int(epocs * 0.2)
        ajuste_lineal = int(epocs * 0.8)

        # generamos lista de coeficientes por cada epoca
        coefs =  coef_learn[1] * np.ones(epocs)     # coeficientes de ajuste fino
        coefs[ajuste_topologico:ajuste_lineal] = np.linspace(coef_learn[0],coef_learn[1],ajuste_lineal-ajuste_topologico)
        coefs[:ajuste_topologico] = coef_learn[0]   # coeficientes de ajuste topologico
        
        vecindades = np.zeros(epocs).astype(int)    # vecindad en ajuste fino
        vecindades[ajuste_topologico:ajuste_lineal] = np.linspace(vecindad,0,ajuste_lineal-ajuste_topologico).astype(int)
        vecindades[:ajuste_topologico] = vecindad   # vecindad en ajuste topologico

        peso_history = [np.copy(self.pesos)]

        for epoc in range(epocs):

            print(f'epoca {epoc}: [ vecindad: {vecindades[epoc]}, velocidad de aprendizaje: {coefs[epoc]} ]')
            for x in data_set:
                # Encontrar la neurona ganadora
                ganadora = self.encontrar_neurona_ganadora(x)
                # actulizar ganadora y vecidad
                # self.actualizar_vecindad_rombo(ganadora, x, vecindades[epoc], coefs[epoc])
                self.actualizar_vecindad_cuadrada(ganadora, x, vecindades[epoc], coefs[epoc])
                
            peso_history.append(np.copy(self.pesos))
        
        return peso_history

    def plot_pesos(self, pesos, color = 'k'):
        for i in range(pesos.shape[0]):
            for j in range(pesos.shape[1]):
                plt.plot(pesos[i,j,0], pesos[i,j,1], f'*{color}')
                # Graficar conexiones
                if i+1 < pesos.shape[0]:
                    plt.plot(pesos[i:i+2,j,0], pesos[i:i+2,j,1], f'-{color}')    
                if j+1 < pesos.shape[1]:
                    plt.plot(pesos[i,j:j+2,0], pesos[i,j:j+2,1], f'-{color}')

    def animaccion(self, frame, pesos, x):
        plt.clf()
        plt.title(f'Frame {frame}')
        plt.plot(x[:,0], x[:,1], '.r')
        self.plot_pesos(pesos[frame])


if __name__=='__main__':
    arch_name_trn = abspath('data/Guia4/circulo.csv')
    # arch_name_trn = abspath('data/Guia4/te.csv')
    data = np.genfromtxt(arch_name_trn, delimiter=',')
    x = data[:, 0:2]

    som = SOM(2, (10,10))

    plt.figure()
    plt.plot(x[:,0], x[:,1], '.r')
    som.plot_pesos(np.copy(som.pesos))

    time_ini = time.time()
    pesos = som.trn(data_set=x, vecindad=5, epocs = 500, coef_learn=[0.9, 0.1])
    print(f'tiempo de ejecucion: {time.time() - time_ini} seg')

    plt.figure()
    plt.plot(x[:,0], x[:,1], '.r')
    som.plot_pesos(np.copy(som.pesos))
    plt.show()

    # fig = plt.figure()
    # anim = FuncAnimation(fig, som.animaccion, frames=len(pesos), interval=100, fargs=(pesos, x))
    # plt.show()
