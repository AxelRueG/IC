import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath

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
                self.pesos[i,j] += (coef_learn * ( x - self.pesos[index] ))

    def actualizar_vecindad(self, index, x, k, coef_learn):
        for i in range(-k,k+1):
            for j in range(-k,k+1):
                if abs(i)+abs(j) < k:
                    try: 
                        self.pesos[index[0]+i,index[1]+j] += (coef_learn * ( x - self.pesos[index] ))
                    except:
                        pass

    # --- Entrenamiento ------------------------------------------------------------------
    def trn(self, data_set, vecindad = 5, epocs = [1000,1000,500], coef_learn=[0.1, 0.01]):
        # Entrenamiento del SOM ajuste grueso
        for _ in range(epocs[0]):
            for x in data_set:
                # Encontrar la neurona ganadora
                ganadora = self.encontrar_neurona_ganadora(x)
                self.actualizar_vecindad(ganadora, x, vecindad, coef_learn[0])
        
        # reduccion lineal
        print('empieza a reducir linealmente el coef de aprendizaje')
        coefs = np.linspace(coef_learn[0],coef_learn[1],epocs[1])
        vecindades = np.linspace(vecindad, 0, epocs[1]).astype(int)
        for epoc in range(epocs[1]):
            for x in data_set:
                # Encontrar la neurona ganadora
                ganadora = self.encontrar_neurona_ganadora(x)
                self.actualizar_vecindad(ganadora, x, vecindades[epoc], coefs[epoc])
        
        # Entrenamiento del SOM ajuste fino
        print('empieza ajuste fino')
        for _ in range(epocs[2]):
            for x in data_set:
                # Encontrar la neurona ganadora
                ganadora = self.encontrar_neurona_ganadora(x)
                self.actualizar_vecindad(ganadora, x, 0, coef_learn[1])

    def plot_pesos(self):
        for i in range(self.pesos.shape[0]):
            for j in range(self.pesos.shape[1]):
                plt.plot(self.pesos[i,j,0], self.pesos[i,j,1], '*k')
                # Graficar conexiones
                if i+1 < self.pesos.shape[0]:
                    plt.plot(self.pesos[i:i+2,j,0], self.pesos[i:i+2,j,1], '-k')    
                if j+1 < self.pesos.shape[1]:
                    plt.plot(self.pesos[i,j:j+2,0], self.pesos[i,j:j+2,1], '-k')

        plt.show()


if __name__=='__main__':
    arch_name_trn = abspath('data/Guia1/OR_90_trn.csv')
    data = np.genfromtxt(arch_name_trn, delimiter=',')
    x = data[:, 0:2]

    som = SOM(2, (2,2))

    som.plot_pesos()
    som.trn(x, 1, [100,100,100])

    for val in x:
        plt.plot(val[0], val[1], '.r')
    som.plot_pesos()
