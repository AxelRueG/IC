import numpy as np
import matplotlib.pyplot as plt


class SOM:

    def __init__(self, n_input, map_dim=(10,10), epocs=1000, learning_rate=0.1) -> None:
        self.n_input = n_input              # Dimensión de los datos de entrada
        self.map_dim = map_dim              # Dimensión del mapa SOM (defoult 10x10)
        self.epocs = epocs                  # Número de épocas de entrenamiento
        self.learning_rate = learning_rate  # Tasa de aprendizaje
        self.pesos = np.random.rand(map_dim[0], map_dim[1], n_input)

    def encontrar_neurona_ganadora(self, x):
        distancias = np.linalg.norm(self.pesos - x, axis=2)      # distancia euclidea
        ganadora = np.argmin(distancias)
        return np.unravel_index(ganadora, distancias.shape) # nos devuelve el indice para la matriz

    def trn(self, data_set):
        # Entrenamiento del SOM
        for epoc in range(self.epocs):
            for x in data_set:
                # Encontrar la neurona ganadora
                ganadora = self.encontrar_neurona_ganadora(x)

                # Actualizar los pesos de la neurona ganadora y sus vecinos
                for i in range(self.map_dim[0]):
                    for j in range(self.map_dim[1]):
                        distancia = np.linalg.norm(np.array([i, j]) - np.array(ganadora))
                        influencia = np.exp(-distancia / 2.0)  # Función de influencia
                        self.pesos[i, j] += self.learning_rate * influencia * (x - self.pesos[i, j])