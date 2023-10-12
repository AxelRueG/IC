import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from modelos.genetico import genetico

# datos
arch_name_trn = abspath('data/Guia6/leukemia_train.csv')
arch_name_tst = abspath('data/Guia6/leukemia_test.csv')
data_set = np.genfromtxt(arch_name_trn, delimiter=',')
data = data_set[:,:-1].copy()
yd = data_set[:,-1].copy()
data_set_test = np.genfromtxt(arch_name_tst, delimiter=',')
data_test = data_set_test[:,:-1].copy()
yd_test = data_set_test[:,-1].copy()

# --- decodificacion -----------------------------------------------------------------
def decodificar_leukemia(poblacion, gen_target_max = 0, target_max = 0, target_min = 0):
    return poblacion

def fitness(poblacion):
    fitness_value = []

    for individuo in poblacion:
        model = DecisionTreeClassifier(random_state=0)

        model.fit(data[:, individuo!=0].copy(), yd)
        y = model.predict(data_test[:, individuo!=0])
        fitness_value.append(accuracy_score(yd_test,y))

    return fitness_value

# --- Entrenamiento ------------------------------------------------------------------
def grafica(F, fenotipo, generacion, mejores_apt, target_min, target_max):
    plt.figure(1)
    plt.clf()
    plt.title(f"Iteración nro {generacion}")
    plt.xlabel("Iteración")
    plt.ylabel("Mejor aptitud")
    plt.plot(range(1, generacion + 1), mejores_apt)
    plt.axis([0, generacion, max(mejores_apt)-5, max(mejores_apt)+5])

    plt.pause(0.0000001)


F = lambda x: x

genetico(
    F,
    fitness,
    decode=decodificar_leukemia,
    gen_bits=data.shape[1],
    tamanio_poblacion=50,
    num_generaciones=500,
    porcentaje_hijos=0.80,
    probabilidad_cruza=0.8,
    probabilidad_mutacion=0.40,
    # grafica=grafica,
    min_bits_cruza=1)

