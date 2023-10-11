import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath
from sklearn.model_selection import KFold, cross_val_score

from sklearn.tree import DecisionTreeClassifier

from modelos.genetico import genetico

# datos
arch_name_trn = abspath('data/Guia6/leukemia_train.csv')
data_set = np.genfromtxt(arch_name_trn, delimiter=',')
data = data_set[:,:-1].copy()
yd = data_set[:,-1].copy()

# --- decodificacion -----------------------------------------------------------------
def decodificar_leukemia(poblacion, gen_target_max = 0, target_max = 0, target_min = 0):
    lista_de_data_set = []
    # poblacion es una matriz
    for individuo in poblacion:
        lista_de_data_set.append(data[:,individuo!=0])
    return lista_de_data_set

# data_profe: [1,2,3,45,6,8] dy = 1
# geno:       [0,0,1,1, 0,1]
#             [3,45,8]


def fitness(lista_de_data_set):
    fitness_value = []

    for X in lista_de_data_set:
        model = DecisionTreeClassifier(random_state=0)
        kfold = KFold(n_splits=5, shuffle=True)
        accuracies = cross_val_score(model, X, yd, cv=kfold)
        fitness_value.append(np.mean(accuracies))

    return fitness_value

# --- Entrenamiento ------------------------------------------------------------------
F = lambda x: x

genetico(
    F,
    fitness,
    decode=decodificar_leukemia,
    gen_bits=data.shape[1],
    tamanio_poblacion=10,
    num_generaciones=200,
    porcentaje_hijos=0.80,
    probabilidad_cruza=0.8,
    probabilidad_mutacion=0.40,
    min_bits_cruza=1)

