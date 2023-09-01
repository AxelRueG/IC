import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath
from modelos.MLP import MultiLayerPreceptron
from utils.funciones_de_activacion import sgn
from utils.utils import funcionLinealPerceptron

def guia3_ejer1():
    arch_name_trn = abspath('data/Guia1/XOR_trn.csv')
    arch_name_tst = abspath('data/Guia1/XOR_tst.csv')

    num_max_epox = 100
    completation_criterial = 0.1

    # --- Entrenamiento ---------------------------------------------------------------------
    data = np.genfromtxt(arch_name_trn, delimiter= ',')
    x = data[:,0:2]
    d = data[:,-1]
    _, num_inputs = x.shape

    mlp = MultiLayerPreceptron(num_inputs, [2,1], 0.1)

    epoc = 0
    while (mlp.score(x,d)>completation_criterial and epoc<num_max_epox):
        mlp.trn(x,d)
        epoc += 1

    # --- test ------------------------------------------------------------------------------
    data = np.genfromtxt(arch_name_tst, delimiter=',')
    x = data[:, 0:2]
    d = data[:, -1]


    ws = mlp.getWeigth()

    # grafica de resultados
    for i in range(len(x)):
        y = mlp.eval(x[i,:])
        if (sgn(y) != d[i]):
            plt.plot(x[i,0], x[i,1], '*k')
        elif y > 0:
            plt.plot(x[i,0], x[i,1], '*r')
        else:
            plt.plot(x[i,0], x[i,1], '*b')
    plt.title(f"despues de la epoca {epoc}")

    # -- Descomentar para observar las dos lineas que dividen el espacio en una arq: [2,1] --
    # funcionLinealPerceptron([ws[0][0]])
    # funcionLinealPerceptron([ws[0][1]])
    plt.show()

    print('el error cuadratico medio es:', mlp.score(x,d))