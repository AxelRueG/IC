import numpy as np
import matplotlib.pyplot as plt
from modelos.Perceptron import Perceptron
from utils.funciones_de_activacion import sgn

def ejer1(arch_name, num_max_epoc, completion_criterial):
    # traing data
    data = np.genfromtxt(arch_name, delimiter=',')
    x = data[:, 0:2]
    d = data[:, -1]
    num_ex, x_len = x.shape

    perceptron = Perceptron(x_len, sgn, 0.2)

    # training by times
    for epoc in range(0, num_max_epoc):
        err = 0
        for i in range(0, num_ex):
            err += perceptron.trn(x[i,:], d[i])

        if err/num_ex < completion_criterial: 
            print('han pasado:', epoc)
            break

    # --- test data --------------------------------------------------------------------------------
    data = np.genfromtxt(arch_name, delimiter=',')
    x = data[:, 0:2]
    d = data[:, -1]
    num_ex, _ = x.shape
    err = 0

    # check effectiveness
    for i in range(0, num_ex):
        y = perceptron.eval(x[i,:])
        if (d[i] != y): err+=1

        # generate graphs
        if y == 1:
            plt.plot(x[i,0], x[i,1], '*b')
        else:
            plt.plot(x[i,0], x[i,1], '*r')

    print('el error medio es:', err/num_ex)
    plt.show()