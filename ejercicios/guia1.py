import numpy as np
import matplotlib.pyplot as plt
from modelos.MLP import MultiLayerPreceptron
from modelos.Perceptron import Perceptron
from utils.funciones_de_activacion import sgn, sigmoide

def ejer_1(arch_name, num_max_epoc, completion_criterial):
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


def ejer_3(arch_name, num_epoc, stop_creterial):
    # tenemos que crear un MLP para poder distingir entre los datos dados por "concentlite.csv"
    data = np.genfromtxt(arch_name, delimiter=',')
    num_samples, num_in = data.shape
    x = data[:,0:2]
    d = data[:,-1]

    # mix the data to separate training and test data
    rgn = np.random.default_rng()
    it_random = rgn.permutation(np.arange(num_samples))
    cant_trn = int(num_samples*0.7)
    it_trn = it_random[0:cant_trn]
    it_tst = it_random[cant_trn:]

    # modelo
    model = MultiLayerPreceptron(num_in-1,[4,1],0.1,sigmoide)

    # entrenamiento
    errs = []
    for epoc in range(num_epoc):
        err = 0
        for it in it_trn:
            err += model.traing(x[it,:], d[it])
        
        err /= len(it_trn)
        errs.append(err)
        # print('error medio absoluto por epoca: ', err)
        if err <= stop_creterial:
            print('finished in: ',epoc,' epocs')
            break

    plt.figure()
    plt.plot(np.array(errs))
    plt.show()

    # test
    plt.figure()
    for it in it_tst:
        y = model.eval(x[it,:])
        if (y<0 and d[it]>0) or (y>0 and d[it]<0):
            plt.plot(x[it,0], x[it,1], '*k')
        elif y > 0:
            plt.plot(x[it,0], x[it,1], '*r')
        else:
            plt.plot(x[it,0], x[it,1], '*b')
    
    plt.show()

