import numpy as np
import matplotlib.pyplot as plt
from modelos.MLP import MultiLayerPreceptron
from modelos.Perceptron import Perceptron
from utils.funciones_de_activacion import sgn, sigmoide
from utils.utils import funcionLinealPerceptron
from os.path import abspath

def ejer_1():
    arch_name_trn = abspath('data/Guia1/OR_trn.csv')
    arch_name_tst = abspath('data/Guia1/OR_tst.csv')
    # arch_name_trn = abspath('data/Guia1/XOR_trn.csv')
    # arch_name_tst = abspath('data/Guia1/XOR_tst.csv')
    # arch_name_trn = abspath('data/Guia1/OR_50_trn.csv')
    # arch_name_tst = abspath('data/Guia1/OR_50_tst.csv')
    # arch_name_trn = abspath('data/Guia1/OR_90_trn.csv')
    # arch_name_tst = abspath('data/Guia1/OR_90_tst.csv')

    num_max_epoc = 100
    completion_criterial = .3

    # --- Entrenamiento ------------------------------------------------------------------
    data = np.genfromtxt(arch_name_trn, delimiter=',')
    x = data[:, 0:2]
    d = data[:, -1]
    _, x_len = x.shape

    perceptron = Perceptron(x_len, sgn, 0.2)

    plt.plot(x[:,0],x[:,1],'*')
    plt.title("antes de entrenar")
    funcionLinealPerceptron(perceptron.getWeight())
    
    # training by times
    epoc = 0
    while (perceptron.score(x,d)>completion_criterial and epoc<num_max_epoc):
        perceptron.trn(x,d,'c_error')
        epoc+=1

    # --- test ---------------------------------------------------------------------------
    data = np.genfromtxt(arch_name_tst, delimiter=',')
    x = data[:, 0:2]
    d = data[:, -1]

    # grafica de resultados
    for i in range(len(x)):
        y = perceptron.eval(x[i,:])
        if (y != d[i]):
            plt.plot(x[i,0], x[i,1], '*k')
        elif y > 0:
            plt.plot(x[i,0], x[i,1], '*r')
        else:
            plt.plot(x[i,0], x[i,1], '*b')
    plt.title(f"despues de la epoca {epoc}")
    funcionLinealPerceptron(perceptron.getWeight())

    print('el error medio es:', perceptron.score(x,d))


# def ejer_3(arch_name, num_epoc, stop_creterial):
#     # tenemos que crear un MLP para poder distingir entre los datos
#     data = np.genfromtxt(arch_name, delimiter=',')
#     num_samples, num_in = data.shape
#     x = data[:,0:2]
#     # d = data[:,1::]
#     d = data[:,-1]

#     # modelo
#     model = MultiLayerPreceptron(num_in-1,[2,1],0.1)

#     # --- entrenamiento ----------------------------------------------------------------
#     epoc = 0
#     score = 100
#     while (score>stop_creterial and epoc<num_epoc):
#         model.trn(x,np.transpose([d]))
#         # model.trn(x,d)
#         score = model.score(x,d)
#         epoc+=1

#     print(f'el valor del error obtenido en la epoca {epoc} es {score}')

#     ## --- test ------------------------------------------------------------------------
#     data = np.genfromtxt(abspath("./data/gtp1/XOR_tst.csv"), delimiter=',')
#     x = data[:,0:2]
#     # d = data[:,1::]
#     d = data[:,-1]
#     # y = model.eval(x[0,:])
#     # print(y, d[0])
    
#     plt.figure()
#     for i in range(len(x)):
#         y = model.eval(x[i,:])
#         if y != d[i]:
#             plt.plot(x[i,0], x[i,1], '*k')
#         elif y > 0:
#             plt.plot(x[i,0], x[i,1], '*r')
#         else:
#             plt.plot(x[i,0], x[i,1], '*b')
    
#     plt.show()

