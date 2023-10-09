import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import metrics 
from os.path import abspath
from modelos.SOM import SOM
from modelos.k_means import k_means_online 


def ejer_1():
    # arch_name_trn = abspath('data/Guia4/circulo.csv')
    arch_name_trn = abspath('data/Guia4/te.csv')
    data = np.genfromtxt(arch_name_trn, delimiter=',')
    x = data[:, 0:2]

    som = SOM(2, (20,20))

    plt.figure()
    plt.plot(x[:,0], x[:,1], '.r')
    som.plot_pesos(np.copy(som.pesos))

    time_ini = time.time()
    pesos = som.trn(data_set=x, vecindad=2, epocs = 500, coef_learn=[0.9, 0.1])
    print(f'tiempo de ejecucion: {time.time() - time_ini} seg')

    plt.figure()
    plt.plot(x[:,0], x[:,1], '.r')
    som.plot_pesos(np.copy(som.pesos))
    plt.show()

    # fig = plt.figure()
    # anim = FuncAnimation(fig, som.animaccion, frames=len(pesos), interval=100, fargs=(pesos, x))
    # plt.show()


def ejer_3(x):
    dbs = []
    k=2
    for k in range(2,11):
        pesos = k_means_online(k, x, 0.2) 
        y_kmeans = [ np.argmin(np.linalg.norm(data-pesos, axis=1)) for data in x ]
        db = metrics.davies_bouldin_score(x, y_kmeans)
        dbs.append(db)
        print(f'k: {k}, davies-bouldin: {db}')

    plt.plot(np.arange(2,11),dbs)
    plt.show()

def ejer_2(x):
    # SOM
    som = SOM(4,(1,3))
    som.trn(x,1)

    # K-means
    pesos = k_means_online(3,x, 0.2) 

    y_kmeans = []
    y_som = []
    for i in range(len(x)):
        win = np.argmin(np.linalg.norm(x[i]-pesos, axis=1))
        # win = np.argmin(np.linalg.norm(x[i]-som.pesos, axis=2))
        y_kmeans.append(win)       # guardo la salida ganadora
        y_som.append( np.argmin(np.linalg.norm(x[i]-som.pesos, axis=2)) )
    
    cm = metrics.confusion_matrix(y_som, y_kmeans)
    print(cm)


if __name__=='__main__':
    arch_name_trn = abspath('data/Guia2/irisbin_trn.csv')
    data = np.genfromtxt(arch_name_trn, delimiter=',')
    x = data[:, 0:4]

    # ejer_1()
    # ejer_2(x)
    ejer_3(x)