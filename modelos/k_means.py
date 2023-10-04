import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from os.path import abspath
from SOM import SOM

def k_means_online(k, data_set, coef_learn):
    u = data_set[ np.random.randint(0,data_set.shape[0],k), : ].copy()

    ajuste = 1
    criterio_corte = .0001
    epoca = 0
    while ajuste > criterio_corte:
        u_old = u.copy()
        for x in data_set:
            j = np.argmin( np.linalg.norm(u-x, axis=1) )
            u[j] += coef_learn * (x - u[j])
        ajuste = np.mean(np.linalg.norm(u_old-u, axis=1))
        epoca+=1
        # print(epoca)
    return u

def ejer_3(x):
    # dbs = []
    k=2
    for k in range(2,15):
        pesos = k_means_online(k, x, 0.2) 
        y_kmeans = [ np.argmin(np.linalg.norm(data-pesos, axis=1)) for data in x ]
        db = metrics.davies_bouldin_score(x, y_kmeans)
        # dbs.append(db)
        print(f'davies-bouldin: {db}')

    # plt.plot(dbs)
    # plt.show()

def ejer_2(x):
    # SOM
    som = SOM(4,(1,3))
    som.trn(x,1)

    # K-means
    pesos = k_means_online(3,x, 0.2) 

    y_kmeans = []
    y_som = []
    C = [[],[],[]]
    for i in range(len(x)):
        win = np.argmin(np.linalg.norm(x[i]-pesos, axis=1))
        # win = np.argmin(np.linalg.norm(x[i]-som.pesos, axis=2))
        y_kmeans.append(win)       # guardo la salida ganadora
        y_som.append( np.argmin(np.linalg.norm(x[i]-som.pesos, axis=2)) )
        C[win].append(i)    # genero el conjunto de cluster k-means   
    
    cm = metrics.confusion_matrix(y_som, y_kmeans)
    print(cm)


if __name__=='__main__':
    arch_name_trn = abspath('data/Guia2/irisbin_trn.csv')
    data = np.genfromtxt(arch_name_trn, delimiter=',')
    x = data[:, 0:4]

    ejer_3(x)