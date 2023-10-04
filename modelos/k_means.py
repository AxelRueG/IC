import numpy as np
from os.path import abspath

def k_means_online(k, data_set, coef_learn):
    u = data_set[ np.random.randint(0,data_set.shape[0],k), : ].copy()

    ajuste = 1
    criterio_corte = 0.05
    epoca = 0
    while ajuste> criterio_corte:
        u_old = u.copy()
        for x in data_set:
            j = np.argmin( np.linalg.norm(u-x, axis=1) )
            u[j] += coef_learn * (x - u[j])
        ajuste = np.mean(np.linalg.norm(u_old-u, axis=1))
        epoca+=1
        print(epoca)
    return u


if __name__=='__main__':
    arch_name_trn = abspath('data/Guia1/OR_90_trn.csv')
    data = np.genfromtxt(arch_name_trn, delimiter=',')
    x = data[:, 0:2]

    print(k_means_online(4,x, 0.2))