import numpy as np

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

