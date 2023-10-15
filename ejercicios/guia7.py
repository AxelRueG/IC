import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath
from modelos.enjambre import colonia_de_hormigas, enjambre_gEP

# --------------------------------------------------------------------------------------------------
#                                       EJERCICIO 1 
# --------------------------------------------------------------------------------------------------
# def graficar(poblacion, epoc, x_min, x_max):
#     plt.figure(1)
#     plt.clf()
#     x = np.linspace(x_min,x_max,1024)
#     plt.plot(x,f(x))
#     plt.plot(poblacion,f(poblacion),'ob')
#     plt.title(f'epoca {epoc}')
#     plt.pause(0.1)

# def f(x): return (-x * np.sin(np.sqrt(np.abs(x))))

# enjambre_gEP(f, -512, 512, poblacion=10, dim=1, epoc_max=200, graficar=graficar)

# --------------------------------------- item b ---------------------------------------------------
def graficar_2(poblacion, epoc, x_min, x_max):
    plt.figure(1)
    plt.clf()

    X, Y = np.meshgrid(np.linspace(x_min, x_max, 1000),
                       np.linspace(x_min, x_max, 1000))
    Z = F2(np.array([X, Y]))
    plt.title(f"Iteración nro {epoc}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.pcolormesh(X, Y, Z, cmap='Greys', vmin=np.min(Z), vmax=np.max(Z))
    plt.axis([x_min, x_max, x_min, x_max])

    plt.plot(poblacion[:, 0], poblacion[:, 1], 'o')
    plt.grid(True)
    plt.pause(0.0000001)


def F2(x):
    # if x.shape[0] == 2:
    return ((x[0]**2+x[1]**2)**0.25)*(np.sin(50*((x[0]**2+x[1]**2)**0.1))**2 + 1)
    # else:
        # return ((x[:, 0]**2+x[:, 1]**2)**0.25)*(np.sin(50*((x[:, 0]**2+x[:, 1]**2)**0.1))**2 + 1)
    
enjambre_gEP(F2, -1000, 1000, poblacion=10, dim=2, epoc_max=500, graficar=graficar_2)

# --------------------------------------------------------------------------------------------------
#                                       EJERCICIO 2 
# --------------------------------------------------------------------------------------------------
# route = abspath('./data/Guia7/gr17.csv')
# d = np.genfromtxt(route, delimiter=',')
# mejor_camino, meejor_longitud = colonia_de_hormigas(d)
