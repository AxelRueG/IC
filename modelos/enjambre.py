import numpy as np
import matplotlib.pyplot as plt


def enjambre_gEP(f, x_min, x_max, poblacion=10, dim=1, epoc_max=200, graficar=None):
    '''
    Parámetros del algoritmo

    Args:
    f:          funcion a minimizar
    x_min:      valor minimo del dominio del problema
    x_max:      valor maximo del dominio del problema
    poblacion:  numero de particluas
    dim:        cantidad de datos en el problema
    epoc_max:   numero maximo de epocas
    grafica:    funcion para graficar la animacion

    Returns:
    mejor valor obtenido
    '''

    # ---- Inicializacion --------------------------------------------------------------------------
    x = np.random.uniform(x_min, x_max, (poblacion, dim))
    y = x  # mejor local
    # v = np.random.uniform(size=(poblacion, dim)) - 0.5
    v = np.zeros((poblacion, dim))
    estancamiento_aceptado = 100

    c1 = np.linspace(0.5, 0.2, epoc_max)
    c2 = np.linspace(0.2, 0.5, epoc_max)

    ym = y[np.argmin(f(y))].copy()  # mejor global
    epoc = 0
    estancamiento = 0
    # ---- Aprendizaje -----------------------------------------------------------------------------
    while (epoc < epoc_max and estancamiento < estancamiento_aceptado):
        ym_old = ym.copy()
        # busco la mejor posicion
        for k in range(x.shape[0]):
            if f(x[k]) < f(y[k]):
                y[k] = x[k]
            if f(y[k]) < f(ym):
                ym = y[k].copy()

        # actualizo los pesos
        r1 = np.random.uniform(size=(poblacion, dim))
        r2 = np.random.uniform(size=(poblacion, dim))

        # calculo las nuevas posicioens
        for k in range(x.shape[0]):
            v[k] = 0.1*v[k] + c1[epoc]*r1[k]*(y[k] - x[k]) + c2[epoc]*r2[k]*(ym - x[k])
            aux = x[k] + v[k]
            # checkeamos que no se valla del limite
            # (lo hicimos con un for porque si es de una dimencion mayor explota)
            for i in range(len(aux)):
                if x_min < aux[i] < x_max:
                    x[k][i] = aux[i]

        if graficar:
            graficar(x, epoc, x_min, x_max)

        epoc += 1

        # checkeo cuantas veces tenemos el mismo minimo global para salir del while antes
        if np.all(ym_old != ym):
            estancamiento = 0
        else:
            estancamiento += 1

    return ym


def calcular_probabilidades(sigma, eta, caminos, alpha, beta):
    '''
    Parámetros del algoritmo

    Args:
    N:          feromonas del nodo i
    eta:        inverso de la distancia
    caminos:    lista de nodos tabu
    alpha:      parametro alpha para la probabilidad
    beta:       parametro beta para la probabilidad

    Returns:
    probabilidades de elegir cada camino
    '''
    # creamos mascara
    u = np.ones(len(sigma), dtype=bool)
    u[caminos] = False
    # calculamos la prob
    sigma_iu = sigma[u] # -> tengo los sigma que todavia no visito
    eta_iu = eta[u]     # -> tengo las distancias que no recorrio
    prob = np.zeros(len(sigma))
    prob[u] = (sigma_iu**alpha * eta_iu**beta) / (sigma_iu**alpha @ eta_iu**beta)
    return prob


def colonia_de_hormigas(d, N=10, alpha=1.0, beta=1.0, p=0.2, Q=0.8, iterations=1000):
    '''
    Parámetros del algoritmo

    Args:
    d:      matriz de distancias
    N:      numero de hormigas
    alpha:  parametro alpha para la probabilidad
    beta:   parametro beta para la probabilidad
    p:      factor de evaporación de feromonas
    Q:      Cantidad de feromonas a depositar
    iterations: numero de iteraciones

    Returns:
    mejor camino encontrado junto a su distancia
    '''

    t = 0
    min_it = 5  # Número para cortar cuando todas las hormigas siguen el mismo camino
    mismo_camino = 0

    # Inicialización de feromonas y distancias
    nodos = d.shape[0]  # Número de nodos
    sigma_0 = 1.0  # Nivel inicial de feromonas
    sigma = np.random.uniform(0, sigma_0, (nodos, nodos))
    sigma = (sigma + sigma.T)/2
    np.fill_diagonal(sigma, 0)

    # precalculamos eta
    eta = 1/d                               # diag principal = inf
    np.fill_diagonal(eta, 0)                # fiag principal = 0

    # Inicialización de la mejor solución encontrada
    mejor_camino = None
    mejor_longitud = np.inf
    ml = []

    # Ciclo principal
    while t < iterations and mismo_camino < min_it:
        caminos = [[] for _ in range(N)]

        for ant in range(N):
            camino = [0]  # Inicializar el camino

            # ---- recorrer todos los nodos --------------------------------------------------------
            while len(camino) < nodos:
                i = camino[-1]
                # evaluamos las probabilidades
                probabilidades = calcular_probabilidades(
                    sigma[i], eta[i], camino, alpha, beta)
                siguiente_nodo = np.random.choice(
                    range(nodos), p=probabilidades)
                camino.append(siguiente_nodo)
            # volvemos al origen
            camino.append(camino[0])
            # actualizamos la lista de caminos recorrido por hormiga
            caminos[ant] = camino

            longitud_camino = np.sum([d[camino[i], camino[i + 1]]
                                      for i in range(len(camino) - 1)])

            # actualizamos el historico del mejor camino
            if longitud_camino < mejor_longitud:
                mejor_camino = camino.copy()
                mejor_longitud = longitud_camino
            
            ml.append(mejor_longitud)

        # ---- reducir las feromonas ---------------------------------------------------------------
        sigma = (1 - p) * sigma

        plt.figure(1)
        plt.clf()
        plt.subplot(1,2,1)
        plt.plot(np.arange(len(ml)), ml)
        plt.title(f'mejor distancia {mejor_longitud}')
        plt.subplot(1,2,2)
        plt.imshow(sigma, cmap='viridis')
        plt.colorbar()        
        plt.title(f'feromonas en t={t}')
        plt.pause(0.00001)

        # ---- depositar feromonas -----------------------------------------------------------------
        for k in range(N):
            for i in range(len(caminos[k]) - 1):
                delta_sigma_k_ij = Q    # uniforme
                # mantenemos simetria:
                sigma[caminos[k][i], caminos[k][i+1]] += delta_sigma_k_ij
                sigma[caminos[k][i+1], caminos[k][i]] += delta_sigma_k_ij

        # ---- corte temprano del entrenamiento ----------------------------------------------------
        if all([l == caminos[0] for l in caminos]):
            mismo_camino += 1
            print('caminos iguales')
        else:
            mismo_camino = 0

        t += 1

    for camino in caminos:
        print(camino)
    print("Numero de epocas recorridas: ", t)
    print("Mejor camino encontrado:", mejor_camino)
    print("Longitud del mejor camino:", mejor_longitud)
    # print(sigma)

    return mejor_camino, mejor_longitud
