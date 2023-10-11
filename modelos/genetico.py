import numpy as np
import random
import matplotlib.pyplot as plt

def escalar(number, original_max, original_min, target_max, target_min):
    escalado = (number - original_min) / (original_max - original_min) * (target_max - target_min) + target_min
    return escalado

def seleccionar_padres(pool, tamanio_poblacion):
    padres1 = []
    padres2 = []
    for i in range(len(pool)):
        padre1 = pool[i]
        j = (i + 1) % (tamanio_poblacion - 1) + 1
        while pool[j] == padre1:
            j = (j + 1) % (tamanio_poblacion - 1) + 1
        padre2 = pool[j]

        padres1.append(padre1)
        padres2.append(padre2)

    return padres1, padres2


def decodificar(poblacion, gen_target_max, gen_target_min, target_max, target_min):
    fenotipo = np.zeros(poblacion.shape[0])
    for i in range(poblacion.shape[0]):
        # Decodificar patrón
        patron_str = ''.join(map(str, poblacion[i]))
        patron_dec = int(patron_str, 2)
        fenotipo[i] = escalar(patron_dec, gen_target_max, gen_target_min, target_max, target_min)

    return fenotipo

def escalar(valor, valor_max, valor_min, nuevo_max, nuevo_min):
    return nuevo_min + ((valor - valor_min) * (nuevo_max - nuevo_min)) / (valor_max - valor_min)



# ==================================================================================================
# Función 1
def F1(x):
    return -x * np.sin(np.sqrt(np.abs(x)))

def dF1(x):
    return -np.sin(np.sqrt(np.abs(x))) - (x**2 * np.cos(np.sqrt(np.abs(x))) / (2 * np.abs(x)**(3/2)))

# Parámetros
AptitudRequerida = 1e-7
gen_bits = 21
tamanio_poblacion = 20
target_max = 512
target_min = -512
gen_target_max = 2**gen_bits - 1
gen_target_min = 0
cant_it = 200
porcentaje_hijos = 0.80
probabilidad_cruza = 0.8
probabilidad_mutacion = 0.40
min_bits_cruza = 1
estancamiento_aceptado = int(0.2 * cant_it)
tol = 1e-5

# Función de fitness
def fitness1(x):
    return 1 - F1(x)

# Población inicial
poblacion = np.random.randint(2, size=(tamanio_poblacion, gen_bits))

def decodificar(poblacion, gen_target_max, gen_target_min, target_max, target_min):
    return target_min + (target_max - target_min) * np.sum(poblacion * 2**np.arange(gen_bits - 1, -1, -1), axis=1) / gen_target_max

fenotipo = decodificar(poblacion, gen_target_max, gen_target_min, target_max, target_min)

aptitud = np.zeros(tamanio_poblacion)

for i in range(tamanio_poblacion):
    aptitud[i] = fitness1(fenotipo[i])

mejores_apt = []
mejor_apt, i_mejor_apt = max(aptitud), np.argmax(aptitud)
mejores_apt.append(mejor_apt)

it_actual = 1
cant_max_iguales = 0
no_mejora = 0

while it_actual < cant_it and not no_mejora:
    max_it_anterior = mejor_apt

    # Selección de progenitores por método de ventanas
    progenitores = []

    # 1. Ordenar los individuos según su aptitud de mejor a peor
    i_sorted = np.argsort(aptitud)[::-1]

    # 2. Selección
    for i in range(tamanio_poblacion - 1):
        progenitor_elegido = i_sorted[random.randint(0, len(i_sorted) - 1 - i)]
        progenitores.append(progenitor_elegido)

    # Selección de parejas de progenitores
    padres1, padres2 = progenitores[:tamanio_poblacion // 2], progenitores[tamanio_poblacion // 2:]

    # 3. Cruza
    nueva_poblacion = []

    i = 0

    # Proceso de cruza dejando espacio para una brecha generacional
    while len(nueva_poblacion) < int(porcentaje_hijos * tamanio_poblacion):
        hijo1, hijo2 = poblacion[padres1[i]].copy(), poblacion[padres2[i]].copy()

        # Verificar si debe ocurrir la cruza y hacerla
        if random.random() < probabilidad_cruza:
            bit_cruza = random.randint(min_bits_cruza, gen_bits)
            aux = np.hstack((hijo2[:bit_cruza], hijo1[bit_cruza:]))
            hijo1[:bit_cruza], hijo2[bit_cruza:] = hijo2[:bit_cruza], hijo1[bit_cruza:]
            hijo2 = aux

        nueva_poblacion.append(hijo1)
        nueva_poblacion.append(hijo2)
        i += 1

    nueva_poblacion[0] = poblacion[i_mejor_apt].copy()

    # Tomar algunos de los mejores progenitores para pasar a la siguiente generación
    while len(nueva_poblacion) < tamanio_poblacion:
        nueva_poblacion.append(poblacion[progenitores[i]].copy())
        i += 1

    # Proceso de mutación
    for i in range(1, int(porcentaje_hijos * tamanio_poblacion)):
        # Verificar si debe ocurrir la mutación y hacerla
        if random.random() < probabilidad_mutacion:
            bit_mutacion = random.randint(0, gen_bits - 1)
            nueva_poblacion[i][bit_mutacion] = 1 - nueva_poblacion[i][bit_mutacion]

    poblacion = np.array(nueva_poblacion)

    fenotipo = decodificar(poblacion, gen_target_max, gen_target_min, target_max, target_min)

    for i in range(tamanio_poblacion):
        aptitud[i] = fitness1(fenotipo[i])

    nueva_mejor_apt, i_nueva_mejor_apt = max(aptitud), np.argmax(aptitud)

    # Determinar la mejor aptitud de la iteración
    if nueva_mejor_apt > mejor_apt:
        mejor_apt = nueva_mejor_apt
        i_mejor_apt = i_nueva_mejor_apt

    mejores_apt.append(mejor_apt)

    it_actual += 1

    p = fenotipo[i_mejor_apt]
    pF = F1(p)

    plt.figure(1)
    plt.clf()

    plt.subplot(1, 2, 1)
    plt.title("Iteración nro " + str(it_actual))
    plt.xlabel("Iteración")
    plt.ylabel("Mejor aptitud")
    plt.plot(range(1, it_actual + 1), mejores_apt)
    plt.axis([0, it_actual, 0, max(mejores_apt) + 100])

    x = np.arange(target_min, target_max + 1)
    plt.subplot(1, 2, 2)
    plt.title("Iteración nro " + str(it_actual))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, F1(x))
    plt.plot(p, pF, 'b*')
    plt.grid(True)
    plt.pause(0.000001)

    # Comprobar si debe salirse por estancamiento en el resultado
    if mejor_apt == max_it_anterior:
        cant_max_iguales += 1
    else:
        cant_max_iguales = 0

    if cant_max_iguales > estancamiento_aceptado:
        no_mejora = 1

# Método del gradiente descendente
cant_min_iguales = 0
no_mejora = 0
# Continuación del código

a = 0.3
x0 = np.random.randint(target_min, target_max + 1, size=(tamanio_poblacion, 1))

x_gradiente = x0.copy()
y_gradiente = []

for i in range(1, cant_it + 1):
    gradiente = dF1(x_gradiente)
    x_gradiente = x_gradiente - a * gradiente

    x_gradiente[x_gradiente < -512] = -512
    x_gradiente[x_gradiente > 512] = 512

    y_gradiente.append(F1(x_gradiente))

    j_min_y = np.argmin(y_gradiente[-1])

    plt.figure(3)
    plt.clf()
    plt.title("Iteración nro " + str(i))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, F1(x))
    
    if j_min_y == 0:
        plt.plot(x_gradiente[1:], y_gradiente[-1][1:], 'r*')
    else:
        j = [0] + list(range(1, j_min_y)) + list(range(j_min_y + 1, len(y_gradiente[-1])))
        plt.plot(x_gradiente[j], y_gradiente[-1][j], 'r*')
    
    plt.plot(x_gradiente[j_min_y], y_gradiente[-1][j_min_y], 'b*')
    plt.grid(True)
    plt.pause(0.000001)

    # Comprobar si debe salirse por estancamiento en el resultado
    # if i > 1 and abs(y_gradiente[-2] - y_gradiente[-1]) < tol:
    #     cant_min_iguales += 1
    # else:
    #     cant_min_iguales = 0

    # if cant_min_iguales > estancamiento_aceptado:
    #     no_mejora = 1

    # if no_mejora:
    #     break

# plt.show()