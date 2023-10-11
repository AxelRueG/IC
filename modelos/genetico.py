import numpy as np
import random

# ==================================================================================================
def genetico(
    F,
    fitness,
    decode,
    gen_bits=21,
    tamanio_poblacion=20,
    target_max=512,
    target_min=-512,
    gen_target_max=None,
    num_generaciones=200,
    porcentaje_hijos=0.80,
    probabilidad_cruza=0.8,
    probabilidad_mutacion=0.40,
    min_bits_cruza=1,
    grafica=None
):

    # Parámetros
    AptitudRequerida = 1e-7
    tol = 1e-5
    gen_target_max = gen_target_max if gen_target_max else 2**gen_bits - 1
    estancamiento_aceptado = int(0.2 * num_generaciones)

    # Población inicial
    poblacion = np.random.randint(2, size=(tamanio_poblacion, gen_bits))

    fenotipo = decode(poblacion, gen_target_max, target_max, target_min)
    aptitud = fitness(fenotipo)

    mejores_apt = []
    mejor_apt, i_mejor_apt = max(aptitud), np.argmax(aptitud)
    mejores_apt.append(mejor_apt)

    generacion = 1
    cant_max_iguales = 0
    no_mejora = 0

    while generacion < num_generaciones and not no_mejora:
        max_it_anterior = mejor_apt

        # SELECCION --------------------------------------------------------------------------------
        progenitores = []
        # ordenar los individuos según su aptitud
        i_sorted = np.argsort(aptitud)[::-1]

        # Selección de padre metodo de ventanas
        for i in range(tamanio_poblacion - 2):
            progenitor_elegido = i_sorted[random.randint(
                0, len(i_sorted) - 1 - i)]
            progenitores.append(progenitor_elegido)

        # Selección de parejas de progenitores
        padres1, padres2 = progenitores[:tamanio_poblacion //
                                        2], progenitores[tamanio_poblacion // 2:]

        # CRUZA ------------------------------------------------------------------------------------
        nueva_poblacion = []
        i = 0

        # Proceso de cruza dejando espacio para una brecha generacional
        while len(nueva_poblacion) < int(porcentaje_hijos * tamanio_poblacion):
            hijo1, hijo2 = poblacion[padres1[i]
                                     ].copy(), poblacion[padres2[i]].copy()

            # Verificar si debe ocurrir la cruza y hacerla
            if random.random() < probabilidad_cruza:
                bit_cruza = random.randint(min_bits_cruza, gen_bits)
                aux = np.hstack((hijo2[:bit_cruza], hijo1[bit_cruza:]))
                hijo1 = np.hstack((hijo1[:bit_cruza], hijo2[bit_cruza:]))
                hijo2 = aux

            nueva_poblacion.append(hijo1)
            nueva_poblacion.append(hijo2)
            i += 1

        nueva_poblacion.insert(0, poblacion[i_mejor_apt].copy())

        # Tomar algunos de los mejores progenitores para pasar a la siguiente generación
        while len(nueva_poblacion) < tamanio_poblacion:
            nueva_poblacion.append(poblacion[progenitores[i]].copy())
            i += 1

        # MUTACION ---------------------------------------------------------------------------------
        for i in range(1, int(porcentaje_hijos * tamanio_poblacion)):
            # Verificar si debe ocurrir la mutación y hacerla
            if random.random() < probabilidad_mutacion:
                bit_mutacion = random.randint(0, gen_bits - 1)
                nueva_poblacion[i][bit_mutacion] = 1 - \
                    nueva_poblacion[i][bit_mutacion]

        poblacion = np.array(nueva_poblacion)

        # APTITUD NUEVA ----------------------------------------------------------------------------
        fenotipo = decode(poblacion, gen_target_max,
                          target_max, target_min)
        aptitud = fitness(fenotipo)

        # CHECKEO SI HAY MEJORA --------------------------------------------------------------------
        nueva_mejor_apt, i_nueva_mejor_apt = max(aptitud), np.argmax(aptitud)

        # Determinar la mejor aptitud de la iteración
        if nueva_mejor_apt > mejor_apt:
            mejor_apt = nueva_mejor_apt
            i_mejor_apt = i_nueva_mejor_apt

        mejores_apt.append(mejor_apt)

        generacion += 1  # aumento la generacion

        p = fenotipo[i_mejor_apt]
        pF = F(p)
        print(
            f'[{generacion}]: fenotipo => {p}, valor=> {pF}')

        if grafica:
            grafica(F, fenotipo, generacion,
                    mejores_apt, target_min, target_max)

        # Comprobar si debe salirse por estancamiento en el resultado
        if mejor_apt == max_it_anterior:
            cant_max_iguales += 1
        else:
            cant_max_iguales = 0

        if cant_max_iguales > estancamiento_aceptado:
            no_mejora = 1















# # Método del gradiente descendente
# dF = lambda x: (-np.sin(np.sqrt(np.abs(x))) - (x**2 * np.cos(np.sqrt(np.abs(x))) / (2 * np.abs(x)**(3/2))))
# cant_min_iguales = 0
# no_mejora = 0
# # Continuación del código

# a = 0.3
# x0 = np.random.randint(target_min, target_max + 1, size=(tamanio_poblacion, 1))

# x_gradiente = x0.copy()
# y_gradiente = []

# for i in range(1, num_generaciones + 1):
#     gradiente = dF(x_gradiente)
#     x_gradiente = x_gradiente - a * gradiente

#     x_gradiente[x_gradiente < -512] = -512
#     x_gradiente[x_gradiente > 512] = 512

#     y_gradiente.append(F(x_gradiente))

#     j_min_y = np.argmin(y_gradiente[-1])

#     plt.figure(3)
#     plt.clf()
#     plt.title("Iteración nro " + str(i))
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.plot(x, F(x))

#     if j_min_y == 0:
#         plt.plot(x_gradiente[1:], y_gradiente[-1][1:], 'r*')
#     else:
#         j = [0] + list(range(1, j_min_y)) + list(range(j_min_y + 1, len(y_gradiente[-1])))
#         plt.plot(x_gradiente[j], y_gradiente[-1][j], 'r*')

#     plt.plot(x_gradiente[j_min_y], y_gradiente[-1][j_min_y], 'b*')
#     plt.grid(True)
#     plt.pause(0.000001)

#     # Comprobar si debe salirse por estancamiento en el resultado
#     # if i > 1 and abs(y_gradiente[-2] - y_gradiente[-1]) < tol:
#     #     cant_min_iguales += 1
#     # else:
#     #     cant_min_iguales = 0

#     # if cant_min_iguales > estancamiento_aceptado:
#     #     no_mejora = 1

#     # if no_mejora:
#     #     break

# # plt.show()
