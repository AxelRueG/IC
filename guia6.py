import matplotlib.pyplot as plt
import numpy as np
from modelos.genetico import genetico
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from modelos import gradiente

def escalar(number, original_max, target_max, target_min):
    return (number/original_max) * (target_max - target_min) + target_min

# ======= Función 1 ================================================================================
def decodificar(poblacion, gen_target_max, target_max, target_min):
    num_bits = poblacion.shape[1]
    patron_dec = poblacion @ 2 ** np.arange(num_bits)[::-1]
    return escalar(patron_dec, gen_target_max, target_max, target_min)


def grafica_1(F, fenotipo, generacion, mejores_apt, target_min, target_max):
    pF = F(fenotipo)

    plt.figure(1)
    plt.clf()

    plt.subplot(1, 2, 1)
    plt.title(f"Iteración nro {generacion}")
    plt.xlabel("Iteración")
    plt.ylabel("Mejor aptitud")
    plt.plot(range(1, generacion + 1), mejores_apt)
    plt.axis([0, generacion, max(mejores_apt)-50, max(mejores_apt)+50])

    x = np.arange(target_min, target_max + 1)
    plt.subplot(1, 2, 2)
    plt.title(f"Iteración nro {generacion}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, F(x))
    # plt.plot(p, pF, 'b*')
    plt.plot(fenotipo, pF, 'b*')
    plt.grid(True)
    plt.pause(0.000001)


def F1(x): return (-x * np.sin(np.sqrt(np.abs(x))))
def fitness1(x): return 1 - F1(x)

# ======= Función 2 ================================================================================
def decodificar_fun2(poblacion, gen_target_max, target_max, target_min):
    num_bits = poblacion.shape[1]
    x_decode = poblacion[:,:num_bits//2] @ 2 ** np.arange(num_bits // 2)[::-1]
    y_decode = poblacion[:,num_bits//2:] @ 2 ** np.arange(num_bits // 2)[::-1]
    x = escalar(x_decode, gen_target_max, target_max, target_min)
    y = escalar(y_decode, gen_target_max, target_max, target_min)
    return np.hstack((np.transpose([x]),np.transpose([y])))
    


def grafica_1(F, fenotipo, generacion, mejores_apt, target_min, target_max):
    pF = F(fenotipo)

    plt.figure(1)
    plt.clf()

    plt.subplot(1, 2, 1)
    plt.title(f"Iteración nro {generacion}")
    plt.xlabel("Iteración")
    plt.ylabel("Mejor aptitud")
    plt.plot(range(1, generacion + 1), mejores_apt)
    plt.axis([0, generacion, max(mejores_apt)-50, max(mejores_apt)+50])

    
    X,Y = np.meshgrid(np.linspace(-100,100,2000), np.linspace(-100,100,2000))
    Z = F2(np.array([X,Y]))
    plt.subplot(1, 2, 2)
    plt.title(f"Iteración nro {generacion}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.pcolormesh(X,Y,Z, cmap='Greys', vmin=np.min(Z), vmax=np.max(Z))
    # set the limits of the plot to the limits of the data
    plt.axis([-100,100,-100,100])

    plt.plot(fenotipo[:,0], fenotipo[:,1], 'o')
    plt.grid(True)
    plt.pause(0.0000001)


def F2(x):
    if x.shape[0] == 2:
        return ((x[0]**2+x[1]**2)**0.25)*(np.sin(50*((x[0]**2+x[1]**2)**0.1))**2 + 1)
    else: 
        return ((x[:,0]**2+x[:,1]**2)**0.25)*(np.sin(50*((x[:,0]**2+x[:,1]**2)**0.1))**2 + 1)
fitness2 = lambda x: 1 - F2(x)


# # Ejercicio 1a
# genetico(
#     F1,
#     fitness1,
#     decode=decodificar,
#     gen_bits=21,
#     tamanio_poblacion=20,
#     target_max=512,
#     target_min=-512,
#     num_generaciones=200,
#     porcentaje_hijos=0.80,
#     probabilidad_cruza=0.8,
#     probabilidad_mutacion=0.40,
#     min_bits_cruza=1,
#     grafica=grafica_1)

# Ejercicio 1b
genetico(
    F2,
    fitness2,
    decode=decodificar_fun2,
    gen_bits=40,
    tamanio_poblacion=30,
    gen_target_max=2**20-1,
    target_max=100,
    target_min=-100,
    num_generaciones=200,
    porcentaje_hijos=0.80,
    probabilidad_cruza=0.8,
    probabilidad_mutacion=0.40,
    # grafica=grafica_1,
    min_bits_cruza=1)

plt.show()


'''
# Ejemplo GRAFICA 3d:
# Definimos una función de ejemplo que queremos minimizar
# f = lambda x: x[1]**2 + x[0]**2

# Punto inicial de búsqueda
initial_x = np.array([1.0, 1.0])

# Tasa de aprendizaje (puedes ajustar este valor)
learning_rate = 0.1

# Número de iteraciones (puedes ajustar este valor)
num_iterations = 100

# Ejecutar el Descenso del Gradiente
minimum_value, minimizer = gradiente.gradient_descent(f, initial_x, learning_rate, num_iterations)

print(f"Valor mínimo encontrado: {minimum_value}")
print(f"Ubicación del mínimo: {minimizer}")

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

X,Y = np.meshgrid(np.linspace(-100,100,2000), np.linspace(-100,100,2000))
Z = f(np.array([X,Y]))

# Plot the surface.
ax.scatter(minimizer[0],minimizer[1],minimum_value, marker='o', cmap='viridis')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
# surf = ax.plot_wireframe(X, Y, Z)

# Customize the z axis.
# ax.set_zlim(-1.01, 5.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
'''