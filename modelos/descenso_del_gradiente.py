import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def forward_difference(f, x, h):
    """
    Calcula la derivada de la función f en el punto x utilizando la diferencia hacia adelante.
    
    Args:
    f: La función de la cual se quiere calcular la derivada.
    x: El punto en el cual se desea calcular la derivada.
    h: El tamaño del paso (diferencia) entre x y x + h.

    Returns:
    La aproximación de la derivada de f en el punto x.
    """
    df = (f(x + h) - f(x)) / h
    return df

def gradient_descent(f, initial_x, learning_rate, num_iterations):
    """
    Algoritmo de Descenso del Gradiente para minimizar una función utilizando diferencias finitas.
    
    Args:
    f: La función objetivo que se desea minimizar.
    initial_x: El punto inicial de búsqueda.
    learning_rate: La tasa de aprendizaje que controla el tamaño de los pasos en cada iteración.
    num_iterations: El número de iteraciones del algoritmo.

    Returns:
    El valor mínimo encontrado y la ubicación en la que se encuentra.
    """
    x = initial_x

    for _ in range(num_iterations):
        gradient = forward_difference(f, x, h=0.001)  # Calcular el gradiente usando diferencias finitas
        x -= learning_rate * gradient  # Actualizar la posición usando el descenso del gradiente

    # Calcular el valor mínimo encontrado
    minimum_value = f(x)

    return minimum_value, x


# Ejemplo de uso:
# Definimos una función de ejemplo
def ejemplo_funcion(x):
    return x**2

# Punto en el que queremos calcular la derivada
# x0 = np.array([1.0])


# Tasa de aprendizaje (puedes ajustar este valor)
# learning_rate = 0.1

# # Número de iteraciones (puedes ajustar este valor)
# num_iterations = 100

# Ejecutar el Descenso del Gradiente
# minimum_value, minimizer = gradient_descent(ejemplo_funcion, x0, learning_rate, num_iterations)

# print(f"Valor mínimo encontrado: {minimum_value}")
# print(f"Ubicación del mínimo: {minimizer}")

x, y = np.meshgrid(np.linspace(-5,5,50), np.linspace(-5,5,50))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(x, y, ejemplo_funcion(np.array([x,y])), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# plt.surface()