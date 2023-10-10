import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator
from modelos import gradiente


def genetico(code, decode, range, max_epoc):
    poblacion = code(np.random.randint())




'''
Ejercicio 1a con 10 bits podemos representar todos los puntos
'''
f1 = lambda x: -x * np.sin(np.sqrt(np.abs(x)))
x = np.linspace(-512,511,1024)

plt.plot(x,f1(x))
plt.show()


















# # Ejemplo de uso:
# # Definimos una función de ejemplo que queremos minimizar
# f = lambda x: x[1]**2 + x[0]**2

# # Punto inicial de búsqueda
# initial_x = np.array([1.0, 1.0])

# # Tasa de aprendizaje (puedes ajustar este valor)
# learning_rate = 0.1

# # Número de iteraciones (puedes ajustar este valor)
# num_iterations = 100

# # Ejecutar el Descenso del Gradiente
# minimum_value, minimizer = gradient_descent(f, initial_x, learning_rate, num_iterations)

# print(f"Valor mínimo encontrado: {minimum_value}")
# print(f"Ubicación del mínimo: {minimizer}")

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# X,Y = np.meshgrid(np.linspace(-2,2,100), np.linspace(-2,2,100))
# Z = f(np.array([X,Y]))

# # Plot the surface.
# # surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)

# ax.scatter(minimizer[0],minimizer[1],minimum_value, marker='o', cmap='viridis')
# surf = ax.plot_wireframe(X, Y, Z)

# # Customize the z axis.
# ax.set_zlim(-1.01, 5.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# # Add a color bar which maps values to colors.
# # fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()