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

    for i in range(num_iterations):
        gradient = forward_difference(f, x, h=0.001)  # Calcular el gradiente usando diferencias finitas
        x = x - learning_rate * gradient  # Actualizar la posición usando el descenso del gradiente

    # Calcular el valor mínimo encontrado
    minimum_value = f(x)

    return minimum_value, x
