import numpy as np

# Función de inicialización de la población
def initialize_population(population_size, min_value, max_value):
    return np.random.uniform(min_value, max_value, size=population_size)

# @TODO solucionar por competencia
# Función de selección de padres (torneo binario)
def select_parents(population, fitness_values):
    parent_indices = np.random.choice(len(population), size=2, replace=False)
    return population[parent_indices]

# Función de cruce (combinación de dos padres)


def crossover(parent1, parent2):
    alpha = np.random.uniform(0, 1)
    child = alpha * parent1 + (1 - alpha) * parent2
    return child

# Función de mutación
def mutate(individual, mutation_rate, min_value, max_value):
    if np.random.rand() < mutation_rate:
        mutation = np.random.uniform(min_value, max_value)
        individual += mutation
    return individual


def genetico(f, population_size=100, mutation_rate=0.1, generations=100, min_value=-512.0, max_value=511.0):
    # Parámetros del algoritmo genético
    population_size = 100
    mutation_rate = 0.1
    generations = 100
    min_value = -512.0
    max_value = 511.0

    # Inicialización de la población
    population = initialize_population(population_size, min_value, max_value)

    # Ciclo principal del algoritmo genético
    for generation in range(generations):
        # Evaluación de la aptitud (fitness) de cada individuo
        fitness_values = np.array([f(x) for x in population])

        # Selección de padres y creación de la nueva generación
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = select_parents(population, fitness_values)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate, min_value, max_value)
            new_population.append(child)
        population = np.array(new_population)

        # Encontrar el mejor individuo en esta generación
        best_individual = population[np.argmax(fitness_values)]
        best_fitness = f(best_individual)

        print(
            f"Generación {generation}: Mejor fitness = {best_fitness}, Mejor individuo = {best_individual}")
