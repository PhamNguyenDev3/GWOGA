
# grey_wolf_optimizer.py
import numpy as np
import matplotlib.pyplot as plt

class GWO:
    def __init__(self, dim, popSize, Iter, lb, ub, fitness_func):
        self.dim = dim
        self.popSize = popSize
        self.Iter = Iter
        self.lb = lb
        self.ub = ub
        # self.search_space = search_space
        self.fitness_func = fitness_func

    def initialize_population(self):
        return np.random.uniform(self.lb, self.ub, size=(self.popSize, self.dim))

    def calculate_fitness(self, wolves):
        fitness = np.zeros(self.popSize)
        for i, wolf in enumerate(wolves):
            fitness[i] = self.fitness_func(wolf)
        return fitness

    def optimize(self):
        wolves = self.initialize_population()
        fitness_history = []
        array_fitness = []
        for iteration  in range(self.Iter):
            fitness = self.calculate_fitness(wolves)
            alpha_index = np.argmin(fitness)
            beta_index = np.argsort(fitness)[1]
            delta_index = np.argsort(fitness)[2]

            alpha, beta, delta = wolves[alpha_index], wolves[beta_index], wolves[delta_index]

            a = 2 - 2 * (iteration  / self.Iter)  # linearly decreased from 2 to 0

            new_wolves = np.zeros_like(wolves)

            for i in range(self.popSize):
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)

                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = abs(C1 * alpha - wolves[i])
                X1 = alpha - A1 * D_alpha

                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(C2 * beta - wolves[i])
                X2 = beta - A2 * D_beta

                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(C3 * delta - wolves[i])
                X3 = delta - A3 * D_delta

                wolves_new = (X1 + X2 + X3) / 3
                # new_wolves[i] = wolves_new

                if self.fitness_func(wolves_new) < self.fitness_func(wolves[i]):
                    new_wolves[i] = wolves_new
                else:
                    new_wolves[i] = wolves[i]

            wolves = new_wolves

            fitness_history.append(np.min(fitness))
            if iteration % 100 == 0:
                array_fitness.append(np.min(fitness_history))

        return fitness_history, array_fitness, wolves
    


class GeneticAlgorithm:
    def __init__(self, dim, popSize, Iter, lb, ub, mutation_rate, crossover_rate, fitness_func):
        self.dim = dim
        self.popSize = popSize
        self.Iter = Iter
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.fitness_func = fitness_func
        self.lb = lb
        self.ub = ub

    def initialize_population(self):
        min_val, max_val = self.lb, self.ub
        return np.random.uniform(min_val, max_val, size=(self.popSize, self.dim))

    def calculate_fitness(self, population):
        return np.array([self.fitness_func(individual) for individual in population])

 
    def selection(self, population, fitness_scores):
            # Sort the population based on fitness scores in ascending order
            sorted_indices = np.argsort(fitness_scores)
            sorted_population = population[sorted_indices]
            return sorted_population[:self.popSize]
    
    def crossover(self, parents):
        offspring = np.empty((self.popSize, self.dim))
        for i in range(0, self.popSize, 2):
            parent1_idx = i % len(parents)
            parent2_idx = (i + 1) % len(parents)
            parent1 = parents[parent1_idx]
            parent2 = parents[parent2_idx]

            crossover_point = np.random.randint(1, self.dim - 1) if self.dim >= 3 else 1

            offspring[i, 0:crossover_point] = parent1[0:crossover_point]
            offspring[i, crossover_point:] = parent2[crossover_point:]
            offspring[i + 1, 0:crossover_point] = parent2[0:crossover_point]
            offspring[i + 1, crossover_point:] = parent1[crossover_point:]

        return offspring

    def mutate(self, population):
        for i in range(self.popSize):
            for j in range(self.dim):
                if np.random.rand() < self.mutation_rate:
                    min_val = max(self.lb, population[i][j] - 0.1 * abs(self.lb))
                    max_val = min(self.ub, population[i][j] + 0.1 * abs(self.ub))
                    population[i][j] = np.random.uniform(min_val, max_val)
        return population

    def optimize(self):
        population = self.initialize_population()
        fitness_history = []
        array_fitness = []

        for iteration in range(self.Iter):
            fitness_scores = self.calculate_fitness(population)

            parents = self.selection(population, fitness_scores)
            children = self.crossover(parents)
            population_new = self.mutate(children)

            population = np.vstack((population, population_new))
            # Selection is now after crossover and mutation
            fitness_scores = self.calculate_fitness(population)
            sorted_indices = np.argsort(fitness_scores)
            population = population[sorted_indices[:self.popSize]]
            fitness_history.append(np.min(fitness_scores))

            if iteration % 100 == 0:
                array_fitness.append(np.min(fitness_history))

        return fitness_history, array_fitness
    

def run(dim, popSize, Iter, lb, ub, mutation_rate,crossover_rate , fitness_func):


    # Instantiate GWO and GA
    gwo = GWO(dim, popSize, Iter, lb, ub, fitness_func)
    ga = GeneticAlgorithm(dim, popSize, Iter, lb, ub, mutation_rate, crossover_rate, fitness_func)

    # Run optimization for GWO
    fitness_history_gwo, array_fitness_gwo, wolves_gwo = gwo.optimize()

    # Use GWO result to initialize population for GA
    initial_population_ga = wolves_gwo

    # Update GA with the initial population from GWO
    ga.initialize_population = lambda: initial_population_ga

    # Run optimization for GA
    fitness_history_ga, array_fitness_ga = ga.optimize()

    return fitness_history_ga, array_fitness_ga



    