# # import numpy as np
# # import matplotlib.pyplot as plt

# # class GeneticAlgorithm:
# #     def __init__(self, dim, popSize, Iter, lb, ub,mutation_rate, crossover_rate, fitness_func):
# #         self.dim = dim
# #         self.popSize = popSize
# #         self.Iter = Iter
# #         self.mutation_rate = mutation_rate
# #         self.crossover_rate = crossover_rate
# #         self.fitness_func = fitness_func
# #         self.lb = lb
# #         self.ub = ub

# #     def initialize_population(self):
# #         min_val, max_val = self.lb, self.ub
# #         return np.random.uniform(min_val, max_val, size=(self.popSize, self.dim))

# #     def calculate_fitness(self, population):
# #         return np.array([self.fitness_func(individual) for individual in population])



# #     # def selection(self, population, fitness_scores):
# #     #     # Tính toán xác suất chọn lựa dựa trên fitness scores
# #     #     # fitness_scores = np.max(fitness_scores) - fitness_scores
# #     #     # probabilities = fitness_scores / np.sum(fitness_scores)

# #     #     # # Đảm bảo xác suất không có giá trị âm và tổng bằng 1
# #     #     # probabilities = np.clip(probabilities, 0, None)
# #     #     # probabilities /= np.sum(probabilities)

# #     #     # selected_indices = np.random.choice(range(len(population)), size=len(population), p=probabilities)
# #     #     # return population[selected_indices]
# #     #     # selected_indices = np.argsort(fitness_scores)
# #     #     wolves_list = [wolves_list[i] for i in population[:self.popSize]]
# #     #     wolves = np.array(wolves_list)

# #     #     return population[selected_indices]
# #     # def selection(self, population, fitness_scores):
# #     #     # Sort the population based on fitness scores in ascending order
# #     #     sorted_indices = np.argsort(fitness_scores)
# #     #     sorted_population = population[sorted_indices]
# #     # #     return sorted_population[:self.popSize]
# #     # def selection(self, population, fitness_scores):
# #     #     # Tính toán xác suất chọn lựa dựa trên fitness scores
# #     #     fitness_scores = np.max(fitness_scores) - fitness_scores
# #     #     probabilities = fitness_scores / np.sum(fitness_scores)

# #     #     # Đảm bảo xác suất không có giá trị âm và tổng bằng 1
# #     #     probabilities = np.clip(probabilities, 0, None)
# #     #     probabilities /= np.sum(probabilities)

# #     #     selected_indices = np.random.choice(range(len(population)), size=len(population), p=probabilities)
# #     #     return population[selected_indices]
    
# #     def selection(self, population, fitness_scores):
# #             inv_fitness_scores = 1 / (fitness_scores + 1e-10)
# #             probabilities = inv_fitness_scores / np.sum(inv_fitness_scores)
# #             selected_indices = np.random.choice(range(len(population)), size=len(population), p=probabilities)
# #             return population[selected_indices]
# #     # def crossover(self, parents):
# #     #     children = []
# #     #     for i in range(0, len(parents), 2):
# #     #         parent1, parent2 = parents[i], parents[i + 1]
# #     #         if np.random.rand() < self.crossover_rate:
# #     #             crossover_point = np.random.randint(1, self.dim - 1)
# #     #             child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
# #     #             child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
# #     #         else:
# #     #             child1, child2 = parent1.copy(), parent2.copy()
# #     #         children.extend([child1, child2])
# #     #     return np.array(children)
# #     def crossover(self, parents):
# #         offspring = np.empty((self.popSize, self.dim))
# #         for i in range(0, self.popSize, 2):
# #             parent1_idx = i % len(parents)
# #             parent2_idx = (i + 1) % len(parents)
# #             parent1 = parents[parent1_idx]
# #             parent2 = parents[parent2_idx]
            
# #             if self.dim >= 3:
# #                 crossover_point = np.random.randint(1, self.dim - 1)
# #             else:
# #                 crossover_point = 1

# #             offspring[i, 0:crossover_point] = parent1[0:crossover_point]
# #             offspring[i, crossover_point:] = parent2[crossover_point:]
# #             offspring[i + 1, 0:crossover_point] = parent2[0:crossover_point]
# #             offspring[i + 1, crossover_point:] = parent1[crossover_point:]

# #         return offspring

# #     def mutate(self, population):
# #         for i in range(self.popSize):
# #             for j in range(self.dim):
# #                 if np.random.rand() < self.mutation_rate:
# #                     min_val = max(self.lb, population[i][j] - 0.1 * abs(self.lb))
# #                     max_val = min(self.ub, population[i][j] + 0.1 * abs(self.ub))
# #                     population[i][j] = np.random.uniform(min_val, max_val)

# #         return population

# #     # def mutate(self, population):
# #     #     min_val, max_val = self.lb, self.ub
# #     #     for i in range(len(population)):
# #     #         for j in range(self.dim):
# #     #             if np.random.rand() < self.mutation_rate:
# #     #                 population[i][j] += np.random.uniform(-1, 1)
# #     #                 # Đảm bảo giá trị không vượt ra khỏi giới hạn của không gian tìm kiếm
# #     #                 population[i][j] = np.clip(population[i][j], min_val, max_val)
# #     #     return population

# #     # def optimize(self):
# #     #     population = self.initialize_population()
# #     #     fitness_history = []
# #     #     array_fitness = []
# #     #     fitness_scores = self.calculate_fitness(population)
# #     #     fitness_history.append(np.min(fitness_scores))

# #     #     for iteration  in range(self.Iter):
# #     #         fitness_scores = self.calculate_fitness(population)

# #     #         children = self.crossover(parents)
# #     #         population = self.mutate(children)
            
# #     #         parents = self.selection(population, fitness_scores)
# #     #         fitness_history.append(np.min(fitness_scores))

# #     #         if iteration % 100 == 0:
# #     #             array_fitness.append(np.min(fitness_history))

# #     #     return fitness_history, array_fitness


# #     def optimize(self):
# #         population = self.initialize_population()
# #         fitness_history = []
# #         array_fitness = []
# #         fitness_scores = self.calculate_fitness(population)
# #         fitness_history.append(np.min(fitness_scores))

# #         for iteration in range(self.Iter):

# #             fitness_scores = self.calculate_fitness(population)

# #             parents = self.selection(population, fitness_scores)
# #             children = self.crossover(parents)
# #             population = self.mutate(children)

# #             # Selection is now after crossover and mutation
# #             fitness_scores = self.calculate_fitness(population)
# #             # sorted_indices = np.argsort(fitness_scores)
# #             # population = [population[i] for i in sorted_indices[:self.popSize]]
# #             # fitness_history.append(np.min(fitness_scores))

# #             sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k])
# #             population = [population[i] for i in sorted_indices[:self.popSize]]
# #             fitness_history.append(np.min(fitness_scores))


# #             if iteration % 100 == 0:
# #                 array_fitness.append(np.min(fitness_history))

# #         return fitness_history, array_fitness


import numpy as np
import matplotlib.pyplot as plt

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

    # def selection(self, population, fitness_scores):
    #     # Inverse fitness scores to favor lower fitness values
    #     inv_fitness_scores = 1 / (fitness_scores + 1e-10)
    #     probabilities = inv_fitness_scores / np.sum(inv_fitness_scores)
    #     selected_indices = np.random.choice(range(len(population)), size=self.popSize, p=probabilities)
    #     return population[selected_indices]
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

# # Example fitness function
# def fitness_function(individual):
#     return np.sum(individual**2)

# # Parameters
# dim = 10
# popSize = 50
# Iter = 1000
# lb = -5
# ub = 5
# mutation_rate = 0.1
# crossover_rate = 0.7

# # Create GeneticAlgorithm instance
# ga = GeneticAlgorithm(dim, popSize, Iter, lb, ub, mutation_rate, crossover_rate, fitness_function)

# # Run optimization
# fitness_history, array_fitness = ga.optimize()

# # Plotting fitness history
# plt.plot(fitness_history)
# plt.xlabel('Iteration')
# plt.ylabel('Fitness')
# plt.xscale('log')
# plt.title('Fitness over iterations')
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# class GeneticAlgorithm:
#     def __init__(self, dim, popSize, Iter, lb, ub, mutation_rate, crossover_rate, fitness_func):
#         self.dim = dim
#         self.popSize = popSize
#         self.Iter = Iter
#         self.mutation_rate = mutation_rate
#         self.crossover_rate = crossover_rate
#         self.fitness_func = fitness_func
#         self.lb = lb
#         self.ub = ub

#     def initialize_population(self):
#         min_val, max_val = self.lb, self.ub
#         return np.random.uniform(min_val, max_val, size=(self.popSize, self.dim))

#     def calculate_fitness(self, population):
#         return np.array([self.fitness_func(individual) for individual in population])

#     def selection(self, population, fitness_scores):
#         # Sort the population based on fitness scores in ascending order
#         sorted_indices = np.argsort(fitness_scores)
#         sorted_population = population[sorted_indices]
#         return sorted_population[:self.popSize]
    
#     def crossover(self, parents):
#         offspring = np.empty((self.popSize, self.dim))
#         for i in range(0, self.popSize, 2):
#             parent1_idx = i % len(parents)
#             parent2_idx = (i + 1) % len(parents)
#             parent1 = parents[parent1_idx]
#             parent2 = parents[parent2_idx]

#             crossover_point = np.random.randint(1, self.dim - 1) if self.dim >= 3 else 1

#             offspring[i, 0:crossover_point] = parent1[0:crossover_point]
#             offspring[i, crossover_point:] = parent2[crossover_point:]
#             offspring[i + 1, 0:crossover_point] = parent2[0:crossover_point]
#             offspring[i + 1, crossover_point:] = parent1[crossover_point:]

#         return offspring

#     def mutate(self, population):
#         for i in range(self.popSize):
#             for j in range(self.dim):
#                 if np.random.rand() < self.mutation_rate:
#                     min_val = max(self.lb, population[i][j] - 0.1 * abs(self.lb))
#                     max_val = min(self.ub, population[i][j] + 0.1 * abs(self.ub))
#                     population[i][j] = np.random.uniform(min_val, max_val)
#         return population

#     def optimize(self):
#         population = self.initialize_population()
#         fitness_history = []
#         array_fitness = []

#         for iteration in range(self.Iter):
#             fitness_scores = self.calculate_fitness(population)

#             parents = self.selection(population, fitness_scores)
#             children = self.crossover(parents)
#             population_new = self.mutate(children)

#             # Selection is now after crossover and mutation
#             population += population_new
#             fitness_scores = self.calculate_fitness(population)
#             sorted_indices = np.argsort(fitness_scores)
#             population = population[sorted_indices[:self.popSize]]
#             fitness_history.append(np.min(fitness_scores))

#             if iteration % 100 == 0:
#                 array_fitness.append(np.min(fitness_history))

#         return fitness_history, array_fitness

# # Example fitness function
# def fitness_function(individual):
#     return np.sum(individual**2)

# # Parameters
# dim = 10
# popSize = 50
# Iter = 1000
# lb = -5
# ub = 5
# mutation_rate = 0.01
# crossover_rate = 0.7

# # Create GeneticAlgorithm instance
# ga = GeneticAlgorithm(dim, popSize, Iter, lb, ub, mutation_rate, crossover_rate, fitness_function)

# # Run optimization
# fitness_history, array_fitness = ga.optimize()

# # Plotting fitness history
# plt.plot(fitness_history)
# plt.xlabel('Iteration')
# plt.ylabel('Fitness')
# plt.xscale('log')
# plt.title('Fitness over iterations')
# plt.show()
