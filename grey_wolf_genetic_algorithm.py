
import numpy as np
import matplotlib.pyplot as plt
import random

class GWO_GA:
    def __init__(self, dim, popSize, Iter, lb, ub, mutation_rate,crossover_rate , fitness_func):
        self.dim = dim
        self.popSize = popSize
        self.Iter = Iter
        self.lb = lb
        self.ub = ub
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.fitness_func = fitness_func

    def initialize_population(self):
        return np.random.uniform(self.lb, self.ub, size=(self.popSize, self.dim))

    def calculate_fitness(self, wolves):
        # fitness = np.zeros(self.popSize)
        fitness = np.zeros(len(wolves))

        for i, wolf in enumerate(wolves):
            fitness[i] = self.fitness_func(wolf)
        return fitness
    


    def mutate(self, wolve):
        min_val, max_val = self.lb, self.ub
        mutated_wolf = wolve.copy() 

        for j in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                mutated_wolf[j] += np.random.uniform(-1, 1)
             
                mutated_wolf[j] = np.clip(mutated_wolf[j], min_val, max_val)

        return mutated_wolf

    def crossover(self, wolf1, wolf2):
        offspring1 = np.zeros_like(wolf1)
        offspring2 = np.zeros_like(wolf2)

        # Chọn điểm cắt ngẫu nhiên
        crossover_point = np.random.randint(1, self.dim - 1)

        # Tạo con cái thứ nhất
        offspring1[:crossover_point] = wolf1[:crossover_point]
        offspring1[crossover_point:] = wolf2[crossover_point:]

        # Tạo con cái thứ hai
        offspring2[:crossover_point] = wolf2[:crossover_point]
        offspring2[crossover_point:] = wolf1[crossover_point:]

        return offspring1, offspring2
    

    def optimize(self):
        wolves = self.initialize_population()
        fitness_history = []

        for _ in range(self.Iter):
            wolves_list = []
            fitness = self.calculate_fitness(wolves)
            alpha_index = np.argmin(fitness)
            beta_index = np.argsort(fitness)[1]
            delta_index = np.argsort(fitness)[2]

            alpha, beta, delta = wolves[alpha_index], wolves[beta_index], wolves[delta_index]

            a = 2 - 2 * (_ / self.Iter)  # linearly decreased from 2 to 0

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
                wolves_list.extend([wolves_new])
                wolves_list.extend([wolves[i]])

                # Thực hiện đột biến và crossover
                if np.random.rand() < self.mutation_rate:
                    wolves_new = self.mutate(wolves_new)

                if np.random.rand() < self.crossover_rate:
                    # Lai ghép giữa wolves_new và alpha
                    child1, child2 = self.crossover(wolves_new, alpha)
                    wolves_list.extend([child1, child2])
                if np.random.rand() < self.crossover_rate:
                    # Lai ghép giữa wolves_new và beta
                    child3, child4 = self.crossover(wolves_new, beta)
                    wolves_list.extend([child3, child4])

                if np.random.rand() < self.crossover_rate:
                    # Lai ghép giữa wolves_new và delta
                    child5, child6 = self.crossover(wolves_new, delta)
                    wolves_list.extend([child5, child6])

            # fitness_list = self.calculate_fitness(wolves_list)
            # sorted_indices = np.argsort(fitness_list)[:self.popSize]
            # wolves = np.array(wolves_list)[sorted_indices]

            # fitness_history.append(np.argmin(fitness_list))
            fitness_list = self.calculate_fitness(wolves_list)
            sorted_indices = np.argsort(fitness_list)
            wolves_list = [wolves_list[i] for i in sorted_indices[:self.popSize]]
            wolves = np.array(wolves_list)

            fitness_history.append(np.min(fitness_list))
        return fitness_history
    

    # def optimize2(self):
    #     wolves = self.initialize_population()
    #     fitness_history = []
    #     # wolves_list = []
    #     # wolves_list.append(wolves)

    #     for _ in range(self.Iter):
    #         fitness = self.calculate_fitness(wolves)
    #         alpha_index = np.argmin(fitness)
    #         beta_index = np.argsort(fitness)[1]
    #         delta_index = np.argsort(fitness)[2]

    #         alpha, beta, delta = wolves[alpha_index], wolves[beta_index], wolves[delta_index]

    #         a = 2 - 2 * (_ / self.Iter)  # linearly decreased from 2 to 0

    #         new_wolves = np.zeros_like(wolves)

    #         for i in range(self.popSize):
    #             r1 = np.random.random(self.dim)
    #             r2 = np.random.random(self.dim)

    #             A1 = 2 * a * r1 - a
    #             C1 = 2 * r2

    #             D_alpha = abs(C1 * alpha - wolves[i])
    #             X1 = alpha - A1 * D_alpha

    #             r1 = np.random.random(self.dim)
    #             r2 = np.random.random(self.dim)

    #             A2 = 2 * a * r1 - a
    #             C2 = 2 * r2

    #             D_beta = abs(C2 * beta - wolves[i])
    #             X2 = beta - A2 * D_beta

    #             r1 = np.random.random(self.dim)
    #             r2 = np.random.random(self.dim)

    #             A3 = 2 * a * r1 - a
    #             C3 = 2 * r2

    #             D_delta = abs(C3 * delta - wolves[i])
    #             X3 = delta - A3 * D_delta

    #             wolves_new = (X1 + X2 + X3) / 3
    #             # wolves_list.append(wolves_new)

    #             # Thực hiện đột biến và crossover
    #             if np.random.rand() < self.mutation_rate:
    #                 wolves_new = self.mutate(wolves_new)
    #                 # wolves_list.append(wolves_new)


    #             if np.random.rand() < self.crossover_rate:
    #                 # Lai ghép giữa wolves_new và alpha
    #                 child1, child2 = self.crossover(wolves_new, alpha)
    #                 # Lai ghép giữa wolves_new và beta
    #                 child3, child4 = self.crossover(wolves_new, beta)
    #                 # Lai ghép giữa wolves_new và delta
    #                 child5, child6 = self.crossover(wolves_new, delta)



    #                 # Chọn con cái tốt nhất từ các cặp lai ghép
    #                 best_child1 = child1 if self.fitness_func(child1) < self.fitness_func(child2) else child2
    #                 best_child2 = child3 if self.fitness_func(child3) < self.fitness_func(child4) else child4
    #                 best_child3 = child5 if self.fitness_func(child5) < self.fitness_func(child6) else child6

    #                 # Chọn con cái tốt nhất từ tất cả các con cái
    #                 best_child = best_child1 if self.fitness_func(best_child1) < self.fitness_func(best_child2) else best_child2
    #                 best_child = best_child if self.fitness_func(best_child) < self.fitness_func(best_child3) else best_child3

    #                 wolves_new = best_child
                
    #             # Kiểm tra và cập nhật sói mới vào quần thể
    #             if self.fitness_func(wolves_new) < self.fitness_func(wolves[i]):
    #                 new_wolves[i] = wolves_new
    #             else:
    #                 new_wolves[i] = wolves[i]

    #         # Cập nhật quần thể
    #         # fitness = self.calculate_fitness(new_wolves)
    #         # # sorted_indices = np.argsort(fitness)
    #         # # sorted_indices = np.argsort(fitness)[:self.popSize]
    #         # # wolves = wolves[sorted_indices]
    #         # # wolves = wolves[sorted_indices[:self.popSize]]
    #         wolves = new_wolves

    #         fitness_history.append(np.min(fitness))
        # return fitness_history

    def optimize2(self):
            wolves = self.initialize_population()
            fitness_history = []

            for _ in range(self.Iter):
                fitness = self.calculate_fitness(wolves)
                sorted_indices = np.argsort(fitness)
                wolves = wolves[sorted_indices]

                alpha, beta, delta = wolves[:3]

                a = 2 - 2 * (_ / self.Iter)  # linearly decreased from 2 to 0

                new_wolves = np.zeros_like(wolves)

                for i, wolf in enumerate(wolves):
                    r1, r2, r3 = np.random.random((3, self.dim))

                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2

                    D_alpha = abs(C1 * alpha - wolf)
                    X1 = alpha - A1 * D_alpha

                    r1, r2, r3 = np.random.random((3, self.dim))

                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2

                    D_beta = abs(C2 * beta - wolf)
                    X2 = beta - A2 * D_beta

                    r1, r2, r3 = np.random.random((3, self.dim))

                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2

                    D_delta = abs(C3 * delta - wolf)
                    X3 = delta - A3 * D_delta

                    wolves_new = (X1 + X2 + X3) / 3

                    if np.random.rand() < self.mutation_rate:
                        wolves_new = self.mutate(wolves_new)

                    if np.random.rand() < self.crossover_rate:
                        child1, child2 = self.crossover(wolves_new, alpha)
                        child3, child4 = self.crossover(wolves_new, beta)
                        child5, child6 = self.crossover(wolves_new, delta)

                        best_child1 = child1 if self.fitness_func(child1) < self.fitness_func(child2) else child2
                        best_child2 = child3 if self.fitness_func(child3) < self.fitness_func(child4) else child4
                        best_child3 = child5 if self.fitness_func(child5) < self.fitness_func(child6) else child6

                        best_child = best_child1 if self.fitness_func(best_child1) < self.fitness_func(best_child2) else best_child2
                        best_child = best_child if self.fitness_func(best_child) < self.fitness_func(best_child3) else best_child3

                        wolves_new = best_child

                    new_wolves[i] = wolves_new if self.fitness_func(wolves_new) < fitness[i] else wolf

                fitness_history.append(np.min(fitness))
                wolves = new_wolves

            return fitness_history
    def optimize3(self):
        wolves = self.initialize_population()
        fitness_history = []
        array_fitness = []

        for iteration  in range(self.Iter):
            wolves_list = []
            fitness = self.calculate_fitness(wolves)
            sorted_indices = sorted(range(len(fitness)), key=lambda k: fitness[k])
            alpha_index, beta_index, delta_index = sorted_indices[:3]

            alpha, beta, delta = wolves[alpha_index], wolves[beta_index], wolves[delta_index]

            a = 2 - 2 * (iteration  / self.Iter)  # linearly decreased from 2 to 0

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
                wolves_list.append(wolves_new)
                wolves_list.append(wolves[i])

                # Thực hiện đột biến và crossover
                if np.random.rand() < self.mutation_rate:
                    wolves_new = self.mutate(wolves_new)

                if np.random.rand() < self.crossover_rate:
                    # Lai ghép giữa wolves_new và alpha
                    child1, child2 = self.crossover(wolves_new, alpha)
                    wolves_list.extend([child1, child2])
                if np.random.rand() < self.crossover_rate:
                    # Lai ghép giữa wolves_new và beta
                    child3, child4 = self.crossover(wolves_new, beta)
                    wolves_list.extend([child3, child4])

                if np.random.rand() < self.crossover_rate:
                    # Lai ghép giữa wolves_new và delta
                    child5, child6 = self.crossover(wolves_new, delta)
                    wolves_list.extend([child5, child6])

            fitness_list = self.calculate_fitness(wolves_list)
            sorted_indices = sorted(range(len(fitness_list)), key=lambda k: fitness_list[k])
            wolves_list = [wolves_list[i] for i in sorted_indices[:self.popSize]]
            wolves = np.array(wolves_list)

            fitness_history.append(np.min(fitness_list))
            if iteration % 100 == 0:
                array_fitness.append(np.min(fitness_history))

        return fitness_history, array_fitness

    def optimize4(self):
        wolves = self.initialize_population()
        fitness_history = []
        array_fitness = []

        for iteration  in range(self.Iter):
            wolves_list = []
            fitness = self.calculate_fitness(wolves)
            sorted_indices = sorted(range(len(fitness)), key=lambda k: fitness[k])
            alpha_index, beta_index, delta_index = sorted_indices[:3]

            alpha, beta, delta = wolves[alpha_index], wolves[beta_index], wolves[delta_index]

            a = 2 - 2 * (iteration  / self.Iter)  # linearly decreased from 2 to 0

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
                wolves_list.append(wolves_new)
                wolves_list.append(wolves[i])

                

                if np.random.rand() < self.crossover_rate:
                    # Lai ghép giữa wolves_new và alpha
                    child1, child2 = self.crossover(wolves_new, alpha)
                    wolves_list.extend([child1, child2])
                    # Thực hiện đột biến và crossover
                    if np.random.rand() < self.mutation_rate:
                        wolves_new = self.mutate(child1)
                        wolves_list.append(wolves_new)

                        wolves_new = self.mutate(child1)
                        wolves_list.append(wolves_new)



                if np.random.rand() < self.crossover_rate:
                    # Lai ghép giữa wolves_new và beta
                    child3, child4 = self.crossover(wolves_new, beta)
                    wolves_list.extend([child3, child4])
                    if np.random.rand() < self.mutation_rate:
                        wolves_new = self.mutate(child3)
                        wolves_list.append(wolves_new)
                        wolves_new = self.mutate(child4)
                        wolves_list.append(wolves_new)

                if np.random.rand() < self.crossover_rate:
                    # Lai ghép giữa wolves_new và delta
                    child5, child6 = self.crossover(wolves_new, delta)
                    wolves_list.extend([child5, child6])
                    if np.random.rand() < self.mutation_rate:
                        wolves_new = self.mutate(child5)
                        wolves_list.append(wolves_new)
                        wolves_new = self.mutate(child6)
                        wolves_list.append(wolves_new)

            fitness_list = self.calculate_fitness(wolves_list)
            sorted_indices = sorted(range(len(fitness_list)), key=lambda k: fitness_list[k])
                        # Select 50% best wolves
            best_wolves = sorted_indices[:self.Iter//2]
            # Select 50% random wolves from the remaining ones
            remaining_wolves = wolves_list[self.Iter//2:]
            random_wolves = random.sample(remaining_wolves, self.Iter//2)
            wolves_list = [wolves_list[i] for i in sorted_indices[:self.popSize]]
            wolves = best_wolves + random_wolves
            wolves = np.array(wolves)

            fitness_history.append(np.min(fitness_list))
            if iteration % 100 == 0:
                array_fitness.append(np.min(fitness_history))

        return fitness_history, array_fitness



