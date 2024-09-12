# import numpy as np
# import matplotlib.pyplot as plt

# class HybridGA_EFO:
#     def __init__(self, N_var, N_emp, Max_gen, minval, maxval, R_rate, Ps_rate, P_field, N_field, problemIndex, fitness_func, crossover_rate=0.8, mutation_rate=0.1):
#         self.N_var = N_var
#         self.N_emp = N_emp
#         self.Max_gen = Max_gen
#         self.minval = minval
#         self.maxval = maxval
#         self.R_rate = R_rate
#         self.Ps_rate = Ps_rate
#         self.P_field = P_field
#         self.N_field = N_field
#         self.problemIndex = problemIndex
#         self.fitness_function = fitness_func
#         self.phi = (1 + np.sqrt(5)) / 2  # Tỉ lệ vàng

#         # Tham số GA
#         self.crossover_rate = crossover_rate
#         self.mutation_rate = mutation_rate

#         # Khởi tạo quần thể ban đầu
#         self.ga_pop = self.initialize_population()
#         self.efo_pop = None
#         self.best_fitness_history = []

#     def initialize_population(self):
#         pop = self.minval + (self.maxval - self.minval) * np.random.rand(self.N_emp, self.N_var)
#         fit = np.array([self.fitness_function(pop[i, :]) for i in range(self.N_emp)])
#         pop = np.hstack((pop, fit.reshape(-1, 1)))
#         return self.sort_population(pop)

#     def sort_population(self, pop):
#         return pop[pop[:, self.N_var].argsort()]

#     def insert_in_population(self, pop, new_ind, position):
#         pop = np.insert(pop, position, new_ind, axis=0)
#         return np.delete(pop, -1, axis=0)

#     def crossover(self, parent1, parent2):
#         # Thực hiện crossover giữa hai cá thể
#         child = np.copy(parent1)
#         if np.random.rand() < self.crossover_rate:
#             crossover_point = np.random.randint(1, self.N_var - 1)
#             child[:crossover_point] = parent2[:crossover_point]
#         return child

#     def mutate(self, individual):
#         # Thực hiện đột biến
#         for i in range(self.N_var):
#             if np.random.rand() < self.mutation_rate:
#                 individual[i] = self.minval + (self.maxval - self.minval) * np.random.rand()
#         return individual

#     def GA_optimize(self):
#         # Giai đoạn tối ưu hóa bằng GA
#         for generation in range(self.Max_gen // 2):  # Chạy GA trong một nửa số vòng lặp
#             # Chọn hai cá thể tốt nhất
#             parent1 = self.ga_pop[0, :self.N_var]
#             parent2 = self.ga_pop[1, :self.N_var]

#             # Lai tạo và đột biến
#             child = self.crossover(parent1, parent2)
#             child = self.mutate(child)

#             # Tính toán fitness của cá thể con
#             child_fitness = self.fitness_function(child)

#             # Cập nhật quần thể nếu con có fitness tốt hơn
#             if child_fitness < self.ga_pop[-1, self.N_var]:
#                 self.ga_pop = self.insert_in_population(self.ga_pop, np.hstack((child, child_fitness)), -1)

#             # Lưu lại lịch sử fitness tốt nhất
#             self.best_fitness_history.append(self.ga_pop[0, self.N_var])

#         # Kết thúc GA, lưu quần thể
#         self.ga_pop = self.sort_population(self.ga_pop)

#     def EFO_optimize(self):
#         # Giai đoạn tối ưu hóa bằng EFO, sử dụng kết quả từ GA
#         self.efo_pop = self.ga_pop.copy()  # Sử dụng kết quả từ GA

#         # Khởi tạo các chỉ số ngẫu nhiên cho EFO
#         self.r_index1 = np.random.randint(0, round(self.N_emp * self.P_field), (self.N_var, self.Max_gen))
#         self.r_index2 = np.random.randint(round(self.N_emp * (1 - self.N_field)), self.N_emp, (self.N_var, self.Max_gen))
#         self.r_index3 = np.random.randint(round(self.N_emp * self.P_field), round(self.N_emp * (1 - self.N_field)), (self.N_var, self.Max_gen))
#         self.ps = np.random.rand(self.N_var, self.Max_gen)
#         self.r_force = np.random.rand(self.Max_gen)
#         self.rp = np.random.rand(self.Max_gen)
#         self.randomization = np.random.rand(self.Max_gen)

#         generation = 0
#         RI = 0

#         while generation < self.Max_gen // 2:  # Chạy EFO trong nửa số vòng lặp còn lại
#             r = self.r_force[generation]
#             new_emp = np.zeros(self.N_var + 1)

#             for i in range(self.N_var):
#                 if self.ps[i, generation] > self.Ps_rate:
#                     new_emp[i] = (self.efo_pop[self.r_index3[i, generation], i] +
#                                   self.phi * r * (self.efo_pop[self.r_index1[i, generation], i] - self.efo_pop[self.r_index3[i, generation], i]) +
#                                   r * (self.efo_pop[self.r_index3[i, generation], i] - self.efo_pop[self.r_index2[i, generation], i]))
#                 else:
#                     new_emp[i] = self.efo_pop[self.r_index1[i, generation], i]

#                 # Kiểm tra giới hạn biên
#                 if new_emp[i] >= self.maxval or new_emp[i] <= self.minval:
#                     new_emp[i] = self.minval + (self.maxval - self.minval) * self.randomization[generation]

#             if self.rp[generation] < self.R_rate:
#                 new_emp[RI] = self.minval + (self.maxval - self.minval) * self.randomization[generation]
#                 RI = (RI + 1) % self.N_var

#             new_emp[self.N_var] = self.fitness_function(new_emp[:self.N_var])

#             # Cập nhật quần thể nếu phần tử mới tốt hơn
#             if new_emp[self.N_var] < self.efo_pop[-1, self.N_var]:
#                 position = np.where(self.efo_pop[:, self.N_var] > new_emp[self.N_var])[0][0]
#                 self.efo_pop = self.insert_in_population(self.efo_pop, new_emp, position)

#             self.best_fitness_history.append(self.efo_pop[0, self.N_var])

#             generation += 1

#         # Quần thể sau EFO
#         self.efo_pop = self.sort_population(self.efo_pop)

#     def optimize(self):
#         # Chạy tối ưu hóa bằng GA trước
#         self.GA_optimize()

#         # Sau đó dùng EFO để bổ trợ tìm kiếm
#         self.EFO_optimize()

#         # Trả về cá thể tốt nhất sau cả GA và EFO
#         best_individual = self.efo_pop[0, :self.N_var]
#         best_fitness = self.efo_pop[0, self.N_var]

#         besterr = best_fitness - (self.problemIndex * 100)
#         return besterr, self.best_fitness_history
# import numpy as np
# import matplotlib.pyplot as plt

# class HybridGA_EFO:
#     def __init__(self, N_var, N_emp, Max_gen, minval, maxval, R_rate, Ps_rate, P_field, N_field, problemIndex, fitness_func, crossover_rate=0.8, mutation_rate=0.1):
#         self.N_var = N_var
#         self.N_emp = N_emp
#         self.Max_gen = Max_gen
#         self.minval = minval
#         self.maxval = maxval
#         self.R_rate = R_rate
#         self.Ps_rate = Ps_rate
#         self.P_field = P_field
#         self.N_field = N_field
#         self.problemIndex = problemIndex
#         self.fitness_function = fitness_func
#         self.phi = (1 + np.sqrt(5)) / 2  # Tỉ lệ vàng

#         # Tham số GA
#         self.crossover_rate = crossover_rate
#         self.mutation_rate = mutation_rate

#         # Khởi tạo quần thể ban đầu
#         self.ga_pop = self.initialize_population()
#         self.efo_pop = None
#         self.best_fitness_history = []

#         self.ps = np.random.rand(N_var, Max_gen)
#         self.r_force = np.random.rand(Max_gen)
#         self.rp = np.random.rand(Max_gen)
#         self.randomization = np.random.rand(Max_gen)

#     def calculate_fitness(self, population):
#         return np.array([self.fitness_function(individual) for individual in population])

#     def initialize_population(self):
#         # Khởi tạo quần thể ngẫu nhiên
#         em_pop = self.minval + (self.maxval - self.minval) * np.random.rand(self.N_emp, self.N_var)
#         fit = np.array([self.fitness_function(em_pop[i, :]) for i in range(self.N_emp)])
#         em_pop = np.hstack((em_pop, fit.reshape(-1, 1)))
#         return self.sort_population(em_pop)

#     def sort_population(self, em_pop):
#         return em_pop[em_pop[:, self.N_var].argsort()]

#     def insert_in_population(self, em_pop, new_emp, position):
#         em_pop = np.insert(em_pop, position, new_emp, axis=0)
#         return np.delete(em_pop, -1, axis=0)

#     def selection(self, population, fitness_scores):
#             # Sort the population based on fitness scores in ascending order
#             sorted_indices = np.argsort(fitness_scores)
#             sorted_population = population[sorted_indices]
#             return sorted_population[:self.N_var]

#     def crossover(self, parents):
#         offspring = np.empty((self.N_emp, self.N_var))
#         for i in range(0, self.N_emp, 2):
#             parent1_idx = i % len(parents)
#             parent2_idx = (i + 1) % len(parents)
#             parent1 = parents[parent1_idx]
#             parent2 = parents[parent2_idx]

#             crossover_point = np.random.randint(1, self.N_emp - 1) if self.N_emp >= 3 else 1

#             offspring[i, 0:crossover_point] = parent1[0:crossover_point]
#             offspring[i, crossover_point:] = parent2[crossover_point:]
#             offspring[i + 1, 0:crossover_point] = parent2[0:crossover_point]
#             offspring[i + 1, crossover_point:] = parent1[crossover_point:]

#         return offspring


#         # Hàm cập nhật vị trí cho từng phần tử
#     def update_position_for_element(self, em_pop, i, generation, phi, Ps_rate, minval, maxval):
#         r_index1 = np.random.randint(0, round(self.N_emp * self.P_field))  # Chọn chỉ số từ trường dương
#         r_index2 = np.random.randint(round(self.N_emp * (1 - self.N_field)), self.N_emp)  # Chọn chỉ số từ trường âm
#         r_index3 = np.random.randint(round(self.N_emp * self.P_field), round(self.N_emp * (1 - self.N_field)))  # Chọn chỉ số từ trường trung lập

#         ps = np.random.rand()  # Xác suất cho việc chọn trường dương
#         r = np.random.rand()  # Lực ngẫu nhiên
#         randomization = np.random.rand()  # Ngẫu nhiên hóa cho giá trị nằm trong biên

#         # Cập nhật vị trí của một phần tử
#         if ps > Ps_rate:
#             new_position = (em_pop[r_index3, i] +
#                             phi * r * (em_pop[r_index1, i] - em_pop[r_index3, i]) +
#                             r * (em_pop[r_index3, i] - em_pop[r_index2, i]))
#         else:
#             new_position = em_pop[r_index1, i]

#         # Kiểm tra xem giá trị có nằm trong biên không
#         if new_position >= maxval or new_position <= minval:
#             new_position = minval + (maxval - minval) * randomization

#         return new_position


#     def optimize(self):
#         generation = 0
#         RI = 0

#         while generation < self.Max_gen:
#             r = self.r_force[generation]
#             new_emp = np.zeros(self.N_var + 1)
#             fitness_scores = self.calculate_fitness(new_emp)

#             parents = self.selection(new_emp, fitness_scores)
#             new_emp = self.crossover(parents)

#             # Cập nhật vị trí từng phần tử mới
#             for i in range(self.N_var):
#                 new_emp[i] = self.update_position_for_element(self.em_pop, i, generation, self.phi, self.Ps_rate, self.minval, self.maxval)

#             # Thay thế một phần tử ngẫu nhiên trong số phần tử được sinh ra
#             if np.random.rand() < self.R_rate:
#                 new_emp[RI] = self.minval + (self.maxval - self.minval) * np.random.rand()
#                 RI = (RI + 1) % self.N_var

#             # Tính toán giá trị fitness cho phần tử mới
#             new_emp[self.N_var] = self.fitness_function(new_emp[:self.N_var])

#             # Cập nhật quần thể nếu phần tử mới có fitness tốt hơn phần tử tệ nhất
#             if new_emp[self.N_var] < self.em_pop[-1, self.N_var]:
#                 position = np.where(self.em_pop[:, self.N_var] > new_emp[self.N_var])[0][0]
#                 self.em_pop = self.insert_in_population(self.em_pop, new_emp, position)

#             # Lưu trữ fitness tốt nhất
#             self.best_fitness_history.append(self.em_pop[0, self.N_var])

#             generation += 1

#         besterr = self.em_pop[0, self.N_var] - (self.problemIndex * 100)
#         return besterr, self.best_fitness_history


import numpy as np
import matplotlib.pyplot as plt

class HybridGA_EFO:
    def __init__(self, N_var, N_emp, Max_gen, minval, maxval, R_rate, Ps_rate, P_field, N_field, problemIndex, fitness_func, crossover_rate=0.8, mutation_rate=0.1):
        self.N_var = N_var
        self.N_emp = N_emp
        self.Max_gen = Max_gen
        self.minval = minval
        self.maxval = maxval
        self.R_rate = R_rate
        self.Ps_rate = Ps_rate
        self.P_field = P_field
        self.N_field = N_field
        self.problemIndex = problemIndex
        self.fitness_function = fitness_func
        self.phi = (1 + np.sqrt(5)) / 2  # Tỉ lệ vàng

        # Tham số GA
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        # Khởi tạo quần thể ban đầu
        self.ga_pop = self.initialize_population()
        self.efo_pop = None
        self.best_fitness_history = []

        self.ps = np.random.rand(N_var, Max_gen)
        self.r_force = np.random.rand(Max_gen)
        self.rp = np.random.rand(Max_gen)
        self.randomization = np.random.rand(Max_gen)

    def calculate_fitness(self, population):
        return np.array([self.fitness_function(individual) for individual in population])

    def initialize_population(self):
        # Khởi tạo quần thể ngẫu nhiên
        em_pop = self.minval + (self.maxval - self.minval) * np.random.rand(self.N_emp, self.N_var)
        fit = np.array([self.fitness_function(em_pop[i, :]) for i in range(self.N_emp)])
        em_pop = np.hstack((em_pop, fit.reshape(-1, 1)))
        return self.sort_population(em_pop)

    def sort_population(self, em_pop):
        return em_pop[em_pop[:, self.N_var].argsort()]

    def insert_in_population(self, em_pop, new_emp, position):
        em_pop = np.insert(em_pop, position, new_emp, axis=0)
        return np.delete(em_pop, -1, axis=0)

    def selection(self, population, fitness_scores):
        # Chọn những cá thể tốt nhất
        sorted_indices = np.argsort(fitness_scores)
        return population[sorted_indices[:len(population)//2]]  # Chọn một nửa quần thể tốt nhất

    def crossover(self, parents):
        offspring = np.empty((self.N_emp, self.N_var))
        num_parents = len(parents)
        for i in range(0, num_parents - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            crossover_point = np.random.randint(1, self.N_var - 1)

            offspring[i, 0:crossover_point] = parent1[0:crossover_point]
            offspring[i, crossover_point:] = parent2[crossover_point:]
            offspring[i + 1, 0:crossover_point] = parent2[0:crossover_point]
            offspring[i + 1, crossover_point:] = parent1[crossover_point:]

        return offspring

    def update_position_for_element(self, em_pop, i, generation, phi, Ps_rate, minval, maxval):
        r_index1 = np.random.randint(0, round(self.N_emp * self.P_field))  # Chọn chỉ số từ trường dương
        r_index2 = np.random.randint(round(self.N_emp * (1 - self.N_field)), self.N_emp)  # Chọn chỉ số từ trường âm
        r_index3 = np.random.randint(round(self.N_emp * self.P_field), round(self.N_emp * (1 - self.N_field)))  # Chọn chỉ số từ trường trung lập

        ps = np.random.rand()  # Xác suất cho việc chọn trường dương
        r = np.random.rand()  # Lực ngẫu nhiên
        randomization = np.random.rand()  # Ngẫu nhiên hóa cho giá trị nằm trong biên

        # Cập nhật vị trí của một phần tử
        if ps > Ps_rate:
            new_position = (em_pop[r_index3, i] +
                            phi * r * (em_pop[r_index1, i] - em_pop[r_index3, i]) +
                            r * (em_pop[r_index3, i] - em_pop[r_index2, i]))
        else:
            new_position = em_pop[r_index1, i]

        # Kiểm tra xem giá trị có nằm trong biên không
        if new_position >= maxval or new_position <= minval:
            new_position = minval + (maxval - minval) * randomization

        return new_position

    def optimize(self):
        generation = 0
        RI = 0

        while generation < self.Max_gen:
            # GA process
            fitness_scores = self.calculate_fitness(self.ga_pop[:, :self.N_var])
            parents = self.selection(self.ga_pop[:, :self.N_var], fitness_scores)
            offspring = self.crossover(parents)

            # EFO process (update positions)
            for i in range(self.N_var):
                offspring[:, i] = self.update_position_for_element(self.ga_pop, i, generation, self.phi, self.Ps_rate, self.minval, self.maxval)

            # Cập nhật random cho một số cá thể
            if np.random.rand() < self.R_rate:
                offspring[RI] = self.minval + (self.maxval - self.minval) * np.random.rand(self.N_var)
                RI = (RI + 1) % self.N_var

            # Tính toán fitness cho quần thể mới
            offspring_fitness = self.calculate_fitness(offspring)
            offspring = np.hstack((offspring, offspring_fitness.reshape(-1, 1)))

            # Cập nhật quần thể chính với các cá thể mới
            combined_population = np.vstack((self.ga_pop, offspring))
            combined_population = self.sort_population(combined_population)
            self.ga_pop = combined_population[:self.N_emp]  # Chọn N_emp cá thể tốt nhất

            # Lưu trữ fitness tốt nhất
            self.best_fitness_history.append(self.ga_pop[0, self.N_var])

            generation += 1

        besterr = self.ga_pop[0, self.N_var] - (self.problemIndex * 100)
        return besterr, self.best_fitness_history
