import numpy as np
import matplotlib.pyplot as plt

class ElectromagneticFieldOptimization:
    def __init__(self, N_var, N_emp, Max_gen, minval, maxval, R_rate, Ps_rate, P_field, N_field, problemIndex, fitness_func):
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

        # Khởi tạo quần thể ban đầu
        self.em_pop = self.initialize_population()
        self.best_fitness_history = []

        # Khởi tạo các chỉ số ngẫu nhiên
        self.r_index1 = np.random.randint(0, round(N_emp * P_field), (N_var, Max_gen))
        self.r_index2 = np.random.randint(round(N_emp * (1 - N_field)), N_emp, (N_var, Max_gen))
        self.r_index3 = np.random.randint(round(N_emp * P_field), round(N_emp * (1 - N_field)), (N_var, Max_gen))
        self.ps = np.random.rand(N_var, Max_gen)
        self.r_force = np.random.rand(Max_gen)
        self.rp = np.random.rand(Max_gen)
        self.randomization = np.random.rand(Max_gen)


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
        # Hàm cập nhật vị trí cho phần tử mới
    # def update_position(self, em_pop, N_var, generation, phi, Ps_rate, minval, maxval):
    #     r_index1 = np.random.randint(0, round(self.N_emp * self.P_field))  # Chọn chỉ số từ trường dương
    #     r_index2 = np.random.randint(round(self.N_emp * (1 - self.N_field)), self.N_emp)  # Chọn chỉ số từ trường âm
    #     r_index3 = np.random.randint(round(self.N_emp * self.P_field), round(self.N_emp * (1 - self.N_field)))  # Chọn chỉ số từ trường trung lập

    #     ps = np.random.rand()  # Xác suất cho việc chọn trường dương
    #     r = np.random.rand()  # Lực ngẫu nhiên
    #     randomization = np.random.rand()  # Ngẫu nhiên hóa cho giá trị nằm trong biên

    #     new_emp = np.zeros(N_var)

    #     for i in range(N_var):
    #         if ps > Ps_rate:
    #             new_emp[i] = (em_pop[r_index3, i] +
    #                           phi * r * (em_pop[r_index1, i] - em_pop[r_index3, i]) +
    #                           r * (em_pop[r_index3, i] - em_pop[r_index2, i]))
    #         else:
    #             new_emp[i] = em_pop[r_index1, i]

    #         # Kiểm tra xem giá trị có nằm trong biên không
    #         if new_emp[i] >= maxval or new_emp[i] <= minval:
    #             new_emp[i] = minval + (maxval - minval) * randomization

    #     return new_emp

        # Hàm cập nhật vị trí cho từng phần tử
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


    # def optimize(self):
    #     generation = 0
    #     RI = 0

    #     while generation < self.Max_gen:
    #         r = self.r_force[generation]
    #         new_emp = np.zeros(self.N_var + 1)

    #         # Cập nhật vị trí phần tử mới
    #         new_emp[:self.N_var] = self.update_position( self.em_pop,  self.N_var, generation,  self.phi,  self.Ps_rate,  self.minval,  self.maxval)
    #         # for i in range(self.N_var):
    #         #     if self.ps[i, generation] > self.Ps_rate:
    #         #         new_emp[i] = (self.em_pop[self.r_index3[i, generation], i] +
    #         #                       self.phi * r * (self.em_pop[self.r_index1[i, generation], i] - self.em_pop[self.r_index3[i, generation], i]) +
    #         #                       r * (self.em_pop[self.r_index3[i, generation], i] - self.em_pop[self.r_index2[i, generation], i]))
    #         #     else:
    #         #         new_emp[i] = self.em_pop[self.r_index1[i, generation], i]

    #         #     # Kiểm tra xem giá trị có nằm trong biên không
    #         #     if new_emp[i] >= self.maxval or new_emp[i] <= self.minval:
    #         #         new_emp[i] = self.minval + (self.maxval - self.minval) * self.randomization[generation]
    #         if np.random.rand() <  self.R_rate:
    #             new_emp[RI] =  self.minval + ( self.maxval -  self.minval) * np.random.rand()
    #             RI = (RI + 1) %  self.N_var

    #         # Thay thế một phần tử ngẫu nhiên trong số phần tử được sinh ra
    #         # if self.rp[generation] < self.R_rate:
    #         #     new_emp[RI] = self.minval + (self.maxval - self.minval) * self.randomization[generation]
    #         #     RI = (RI + 1) % self.N_var

    #         # Tính toán giá trị fitness cho phần tử mới
    #         new_emp[self.N_var] = self.fitness_function(new_emp[:self.N_var])

    #         # Cập nhật quần thể nếu phần tử mới có fitness tốt hơn phần tử tệ nhất
    #         if new_emp[self.N_var] < self.em_pop[-1, self.N_var]:
    #             position = np.where(self.em_pop[:, self.N_var] > new_emp[self.N_var])[0][0]
    #             self.em_pop = self.insert_in_population(self.em_pop, new_emp, position)

    #         # Lưu trữ fitness tốt nhất
    #         self.best_fitness_history.append(self.em_pop[0, self.N_var])

    #         generation += 1

    #     besterr = self.em_pop[0, self.N_var] - (self.problemIndex * 100)
    #     return besterr, self.best_fitness_history
    # def optimize(self):
    #     generation = 0
    #     RI = 0

    #     while generation < self.Max_gen:
    #         r = self.r_force[generation]
    #         new_emp = np.zeros(self.N_var + 1)

    #         # Cập nhật vị trí phần tử mới
    #         new_emp[:self.N_var] = self.update_position(self.em_pop, self.N_var, generation, self.phi, self.Ps_rate, self.minval, self.maxval)

    #         # Thay thế một phần tử ngẫu nhiên trong số phần tử được sinh ra
    #         if np.random.rand() < self.R_rate:
    #             new_emp[RI] = self.minval + (self.maxval - self.minval) * np.random.rand()
    #             RI = (RI + 1) % self.N_var

    #         # Tính toán giá trị fitness cho phần tử mới
    #         new_emp[self.N_var] = self.fitness_function(new_emp[:self.N_var])

    #         # Cập nhật quần thể nếu phần tử mới có fitness tốt hơn phần tử tệ nhất
    #         if new_emp[self.N_var] < self.em_pop[-1, self.N_var]:
    #             position = np.where(self.em_pop[:, self.N_var] > new_emp[self.N_var])[0][0]
    #             self.em_pop = self.insert_in_population(self.em_pop, new_emp, position)

    #         # Lưu trữ fitness tốt nhất
    #         self.best_fitness_history.append(self.em_pop[0, self.N_var])

    #         generation += 1

    #     besterr = self.em_pop[0, self.N_var] - (self.problemIndex * 100)
    #     return besterr, self.best_fitness_history
    def optimize(self):
        generation = 0
        RI = 0

        while generation < self.Max_gen:
            r = self.r_force[generation]
            new_emp = np.zeros(self.N_var + 1)

            # Cập nhật vị trí từng phần tử mới
            for i in range(self.N_var):
                new_emp[i] = self.update_position_for_element(self.em_pop, i, generation, self.phi, self.Ps_rate, self.minval, self.maxval)

            # Thay thế một phần tử ngẫu nhiên trong số phần tử được sinh ra
            if np.random.rand() < self.R_rate:
                new_emp[RI] = self.minval + (self.maxval - self.minval) * np.random.rand()
                RI = (RI + 1) % self.N_var

            # Tính toán giá trị fitness cho phần tử mới
            new_emp[self.N_var] = self.fitness_function(new_emp[:self.N_var])

            # Cập nhật quần thể nếu phần tử mới có fitness tốt hơn phần tử tệ nhất
            if new_emp[self.N_var] < self.em_pop[-1, self.N_var]:
                position = np.where(self.em_pop[:, self.N_var] > new_emp[self.N_var])[0][0]
                self.em_pop = self.insert_in_population(self.em_pop, new_emp, position)

            # Lưu trữ fitness tốt nhất
            self.best_fitness_history.append(self.em_pop[0, self.N_var])

            generation += 1

        besterr = self.em_pop[0, self.N_var] - (self.problemIndex * 100)
        return besterr, self.best_fitness_history

