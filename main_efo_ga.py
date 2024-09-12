import numpy as np
import matplotlib.pyplot as plt
from becnhmark import *
from genetic_algorithm import GeneticAlgorithm
from grey_wolf_optimizer import GWO
from grey_wolf_genetic_algorithm import GWO_GA
from efo import ElectromagneticFieldOptimization
from efo_ga import HybridGA_EFO

# Hàm để chạy thuật toán và lấy lịch sử fitness
def run_algorithm(algorithm, function_index, popSize, max_iter, mutation_rate, crossover_rate):
    function_name, lb, ub, dim = Get_Functions_details1(function_index)
    fitness_func = function_name

    if algorithm == "GA":
        ga = GeneticAlgorithm(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, mutation_rate=mutation_rate, crossover_rate=crossover_rate, fitness_func=fitness_func)
        fitness_history = ga.optimize()
    elif algorithm == "GWO":
        gwo = GWO(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, fitness_func=fitness_func)
        fitness_history = gwo.optimize()
    elif algorithm == "GWO-GA":
        gwo_ga = GWO_GA(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, mutation_rate=mutation_rate, crossover_rate=crossover_rate, fitness_func=fitness_func)
        fitness_history = gwo_ga.optimize()
    elif algorithm == "GWO-GA2":
        gwo_ga = GWO_GA(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, mutation_rate=mutation_rate, crossover_rate=crossover_rate, fitness_func=fitness_func)
        fitness_history = gwo_ga.optimize2()
    elif algorithm == "GWO-GA3":
        gwo_ga = GWO_GA(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, mutation_rate=mutation_rate, crossover_rate=crossover_rate, fitness_func=fitness_func)
        fitness_history = gwo_ga.optimize3()
    else:
        raise ValueError("Invalid algorithm name")

    return fitness_history

# Hàm để chạy thuật toán và lấy lịch sử fitness
def run_algorithm2(algorithm, function_index, popSize, max_iter, mutation_rate, crossover_rate):
    function_name, lb, ub, dim = Get_Functions_details3(function_index)
    fitness_func = function_name

    if algorithm == "GA":
        ga = GeneticAlgorithm(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, mutation_rate=mutation_rate, crossover_rate=crossover_rate, fitness_func=fitness_func)
        fitness_history = ga.optimize()
    elif algorithm == "GWO":
        gwo = GWO(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, fitness_func=fitness_func)
        fitness_history = gwo.optimize()
    elif algorithm == "GWO-GA":
        gwo_ga = GWO_GA(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, mutation_rate=mutation_rate, crossover_rate=crossover_rate, fitness_func=fitness_func)
        fitness_history = gwo_ga.optimize()
    elif algorithm == "GWO-GA2":
        gwo_ga = GWO_GA(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, mutation_rate=mutation_rate, crossover_rate=crossover_rate, fitness_func=fitness_func)
        fitness_history = gwo_ga.optimize2()
    elif algorithm == "GWO-GA3":
        gwo_ga = GWO_GA(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, mutation_rate=mutation_rate, crossover_rate=crossover_rate, fitness_func=fitness_func)
        fitness_history = gwo_ga.optimize3()
    else:
        raise ValueError("Invalid algorithm name")

    return fitness_history

# Hàm để chạy thuật toán và lấy lịch sử fitness
def run_algorithm3(algorithm, function_index, popSize, max_iter, mutation_rate, crossover_rate):
    function_name, lb, ub, dim = Get_Functions_details2(function_index)
    fitness_func = function_name
        # Khởi tạo GA

    # Khởi tạo EFO

    # Chạy GA và EFO


    if algorithm == "GA":
        # ga = GeneticAlgorithm(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, mutation_rate=mutation_rate, crossover_rate=crossover_rate, fitness_func=fitness_func)
        ga = GeneticAlgorithm(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, mutation_rate=0.1, crossover_rate=0.8, fitness_func=fitness_func)

        fitness_history , array_fitness = ga.optimize()
    elif algorithm == "EFO":
        # efo = EFO(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, fitness_func=fitness_func)
        # efo = EFO(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, R_rate=0.1, Ps_rate=0.7, P_field=0.3, N_field=0.2, fitness_func=sphere_function)
        efo = ElectromagneticFieldOptimization(N_var = dim, N_emp= popSize, Max_gen =max_iter, minval = lb, maxval=ub, R_rate= 0.1, Ps_rate= 0.7, P_field = 0.3, N_field = 0.2, problemIndex= 1, fitness_func=fitness_func)
        # efo = EFO(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, R_rate=0.1, Ps_rate=0.7, P_field=0.3, N_field=0.2, fitness_func=fitness_func)
        best, fitness_history = efo.optimize()
    elif algorithm == "EFO_GA":
        # efo = EFO(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, fitness_func=fitness_func)
        # efo = EFO(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, R_rate=0.1, Ps_rate=0.7, P_field=0.3, N_field=0.2, fitness_func=sphere_function)
        efo = HybridGA_EFO(N_var = dim, N_emp= popSize, Max_gen =max_iter, minval = lb, maxval=ub, R_rate= 0.1, Ps_rate= 0.7, P_field = 0.3, N_field = 0.2, problemIndex= 1, fitness_func=fitness_func, crossover_rate=0.8,mutation_rate=0.1)
        # efo = EFO(dim=dim, popSize=popSize, Iter=max_iter, lb=lb, ub=ub, R_rate=0.1, Ps_rate=0.7, P_field=0.3, N_field=0.2, fitness_func=fitness_func)
        best, fitness_history = efo.optimize()
    else:
        raise ValueError("Invalid algorithm name")

    return fitness_history

# Hàm để chạy các thuật toán và vẽ biểu đồ
def plot_comparison(algorithms, function_indices, max_iter):
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    for i, function_index in enumerate(function_indices):
        row = i // 3
        col = i % 3
        ax = axs[row, col]
        print("function: ", function_index);
        for algorithm in algorithms:
            fitness_history = run_algorithm(algorithm, function_index, popSize, max_iter, mutation_rate, crossover_rate)
            ax.plot(fitness_history, label=algorithm)
        # ax.set_title(f"F{function_index+1}", fontsize=10)
        ax.set_title(f"{function_indices[i]}", fontsize=10)
        ax.set_xlabel("Iterations", fontsize=8)
        ax.set_ylabel("Fitness", fontsize=8)
        ax.set_yscale('log')
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.show()

# Hàm để chạy các thuật toán và vẽ biểu đồ
def plot_comparison2(algorithms, function_indices, max_iter):
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    for i, function_index in enumerate(function_indices):
        row = i // 4
        col = i % 4
        ax = axs[row, col]
        print("function: ", function_index);
        for algorithm in algorithms:
            fitness_history = run_algorithm2(algorithm, function_index, popSize, max_iter, mutation_rate, crossover_rate)
            ax.plot(fitness_history, label=algorithm)
        # ax.set_title(f"F{function_index+1}", fontsize=10)
        ax.set_title(f"{function_indices[i]}", fontsize=10)
        ax.set_xlabel("Iterations", fontsize=8)
        ax.set_ylabel("Fitness", fontsize=8)
        ax.set_yscale('log')
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.show()

# Hàm để chạy các thuật toán và vẽ biểu đồ
def plot_comparison3(algorithms, function_indices, max_iter):
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    for i, function_index in enumerate(function_indices):
        row = i // 3
        col = i % 3
        ax = axs[row, col]
        print("function: ", function_index);
        for algorithm in algorithms:
            fitness_history = run_algorithm3(algorithm, function_index, popSize, max_iter, mutation_rate, crossover_rate)
            ax.plot(fitness_history, label=algorithm)
        # ax.set_title(f"F{function_index+1}", fontsize=10)
        ax.set_title(f"{function_indices[i]}", fontsize=10)
        ax.set_xlabel("Iterations", fontsize=8)
        ax.set_ylabel("Fitness", fontsize=8)
        ax.set_yscale('log')
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.show()
# Chạy và vẽ biểu đồ so sánh
algorithms = ["GA", "EFO", "EFO_GA"]
function_indices = [ 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F24',  'F10', 'F11', 'F12', 'F13']  # Chọn các hàm thử nghiệm
# function_indices = [ 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F24',  'F10', 'F11', 'F12', 'F13']  # Chọn các hàm thử nghiệm
# function_indices = [ 'F24']  # Chọn các hàm thử nghiệm
function_indices = [ 'F1', 'F2', 'F3', 'F4']  # Chọn các hàm thử nghiệm


max_iter = 500 # Số lượng lặp tối đa

# popSize = 50
# mutation_rate = 0.1
# crossover_rate = 0.7
# plot_comparison(algorithms, function_indices, max_iter)


# popSize = 50
# mutation_rate = 0.1
# crossover_rate = 0.7
# plot_comparison2(algorithms, function_indices, max_iter)

popSize = 50
mutation_rate = 0.1
crossover_rate = 0.7
plot_comparison3(algorithms, function_indices, max_iter)


# import numpy as np
# import matplotlib.pyplot as plt
# from becnhmark import *
# from genetic_algorithm import GeneticAlgorithm
# from grey_wolf_optimizer import GWO
# from grey_wolf_genetic_algorithm import GWO_GA
# from efo import ElectromagneticFieldOptimization


# def sphere_function(x):
#     return np.sum(x**2)

# # Thông số chung
# dim = 5
# popSize = 50
# Iter = 100
# lb = -5.0
# ub = 5.0

# # Khởi tạo GA
# ga = GeneticAlgorithm(dim=dim, popSize=popSize, Iter=Iter, lb=lb, ub=ub, mutation_rate=0.1, crossover_rate=0.8, fitness_func=sphere_function)

# # Khởi tạo EFO
# efo = ElectromagneticFieldOptimization(N_var = dim, N_emp= popSize, Max_gen =Iter, minval = lb, maxval=ub, R_rate= 0.1, Ps_rate= 0.7, P_field = 0.3, N_field = 0.2, problemIndex= 1, fitness_func=sphere_function)

# # Chạy GA và EFO
# ga_fitness_history, ga_array_fitness = ga.optimize()
# best , efo_fitness_history = efo.optimize()

# # Vẽ biểu đồ so sánh
# plt.figure(figsize=(10, 6))
# plt.plot(ga_fitness_history, label='GA - Best Fitness', color='green')
# plt.plot(efo_fitness_history, label='EFO - Best Fitness', color='blue')
# plt.title('Comparison of GA and EFO - Best Fitness Over Generations')
# plt.xlabel('Generation')
# plt.ylabel('Best Fitness')
# plt.yscale('log')
# plt.legend()
# plt.grid(True)
# plt.show()
