import numpy as np
import matplotlib.pyplot as plt
from becnhmark import *
from genetic_algorithm import GeneticAlgorithm
from grey_wolf_optimizer import GWO
from grey_wolf_genetic_algorithm import GWO_GA

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

# Hàm để chạy các thuật toán và vẽ biểu đồ
def plot_comparison(algorithms, function_indices, max_iter):
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    for i, function_index in enumerate(function_indices):
        row = i // 4
        col = i % 4
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
        row = i // 4
        col = i % 4
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
algorithms = ["GA", "GWO", "GWO-GA","GWO-GA2","GWO-GA3"]
# function_indices = [ 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F9', 'F10', 'F11', 'F12', 'F13', 'F24', 'F15',    'F20', 'F21', 'F22', 'F23' ]  # Chọn các hàm thử nghiệm
# function_indices = [ 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F9', 'F10', 'F11', 'F12', 'F13', 'F24', 'F15']  # Chọn các hàm thử nghiệm
function_indices = [ 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F24',  'F10', 'F11', 'F12', 'F13']  # Chọn các hàm thử nghiệm
function_indices = [ 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F24',  'F10', 'F11', 'F12', 'F13']  # Chọn các hàm thử nghiệm
# function_indices = [ 'F24']  # Chọn các hàm thử nghiệm

max_iter = 500 # Số lượng lặp tối đa

popSize = 50
mutation_rate = 0.1
crossover_rate = 0.7
plot_comparison(algorithms, function_indices, max_iter)


popSize = 50
mutation_rate = 0.1
crossover_rate = 0.7
plot_comparison2(algorithms, function_indices, max_iter)

popSize = 50
mutation_rate = 0.1
crossover_rate = 0.7
plot_comparison3(algorithms, function_indices, max_iter)