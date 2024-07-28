import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from plotting.error_calculation import MAE_calculate, MSE_calculate, RMSE_calculate, MRE_calculate, MAPE_calculate

# plot style
plt.rcParams.update({'font.size': 14})
solver_name_dict = {
    0: 'Exact Riemann',
    1: "Lax Friedrichs",
    2: 'Roe',
    3: 'Osher',
    4: 'Exact Riemann Appr',
}
solver_color_dict = {
    0: 'grey',
    1: 'orange',
    2: "blue",
    3: 'green',
    4: 'magenta'
}
columns_names = ["Density", "Velocity", "Pressure", "Energy"]


def plot_solution_validation(problem_type, solver, n_cells, output_time):
    # Read the .out file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '..', f"output/{problem_type}", f'solver{solver}_t{output_time:.3f}_n{n_cells}.out')
    data = pd.read_csv(file_path, sep='\s+', header=None)

    # read the json analytic solution file
    jsonfile_path = os.path.join(script_dir, 'analytic_solutions.json')
    with open(jsonfile_path, 'r') as file:
        analytic_solution = json.load(file)

    columns = np.zeros((data.shape[1],data.shape[0]))
    for i in range(5):
        columns[i] = data[i].to_numpy()

    # Figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    for i in range(4):
        # the indexing for axes is to transform it into 2d subplotting
        # Plot the analytic solution
        axes[i//2, i%2].plot(analytic_solution[f"{problem_type}"][columns_names[i]]["x"], analytic_solution[f"{problem_type}"][columns_names[i]]["y"], label=f"Analytic", color="black")
        # Plot numerical solution
        axes[i//2, i%2].plot(columns[0], columns[i+1], label=f"{solver_name_dict[solver]}", marker='.', color=solver_color_dict[solver], linewidth=0)
        axes[i//2, i%2].set_xlabel('x')
        axes[i//2, i%2].set_ylabel(f'{columns_names[i]}')
        axes[i//2, i%2].grid(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_convergence_spatial(problem_type, solver, n_cells_list, output_time):
    # read the json analytic solution file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jsonfile_path = os.path.join(script_dir, 'analytic_solutions.json')
    with open(jsonfile_path, 'r') as file:
        analytic_solution = json.load(file)
    # save analytic in list
    x_analytic = []
    y_analytic = []
    for i in range(4):
        x_analytic.append(analytic_solution[f"{problem_type}"][columns_names[i]]["x"])
        y_analytic.append(analytic_solution[f"{problem_type}"][columns_names[i]]["y"])

    # calculate the error for each n_cell
    errors = np.zeros((4,len(n_cells_list)))
    for j, n_cells in enumerate(n_cells_list):
        # Read the .out file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, '..', f"output/{problem_type}", f'solver{solver}_t{output_time:.3f}_n{n_cells}.out')
        data = pd.read_csv(file_path, sep='\s+', header=None)
        columns_solver = np.zeros((data.shape[1],data.shape[0]))
        for i in range(5):
            columns_solver[i] = data[i].to_numpy()

        #interpolate the analytic data to fit the cell number
        for i in range(4): # each info type
            x_old = np.linspace(0, 1, num=len(x_analytic[i]))
            y_old = np.array(y_analytic[i])
            x_new = np.linspace(0,1, n_cells)
            analytic_interp = np.interp(x_new, x_old, y_old)
            error_absolute = np.abs(analytic_interp[i] - columns_solver[i+1])
            error_relative = error_absolute/np.abs(analytic_interp[i] + 1e-10) # calculate error relative for whole array
            errors[i, j] = MSE_calculate(error_absolute)

    # Figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    for i in range(4):
        # the indexing for axes is to transform it into 2d subplotting
        # Plot numerical solution
        axes[i//2, i%2].plot(n_cells_list, errors[i], label=f"{solver_name_dict[solver]}", marker='.', color=solver_color_dict[solver], linewidth=2)
        axes[i//2, i%2].set_xlabel('n cells')
        axes[i//2, i%2].set_ylabel(f'{columns_names[i]} MSE error')
        axes[i//2, i%2].set_yscale('log')
        axes[i//2, i%2].grid(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_convergence_spatial_comparison(problem_type, solver_list, n_cells_list, output_time):
    # read the json analytic solution file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jsonfile_path = os.path.join(script_dir, 'analytic_solutions.json')
    with open(jsonfile_path, 'r') as file:
        analytic_solution = json.load(file)
    # save analytic in list
    x_analytic = []
    y_analytic = []
    for i in range(4):
        x_analytic.append(analytic_solution[f"{problem_type}"][columns_names[i]]["x"])
        y_analytic.append(analytic_solution[f"{problem_type}"][columns_names[i]]["y"])

    # Figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    for solver in solver_list:
        # calculate the error for each n_cell
        errors = np.zeros((4,len(n_cells_list)))
        for j, n_cells in enumerate(n_cells_list):
            # Read the .out file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, '..', f"output/{problem_type}", f'solver{solver}_t{output_time:.3f}_n{n_cells}.out')
            data = pd.read_csv(file_path, sep='\s+', header=None)
            columns_solver = np.zeros((data.shape[1],data.shape[0]))
            for i in range(5):
                columns_solver[i] = data[i].to_numpy()

            #interpolate the analytic data to fit the cell number
            for i in range(4): # each info type
                x_old = np.linspace(0, 1, num=len(x_analytic[i]))
                y_old = np.array(y_analytic[i])
                x_new = np.linspace(0,1, n_cells)
                analytic_interp = np.interp(x_new, x_old, y_old)
                error_absolute = np.abs(analytic_interp[i] - columns_solver[i+1])
                error_relative = error_absolute/np.abs(analytic_interp[i] + 1e-10) # calculate error relative for whole array
                errors[i, j] = MSE_calculate(error_absolute)


        for i in range(4):
            # the indexing for axes is to transform it into 2d subplotting
            # Plot numerical solution
            axes[i//2, i%2].plot(n_cells_list, errors[i], label=f"{solver_name_dict[solver]}", marker='.', color=solver_color_dict[solver], linewidth=2)
            axes[i//2, i%2].set_xlabel('n cells')
            axes[i//2, i%2].set_ylabel(f'{columns_names[i]} MSE error')
            axes[i//2, i%2].set_yscale('log')
            axes[i//2, i%2].grid(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def plot_solution_error(problem_type, solver_list, n_cells, output_time):
    # Read the .out file of the exact solution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '..', f"output/{problem_type}", f'solver0_t{output_time:.3f}_n{n_cells}.out')
    data = pd.read_csv(file_path, sep='\s+', header=None)
    columns_exact = np.zeros((data.shape[1],data.shape[0]))
    for i in range(5):
        columns_exact[i] = data[i].to_numpy()

    # Figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    for i, solver in enumerate(solver_list):
        # Read the .out file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, '..', f"output/{problem_type}", f'solver{solver}_t{output_time:.3f}_n{n_cells}.out')
        data = pd.read_csv(file_path, sep='\s+', header=None)
        columns_solver = np.zeros((data.shape[1],data.shape[0]))
        for i in range(5):
            columns_solver[i] = data[i].to_numpy()

        # calculate error
        error = np.zeros((data.shape[1],data.shape[0]))
        for i in range(4):
            error[i+1] = np.abs(columns_exact[i+1] - columns_solver[i+1])/np.abs(columns_exact[i+1] + 1e-10)

        for i in range(4):
            # the indexing for axes is to transform it into 2d subplotting
            # Plot numerical solution
            axes[i//2, i%2].plot(columns_solver[0], error[i+1], label=f"{solver_name_dict[solver]}", color=solver_color_dict[solver])
            axes[i//2, i%2].set_xlabel('x')
            axes[i//2, i%2].set_yscale('log')
            axes[i//2, i%2].set_ylabel(f'{columns_names[i]}')
            axes[i//2, i%2].grid(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()



def plot_solutions_compare(problem_type, solver_list, n_cells, output_time):
    # Get current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read the json analytic solution file
    jsonfile_path = os.path.join(script_dir, 'analytic_solutions.json')
    with open(jsonfile_path, 'r') as file:
        analytic_solution = json.load(file)

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))

    for j, solver in enumerate(solver_list):
        # Read the .out file
        file_path = os.path.join(script_dir, '..', f"output/{problem_type}", f'solver{solver}_t{output_time:.3f}_n{n_cells}.out')
        data = pd.read_csv(file_path, sep='\s+', header=None)
        
        # Access columns
        columns = np.zeros((data.shape[1],data.shape[0]))
        for i in range(5):
            columns[i] = data[i].to_numpy()
        columns_names = ["Density", "Velocity", "Pressure", "Energy"]

        # Plot each column in a separate subplot
        for i in range(4):
            # axes[i//2, i%2].plot(columns[0], columns[i+1], label=f"{solver_dict[solver]}", marker='.', color=solver_color_dict[solver], linewidth=0)
            axes[i//2, i%2].plot(columns[0], columns[i+1], label=f"{solver_name_dict[solver]}", color=solver_color_dict[solver])


    for i in range(4):
        # Plot the analytic solution
        axes[i//2, i%2].set_xlabel('x')
        axes[i//2, i%2].set_ylabel(f'{columns_names[i]}')
        axes[i//2, i%2].grid(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_runtime(problem_type, solver_list, output_time, n_cells_list):
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))

    # Get current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for i, solver in enumerate(solver_list):
        # Build the array with all rutimes for all cell numbers
        runtime_list = np.zeros(len(n_cells_list))
        for j, n_cells in enumerate(n_cells_list):

            # Read the .out file
            file_path = os.path.join(script_dir, '..', f"output/{problem_type}", f'solver{solver}_t{output_time:.3f}_n{n_cells}_stats.out', )
            runtime = pd.read_csv(file_path, sep='\s+', header=None)
            runtime_list[j] = runtime[0].to_numpy()

        # Plot each column in a separate subplot
        axes.plot(n_cells_list, runtime_list, label=solver_name_dict[solver], color=solver_color_dict[solver], marker='o')
        axes.legend()
        axes.grid(True)

    axes.set_title(f'Runtime comparison of different solvers')
    axes.set_xlabel('Iteration number')
    axes.set_ylabel('Runtime')
    plt.tight_layout()
    plt.show()


def plot_2d_solution(problem_type, solver, n_cells, output_time_start, output_time_step, output_time_end):
    # Get current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    output_time_array = np.arange(output_time_start, output_time_end+output_time_step, output_time_step)
    solution_2d = np.zeros((len(output_time_array), 100))

    for i, output_time in enumerate(output_time_array):
        # Read the .out file
        file_path = os.path.join(script_dir, '..', f"output/{problem_type}", f'solver{solver}_t{output_time:.3f}_n{n_cells}.out')
        data = pd.read_csv(file_path, sep='\s+', header=None)

        # Read data into columns
        columns = np.zeros((data.shape[1],data.shape[0]))
        for i in range(5):
            columns[i] = data[i].to_numpy()

        solution_2d[i] = columns[2] # velocity

    # Color map
    plt.imshow(solution_2d, cmap='viridis')
    plt.gca().invert_yaxis()
    colorbar = plt.colorbar(location='bottom')
    colorbar.set_label('Velocity')
    plt.gca().set_yticks([0, 0.05, 0.4])
    plt.xlabel('X')
    plt.ylabel('Output time')
    plt.show()