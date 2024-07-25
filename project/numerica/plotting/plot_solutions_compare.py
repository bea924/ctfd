import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

##################################
######### USER INPUT #############
##################################
# Choose the problem type: ModifiedSod, StationaryContact
problem_type = "ModifiedSod"
# choose the solver: ExactRiemann [0], LaxFriedrichs [1], Roe [2]
solver_list = [0, 2]
output_time = 0.2
##################################
##################################

solver_dict = {
    0: 'ExactRiemann',
    1: "LaxFriedrichs",
    2: 'Roe'
}

solver_color_dict = {
    0: 'green',
    1: "blue",
    2: 'purple'
}

# Get current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# read the json analytic solution file
jsonfile_path = os.path.join(script_dir, 'analytic_solutions.json')
with open(jsonfile_path, 'r') as file:
    analytic_solution = json.load(file)

fig, axes = plt.subplots(2, 2, figsize=(7, 6))

for j, solver in enumerate(solver_list):
    # Read the .out file
    file_path = os.path.join(script_dir, '..', f"output/{problem_type}", f'solver{solver}_t{output_time:.2f}.out')
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    
    # Access columns
    columns = np.zeros((data.shape[1],data.shape[0]))
    columns[0] = data[0].to_numpy()
    columns[1] = data[1].to_numpy()
    columns[2] = data[2].to_numpy()
    columns[3] = data[3].to_numpy()
    columns[4] = data[4].to_numpy()

    columns_names = ["Density", "Velocity", "Pressure", "Energy"]

    # Plot each column in a separate subplot
    for i in range(4):
        # axes[i//2, i%2].plot(columns[0], columns[i+1], label=solver_dict[solver])
        axes[i//2, i%2].plot(columns[0], columns[i+1], label=f"{solver_dict[solver]}", marker='.', color=solver_color_dict[solver], linewidth=0)


for i in range(4):
    # Plot the analytic solution
    axes[i//2, i%2].plot(analytic_solution[f"{problem_type}"][columns_names[i]]["x"], analytic_solution[f"{problem_type}"][columns_names[i]]["y"], label=f"Analytic", color="orange")
    axes[i//2, i%2].set_xlabel('x')
    axes[i//2, i%2].set_ylabel(f'{columns_names[i]}')
    axes[i//2, i%2].grid(True)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()