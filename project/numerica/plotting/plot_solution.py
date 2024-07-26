import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

##################################
######### USER INPUT #############
##################################
# Choose the problem type: ModifiedSod, StationaryContact
problem_type = "ModifiedSod"
# choose the solver: ExactRiemann [0], ExactRiemannAppr [1], LaxFriedrichs [2], Roe [3], Osher [4]
solver = 4
output_time = 0.2
##################################
##################################

columns_names = ["Density", "Velocity", "Pressure", "Energy"]
axis_names = ["Density [kg/m2]", "Velocity", "Pressure [bar]", "Energy"]
solver_dict = {
    0: 'ExactRiemann',
    1: 'ExactRiemannAppr',
    2: "LaxFriedrichs",
    3: 'Roe',
    4: 'Osher'
}

# Read the .out file
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '..', f"output/{problem_type}", f'solver{solver}_t{output_time:.3f}.out')
data = pd.read_csv(file_path, delim_whitespace=True, header=None)

# read the json analytic solution file
jsonfile_path = os.path.join(script_dir, 'analytic_solutions.json')
with open(jsonfile_path, 'r') as file:
    analytic_solution = json.load(file)

columns = np.zeros((data.shape[1],data.shape[0]))
# Access columns
columns[0] = data[0].to_numpy()
columns[1] = data[1].to_numpy()
columns[2] = data[2].to_numpy()
columns[3] = data[3].to_numpy()
columns[4] = data[4].to_numpy()

# Figure and subplots
fig, axes = plt.subplots(2, 2, figsize=(8, 7))
x = np.arange(0,100)

for i in range(4):
    # the indexing for axes is to transform it into 2d subplotting
    # Plot the analytic solution
    axes[i//2, i%2].plot(analytic_solution[f"{problem_type}"][columns_names[i]]["x"], analytic_solution[f"{problem_type}"][columns_names[i]]["y"], label=f"Analytic", color="orange")
    # Plot numerical solution
    axes[i//2, i%2].plot(columns[0], columns[i+1], label=f"{solver_dict[solver]}", marker='.', color="green", linewidth=0)
    axes[i//2, i%2].set_xlabel('x')
    axes[i//2, i%2].set_ylabel(f'{columns_names[i]}')
    axes[i//2, i%2].grid(True)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()