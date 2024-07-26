import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from style_plots import solver_dict, solver_color_dict

##################################
######### USER INPUT #############
##################################
# Choose the problem type: ModifiedSod, StationaryContact
problem_type = "ModifiedSod"
# choose the solver: ExactRiemann [0], ExactRiemannAppr [1], LaxFriedrichs [2], Roe [3], Osher [4]
solver_list = [0, 2, 3, 4]
output_time = 0.2
n_cells_list = [100, 200, 300, 400, 500, 700, 1000, 2000]
##################################
##################################


fig, axes = plt.subplots(1, 1, figsize=(8, 6))

# Get current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

for i, solver in enumerate(solver_list):
    # Build the array with all rutimes for all cell numbers
    runtime_list = np.zeros(len(n_cells_list))
    for j, n_cells in enumerate(n_cells_list):

        # Read the .out file
        file_path = os.path.join(script_dir, '..', f"output/{problem_type}", f'solver{solver}_t{output_time:.3f}_n{n_cells}_stats.out', )
        runtime = pd.read_csv(file_path, delim_whitespace=True, header=None)
        runtime_list[j] = runtime[0].to_numpy()
    # x = np.arange(0,len(runtime))

    # Plot each column in a separate subplot
    axes.plot(n_cells_list, runtime_list, label=solver_dict[solver], color=solver_color_dict[solver], marker='o')
    axes.legend()
    axes.grid(True)


axes.set_title(f'Runtime comparison of different solvers')
axes.set_xlabel('Iteration number')
axes.set_ylabel('Runtime')
plt.tight_layout()
plt.show()