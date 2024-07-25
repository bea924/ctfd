import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

##################################
######### USER INPUT #############
##################################
# Choose the problem type: ModifiedSod, StationaryContact
problem_type = "ModifiedSod"
# choose the solver: ExactRiemann [0], LaxFriedrichs [1], Roe [2]
solver_list = [0, 1, 2]
output_time = 0.2
##################################
##################################


fig, axes = plt.subplots(1, 1, figsize=(8, 6))
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

for i, solver in enumerate(solver_list):
    # Read the .out file
    file_path = os.path.join(script_dir, '..', f"output/{problem_type}", f'solver{solver}_t{output_time:.2f}_stats.out', )
    runtime = pd.read_csv(file_path, delim_whitespace=True, header=None)

    runtime = runtime[0].to_numpy()
    x = np.arange(0,len(runtime))

    # Plot each column in a separate subplot
    axes.plot(x, runtime, label=solver_dict[i], color=solver_color_dict[solver])
    axes.legend()
    axes.grid(True)


axes.set_title(f'Runtime comparison of different solvers')
axes.set_xlabel('Iteration number')
axes.set_ylabel('Runtime')
plt.tight_layout()
plt.show()