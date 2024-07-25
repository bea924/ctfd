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
solver = 0
output_time_start = 0.0
output_time_end = 0.4
output_time_step = 0.01
##################################
##################################



# Get current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

output_time_array = np.arange(output_time_start, output_time_end+output_time_step, output_time_step)
solution_2d = np.zeros((len(output_time_array), 100))

for i, output_time in enumerate(output_time_array):
    # Read the .out file
    file_path = os.path.join(script_dir, '..', f"output/{problem_type}", f'solver{solver}_t{output_time:.2f}.out')
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)

    # Read data into columns
    columns = np.zeros((data.shape[1],data.shape[0]))
    columns[0] = data[0].to_numpy()
    columns[1] = data[1].to_numpy()
    columns[2] = data[2].to_numpy()
    columns[3] = data[3].to_numpy()
    columns[4] = data[4].to_numpy()

    columns_names = ["X_position", "Density", "Velocity", "Pressure", "Energy"]
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