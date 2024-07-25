import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 0 exact
# 1 lax
# 2 roe
solver = 0
output_time_start = 0.0
output_time_end = 0.7
output_time_step = 0.05

output_time_array = np.arange(output_time_start, output_time_end+output_time_step, output_time_step)
solution_2d = np.zeros((len(output_time_array), 100))

for i, output_time in enumerate(output_time_array):
    # Read the .out file
    file_path = f'output/solver{solver}_t{output_time:.2f}.out' # truncated it at 2 decimal places
    # file_path = f'project/numerica/output/solver{solver}_t{output_time:.2f}.out' # for debugger
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)

    columns = np.zeros((data.shape[1],data.shape[0]))
    # Access columns
    columns[0] = data[0].to_numpy()
    columns[1] = data[1].to_numpy()
    columns[2] = data[2].to_numpy()
    columns[3] = data[3].to_numpy()
    columns[4] = data[4].to_numpy()

    columns_names = ["X_position", "Density", "Velocity", "Pressure", "Energy"]

    solution_2d[i] = columns[4] # i want velocity for now


# Plot the matrix as a color map
plt.imshow(solution_2d, cmap='viridis')  # 'viridis' is a color map, you can choose others like 'hot', 'cool', etc.
plt.gca().invert_yaxis()
plt.colorbar()  # Add a color bar to the side
plt.title('2D Matrix Color Map')
plt.xlabel('X axis')
plt.ylabel('Time axis')
plt.show()