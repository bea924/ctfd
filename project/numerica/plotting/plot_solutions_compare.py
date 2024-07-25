import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 0 exact
# 1 lax
# 2 roe
solver_list = [0, 1, 2]
output_time = 0.20

solver_dict = {
    0: 'Exact',
    1: "Lax",
    2: 'Roe'
}

fig, axes = plt.subplots(4, 1, figsize=(6, 8))

for j, solver in enumerate(solver_list):
    # Read the .out file
    file_path = f'output/solver{solver}_t{output_time:.2f}.out'
    # file_path = f'project/numerica/output/solver{solver}_t{output_time}.out' # for debugger
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)

    # Print the data to verify
    print(data)

    columns = np.zeros((data.shape[1],data.shape[0]))
    # Access columns
    columns[0] = data[0].to_numpy()
    columns[1] = data[1].to_numpy()
    columns[2] = data[2].to_numpy()
    columns[3] = data[3].to_numpy()
    columns[4] = data[4].to_numpy()

    columns_names = ["Density", "Velocity", "Pressure", "Energy"]

    # Create a figure and subplots
    # x = np.arange(0,100)

    # Plot each column in a separate subplot
    for i in range(4):
        axes[i].plot(columns[0], columns[i+1], label=solver_dict[solver])


for i in range(4):
    axes[i].set_title(f'Plot of {columns_names[i]}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel(f'{columns_names[i]}')
    axes[i].legend()
    axes[i].grid(True)

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()