import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 0 exact
# 1 lax
# 2 roe
solver = 0
output_time = 0.2


# Read the .out file
file_path = f'output/output_t{output_time}_solver{solver}.out'
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
fig, axes = plt.subplots(4, 1, figsize=(6, 8))
x = np.arange(0,100)

# Plot each column in a separate subplot
for i in range(4):
    axes[i].plot(columns[0], columns[i+1], label=columns_names[i])
    axes[i].set_title(f'Plot of {columns_names[i]}')
    axes[i].legend()
    axes[i].grid(True)

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()