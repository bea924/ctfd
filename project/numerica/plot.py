import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read the .out file
file_path = 'project/numerica/exact.out'
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

# Print columns to verify
# print("Column 1:")
# print(column_1)
# print("Column 2:")
# print(column_2)
# print("Column 3:")
# print(column_3)
# print("Column 4:")
# print(column_4)
# print("Column 5:")
# print(column_5)


# Create a figure and subplots
fig, axes = plt.subplots(5, 1, figsize=(6, 8))
x = np.arange(0,100)

# Plot each column in a separate subplot
for i in range(5):
    axes[i].plot(x, columns[i], label=f'Column {i+1}')
    axes[i].set_title(f'Plot of Column {i+1}')
    axes[i].legend()
    axes[i].grid(True)

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
