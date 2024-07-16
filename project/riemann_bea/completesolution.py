from completesolution_func import main_riemann_solver
import matplotlib.pyplot as plt
import numpy as np

# Constants for the ideal gas
gamma = 1.4  # Ratio of specific heats for diatomic gas (e.g., air)
MPA = 1e6    # 1 MPa = 1e6 Pa

# Left state (L)
d_L = 1.0         # Density in kg/m³
p_L = 1.0 * MPA     # Pressure in Pa (1 MPa)
u_L = 0.0           # Velocity in m/s

# Right state (R)
# d_R = 0.125       # Density in kg/m³
d_R = 1.0           # Density in kg/m³
p_R = 0.1 * MPA     # Pressure in Pa (0.1 MPa)
u_R = 0.0           # Velocity in m/s

timeout = 1
diaph = 5
n_cells = 100
domain_length = 10

solution = main_riemann_solver(d_L, u_L, p_L, d_R, u_R, p_R, gamma, MPA, domain_length, n_cells, diaph, timeout)

matrix = solution[:,0]
x = np.linspace(0, 10, n_cells)

plt.plot(x, matrix, label="idk")

# Add titles and labels
plt.title('Linear Plot')
plt.xlabel('x')
plt.ylabel('y')

# Add a legend
plt.legend()

# Display the plot
plt.show()


# # Create the plot
# fig, ax = plt.subplots()

# # Create the heatmap
# cax = ax.matshow(solution, cmap='viridis')

# # Add a color bar
# fig.colorbar(cax)

# # Annotate each cell with the numeric value
# for (i, j), val in np.ndenumerate(solution):
#     ax.text(j, i, f'{val}', ha='center', va='center', color='white')

# # Set axis labels
# ax.set_xticks(np.arange(solution.shape[1]))
# ax.set_yticks(np.arange(solution.shape[0]))

# # Optionally, you can add axis labels
# # ax.set_xticklabels(['A', 'B', 'C', 'D'])
# # ax.set_yticklabels(['W', 'X', 'Y', 'Z'])

# # Set title
# plt.title('Matrix Heatmap with Color Scale')

# # Display the plot
# plt.show()
