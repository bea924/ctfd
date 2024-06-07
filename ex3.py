import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from ex3_func import SteadyHeat2D_FVM
from ex3_basics import setUpMesh

##############################################################################
dim = 50
x = np.linspace(-1, 1, dim) 
y = np.linspace(-1, 1, dim)

# Create the mesh grid
# X, Y = np.meshgrid(x, y)
X, T = setUpMesh(n=dim, shape='linear')
# [N E S W]
dued = SteadyHeat2D_FVM(X, T, boundary=['D', 'D', 'D', 'D'], TD=[40, 40, 40, 100])

solution = "steady"
solution = dued.solve(solution='unsteadyi', dt=0.1, t_end=5)
A = dued.A
import matplotlib.pyplot as plt

# Example numpy matrix
matrix = A
matrix = solution.reshape((dim, dim))
c = pd.DataFrame(matrix)
# print(c)

# # Plot the matrix as a grid with colored squares
# plt.imshow(matrix, cmap='viridis')
# plt.colorbar()  # Add color bar for reference
# plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
from matplotlib import cm
surf = ax.plot_surface(X, T, matrix, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()