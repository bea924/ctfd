import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from ex3_func import SteadyHeat2D_FVM
from ex3_basics import setUpMesh



##############################################################################
dim = 6
x = np.linspace(-1, 1, dim)
y = np.linspace(-1, 1, dim)

# Create the mesh grid
X, Y = np.meshgrid(x, y)
# X, Y = setUpMesh(n=dim, shape='linear')
# [N E S W]
dued = SteadyHeat2D_FVM(X, Y, boundary=['N', 'D', 'N', 'D'], TD=[3, 1, 6, 1])

solution = dued.solve()
# dued.solve()
A = dued.A
import matplotlib.pyplot as plt

# Example numpy matrix
matrix = A
matrix = solution.reshape((dim, dim))  # Replace this with your actual matrix
# matrix = np.zeros((dim ,dim))

# Plot the matrix as a grid with colored squares
plt.imshow(matrix, cmap='viridis')
plt.colorbar()  # Add color bar for reference
plt.show()

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# from matplotlib import cm
# surf = ax.plot_surface(X, Y, matrix, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# plt.show()
# print(dued.B)