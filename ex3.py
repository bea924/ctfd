import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from ex3_func import SteadyHeat2D_FVM
from ex3_basics import setUpMesh
import matplotlib.animation as animation

##############################################################################
dim = 50
x = np.linspace(-1, 1, dim) 
y = np.linspace(-1, 1, dim)

# Create the mesh grid
# X, Y = np.meshgrid(x, y)
X, T = setUpMesh(n=dim, shape='linear')
# [N E S W]
dued = SteadyHeat2D_FVM(X, T, boundary=['D', 'D', 'D', 'D'], TD=[40, 40, 40, 100])
mode = "steady"
#solution = dued.solve(mode='unsteadyi', dt=0.1, t_end=20)
import matplotlib.pyplot as plt

dued.stencil()

def animate(num, plot, T0):
    ax.clear()    
    ax.set(xlim3d=(0, 10), xlabel='X')
    ax.set(ylim3d=(0, 10), ylabel='Y')
    ax.set(zlim3d=(20, 100), zlabel='Z')
    T0 = dued.solve_animation(mode='unsteadyi', dt=0.1).reshape((dim,dim))
    plot = ax.plot_surface(X, T, T0, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    return plot,

# Example numpy matrix
#matrix = solution.reshape((dim, dim))
# = pd.DataFrame(matrix)
# print(c)

# # Plot the matrix as a grid with colored squares
# plt.imshow(matrix, cmap='viridis')
# plt.colorbar()  # Add color bar for reference
# plt.show()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

from matplotlib import cm


ax.set(xlim3d=(0, 10), xlabel='X')
ax.set(ylim3d=(0, 10), ylabel='Y')
ax.set(zlim3d=(20, 100), zlabel='Z')

#first frame
T0 = dued.solve_animation(mode='unsteadyi', dt=0.1).reshape((dim,dim))
surf = ax.plot_surface(X, T, T0, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ani = animation.FuncAnimation(
    fig, animate, 100, fargs=(surf, T0), interval=10, blit=False)
plt.show()

