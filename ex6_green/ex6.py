import numpy as np
import matplotlib.pyplot as plt
from ex6_func import GF_function, GF_1D_function, return_GF_matrix, return_GF_1D_array, HeatEq_2D_point


# test plot
Lx = Ly = 1
x_s = y_s = Lx/2
plot_GF = np.zeros((20, 20))
x = np.arange(0, Lx, 0.05)
y = np.arange(0, Ly, 0.05)
for i in range(20):
    for j in range(20):
        plot_GF[i, j] = GF_function(x[i], y[j], x_s, y_s, Lx, Ly, bc_e = "N", bc_n = "N", bc_w = "D", bc_s = "D")
        

# # Plot the heatmap
# plt.imshow(plot_GF, cmap='jet', interpolation='nearest', origin = 'lower')
# plt.colorbar()  # Add a color bar to show the color scale
# plt.title('Heatmap of 2D NumPy Array')
# plt.xlabel('Column Index')
# plt.ylabel('Row Index')
# plt.show()


# 2
omega = 5
lambda_v = 1
Lx = Ly = 1

# current observer point
x = 0.2
y = 0.2

# source surface area
x_s_start = 0
x_s_end = Lx
y_s_start = 0
y_s_end = Ly

# x_s and y_s array for integral calculation
n_step_s = 21
step_s_x = (x_s_end-x_s_start)/n_step_s
step_s_y = (y_s_end-y_s_start)/n_step_s
# x_s = np.arange(x_s_start, x_s_end, step_s_x)
# x_s[-1] = x_s_end # just to make sure we arrive to the border
# y_s = np.arange(y_s_start, y_s_end, step_s_y)
# y_s[-1] = y_s_end
x_s = np.linspace(x_s_start, x_s_end, n_step_s)
y_s = np.linspace(y_s_start, y_s_end, n_step_s)

#boundaries
bc_E = "D"
bc_N = "D"
bc_W = "D"
bc_S = "D"

# boundary values (T or q)
Temp = 10
q = 3

T_xy = HeatEq_2D_point(0.1, 0.1, omega, lambda_v, Lx, Ly, x_s_start, x_s_end, y_s_start, y_s_end, bc_E, bc_N, bc_W, bc_S, Temp, q)
print(T_xy)