import numpy as np
from ex6_func import GF_function, GF_1D_function, return_GF_matrix, return_GF_1D_array

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
n_step_s = 20
step_s_x = (x_s_end-x_s_start)/n_step_s
step_s_y = (y_s_end-y_s_start)/n_step_s
x_s = np.arange(x_s_start, x_s_end, step_s_x)
x_s[-1] = x_s_end # just to make sure we arrive to the border
y_s = np.arange(y_s_start, y_s_end, step_s_y)
y_s[-1] = y_s_end

#boundaries
bc_E = "D"
bc_N = "D"
bc_W = "D"
bc_S = "D"

# boundary values (T or q)
T = 10
q = 3

# 2D GF integral over surface
GF_S = return_GF_matrix(x, y, x_s_start, x_s_end, y_s_start, y_s_end, Lx, Ly, bc_E, bc_N, bc_W, bc_S) # 2D GF(x_s, y_s)
integral_GF_ys = np.trapz(GF_S*omega, y_s, axis=0) #over y
integral_GF_ys_xs = np.trapz(integral_GF_ys, x_s) #over x

# 1D GF as a function of xs and ys
GF_x = return_GF_1D_array(x, x_s_start, x_s_end, Lx, bc_E, bc_W) # 1D GF(x_s)
GF_y = return_GF_1D_array(y, y_s_start, x_s_end, Ly, bc_S, bc_N) # 1D GF(y_s)

# # East integral -> Dirichlet
# dGdx = np.diff(GF_y) / np.diff(x_s)
# integral_GF_E = np.trapz(dGdx*T, y_s) #over y