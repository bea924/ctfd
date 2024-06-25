from ex5_func_sparse import SteadyHeat2Dsparse
from ex5_func import SteadyHeat2D
import matplotlib.pyplot as plt
from matplotlib.pyplot import spy
from scipy.sparse import dia_matrix, csr_array

import time

start_time = time.time()

# Test
Lx = 1
Ly = 1
dimX = 4
dimY = 4
#possibilities J - Jacobi, G - Gauss-Siedel, SOR - SOR
solver = "G"

heat = SteadyHeat2Dsparse(Lx, Ly, dimX, dimY)

heat.set_south("d", T_d=5)
heat.set_west("d", T_d=30)
heat.set_north("d", T_d=30)
heat.set_east("d", T_d=5)

# heat.set_north("d", T_d=0)
# heat.set_south("d", T_d=0)
# heat.set_east("d", T_d=0)
# heat.set_west("d", T_d=0)

if (solver == "J"):
    T = heat.solveJacobi()
elif (solver == "G"):
    T = heat.solveGauss()
    
T.reshape((dimX, dimY))

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)

# solution plot
matrix = T.reshape((dimX, dimY))
plt.imshow(matrix, cmap='magma')
plt.colorbar()
plt.show()


# # sparse matrix plot
# from matplotlib.pyplot import spy
# spy(heat.D_1)
# plt.show()

# from matplotlib.pyplot import spy
# spy(heat.R)
# plt.show()