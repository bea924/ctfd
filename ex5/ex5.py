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
dimX = 8
dimY = 8
#possibilities J - Jacobi, G - Gauss-Siedel, SOR - SOR
solver = "SOR"

heat = SteadyHeat2Dsparse(Lx, Ly, dimX, dimY)

heat.set_south("d", T_d=5)
heat.set_west("d", T_d=30)
heat.set_north("d", T_d=30)
heat.set_east("d", T_d=5)

if (solver == "J"):
    T = heat.solveJacobi()
elif (solver == "G"):
    T = heat.solveGauss()
elif (solver == "SOR"):
    T = heat.solveSOR()
    
T.reshape((dimX, dimY))

end_time = time.time()
elapsed_time = end_time - start_time
print("Time taken using solver " + solver + ": " + str(elapsed_time) + " secs")

# solution plot
matrix = T.reshape((dimX, dimY))
plt.imshow(matrix, cmap='jet')
plt.colorbar()
plt.show()
