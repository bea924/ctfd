from ex5_func_sparse import SteadyHeat2Dsparse
# from ex5_func import SteadyHeat2D
import matplotlib.pyplot as plt
from matplotlib.pyplot import spy
from scipy.sparse import dia_matrix, csr_array

import time

start_time = time.time()

# Your code here
time.sleep(2)  # Example code

end_time = time.time()
elapsed_time = end_time - start_time

# Test
Lx = 1
Ly = 1
dimX = 4
dimY = 4

heat = SteadyHeat2Dsparse(Lx, Ly, dimX, dimY)

heat.set_south("d", T_d=5)
heat.set_west("d", T_d=10)
heat.set_north("d", T_d=10)
heat.set_east("d", T_d=5)

# heat.set_north("d", T_d=0)A
# heat.set_south("d", T_d=0)
# heat.set_east("d", T_d=0)
# heat.set_west("d", T_d=0)


T = heat.solveJacobi()
T.reshape((dimX, dimY))
# Visualize the diag array
# heat.plot_diag()
heat.print_diag()
heat.print_b()



# solution plot
matrix = T.reshape((dimX, dimY))
plt.imshow(matrix, cmap='jet')
plt.colorbar()
plt.show()


# # sparse matrix plot
# from matplotlib.pyplot import spy
# spy(heat.D_1)
# plt.show()

# from matplotlib.pyplot import spy
# spy(heat.R)
# plt.show()