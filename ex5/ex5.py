from ex5_func import SteadyHeat2D
import matplotlib.pyplot as plt

# Test
Lx = 1
Ly = 1
dimX = 5
dimY = 5

heat = SteadyHeat2D(Lx, Ly, dimX, dimY)

heat.set_south("d", T_d=5)
heat.set_west("d", T_d=5)
heat.set_north("d", T_d=10)
heat.set_east("d", T_d=5)

# heat.set_north("d", T_d=0)A
# heat.set_south("d", T_d=0)
# heat.set_east("d", T_d=0)
# heat.set_west("d", T_d=0)


T = heat.solve()
T.reshape((dimX, dimY))



# solution plot
matrix = T.reshape((dimX, dimY))
plt.imshow(matrix, cmap='magma')
plt.colorbar()
plt.show()


# # sparse matrix plot
# from matplotlib.pyplot import spy
# spy(heat.A)
# plt.show()
