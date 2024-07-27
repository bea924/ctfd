import numpy as np
from riemann_functions.global_variables import G8


def laxfriedriechs_solver(n_cells, density, velocity, pressure, sound_speed, conserved_var, dx, dt):
    """
    compute an intercell flux FI(K, I) according to the Lax-Friedrichs scheme
    """ 

    fd = np.zeros((3, n_cells+2))
    fluxes = np.zeros((3, n_cells+2))


    for i in range(n_cells+2):
        conserved_var[0, i] = density[i]
        conserved_var[1, i] = density[i] * velocity[i]
        conserved_var[2, i] = 0.5 * density[i] * velocity[i] * velocity[i] + pressure[i]/G8

        fd[0,i] = conserved_var[1,i]
        fd[1,i] = conserved_var[1,i] * velocity[i] + pressure[i]
        fd[2,i] = velocity[i] *(conserved_var[2,i] + pressure[i])

    for i in range(n_cells+1):
        # lax fiedrich flux
        fluxes[0,i] = 0.5 * (fd[0, i] + fd[0, i+1]) + 0.5 * (dx/dt) * (conserved_var[0,i] - conserved_var[0,i+1]) # 5.77
        fluxes[1,i] = 0.5 * (fd[1, i] + fd[1, i+1]) + 0.5 * (dx/dt) * (conserved_var[1,i] - conserved_var[1,i+1])
        fluxes[2,i] = 0.5 * (fd[2, i] + fd[2, i+1]) + 0.5 * (dx/dt) * (conserved_var[2,i] - conserved_var[2,i+1])

    return fluxes
        