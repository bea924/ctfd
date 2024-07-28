import numpy as np
from riemann_functions.global_variables import G8


def laxfriedriechs_solver(n_cells, density, velocity, pressure, sound_speed, conserved_var, dx, dt):
    """
    calculate the fluxes using the Lax_Friedrichs method
    """ 

    fluxes_cell = np.zeros((3, n_cells+2))
    fluxes_intercell = np.zeros((3, n_cells+2))


    for i in range(n_cells+2):
        conserved_var[0, i] = density[i]
        conserved_var[1, i] = density[i] * velocity[i]
        conserved_var[2, i] = 0.5 * density[i] * velocity[i] * velocity[i] + pressure[i]/G8

        # the fluxes from within the cell
        fluxes_cell[0,i] = conserved_var[1,i]
        fluxes_cell[1,i] = conserved_var[1,i] * velocity[i] + pressure[i]
        fluxes_cell[2,i] = velocity[i] *(conserved_var[2,i] + pressure[i])

    for i in range(n_cells+1):
        # lax fiedrich intercell fluxes
        fluxes_intercell[0,i] = 0.5 * (fluxes_cell[0, i] + fluxes_cell[0, i+1]) + 0.5 * (dx/dt) * (conserved_var[0,i] - conserved_var[0,i+1]) # Taken from Toro equation 5.77
        fluxes_intercell[1,i] = 0.5 * (fluxes_cell[1, i] + fluxes_cell[1, i+1]) + 0.5 * (dx/dt) * (conserved_var[1,i] - conserved_var[1,i+1])
        fluxes_intercell[2,i] = 0.5 * (fluxes_cell[2, i] + fluxes_cell[2, i+1]) + 0.5 * (dx/dt) * (conserved_var[2,i] - conserved_var[2,i+1])

    return fluxes_intercell
        