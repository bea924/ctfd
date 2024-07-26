import numpy as np
from global_variables import GAMMA, G1, G2, G3, G4, G5, G6, G7, G8
from exact_solver import state_variables_sample, pressure_functions_return, exact_riemann_solver


def godunov_exact_riemann_solver(n_cells, density, velocity, pressure, sound_speed):
    """
    compute godunov intercell flux using the exact riemann solver
    """

    x_over_t = 0 # s = x/t must be 0 for the godunov sampling
    fluxes = np.zeros((3, n_cells+2))
    
    for i in range(n_cells+1):
        d_local_L = density[i]
        u_local_L = velocity[i]
        p_local_L = pressure[i]
        c_local_L = sound_speed[i]
        d_local_R = density[i+1]
        u_local_R = velocity[i+1]
        p_local_R = pressure[i+1]
        c_local_R = sound_speed[i+1]

        pm, um = exact_riemann_solver(d_local_L, u_local_L, p_local_L, c_local_L, d_local_R, u_local_R, p_local_R, c_local_R)

        # get godunov state values
        dsam, usam, psam = state_variables_sample(x_over_t, um, pm, d_local_L, u_local_L, p_local_L, c_local_L, d_local_R, u_local_R, p_local_R, c_local_R)
            
        # calculate intercell flux 1 2 3
        fluxes[0, i] = dsam * usam
        fluxes[1, i] = dsam * usam * usam + psam
        energy = 0.5 * usam * usam * dsam + psam/G8
        fluxes[2, i] = usam * (energy + psam)

    return fluxes


def p_starregion_approximate_guess(d_local_L, u_local_L, p_local_L, c_local_L, d_local_R, u_local_R, p_local_R, c_local_R):
    """
    initial guess for the pressure in the Star Region using approximate methods
    """

    cup = 0.25 * (d_local_L + d_local_R) * (c_local_L + c_local_R) # combined term involving the average of densities and sound speeds
    ppv = 0.5 * (p_local_L + p_local_R) + 0.5 * (u_local_L - u_local_R) * cup # Calculates an initial guess for the pressure in the Star Region using a formula that combines the average pressures and the difference in velocities scaled by cup
    ppv = max(0.0, ppv)
    p_min = min(p_local_L, p_local_R)
    p_max = max(p_local_L, p_local_R)
    q_max = p_max / p_min

    if q_max <= 2 and p_min <= ppv and ppv <= p_max:
        # Select PVRS Riemann solver
        pm_start_guess = ppv
    else:
        if (ppv < p_min):
            #select Two-Rarefaction Riemann solverc
            pq = (p_local_L/p_local_R)**G1
            um =  (pq * u_local_L/c_local_L + u_local_R/c_local_R + G4 * (pq - 1.0)) / (pq/c_local_L + 1.0/c_local_R)
            ptl = 1.0 + G7 * (u_local_L - um)/c_local_L
            ptr = 1.0 + G7 * (um - u_local_R)/c_local_R
            pm_start_guess = 0.5 * (p_local_L * ptl**G3 + p_local_R*ptr**G3)
        else:
            #select two-shock Riemann solver
            gel = np.sqrt( (G5/d_local_L) / (G6*p_local_L + ppv) )
            ger = np.sqrt( (G5/d_local_R) / (G6*p_local_R + ppv) )
            pm_start_guess = (gel*p_local_L + ger*p_local_R - (u_local_R -u_local_L)) / (gel + ger) 

    return pm_start_guess