import numpy as np
from riemann_functions.global_variables import GAMMA, G1, G2, G3, G4, G5, G6, G7, G8, MAX_TIMESTEPS
from riemann_functions.exact_riemann_solver import state_variables_sample, pressure_functions_return, exact_riemann_solver, exact_riemann_pu


def godunov_approximate_riemann_solver(n_cells, density, velocity, pressure, sound_speed):
    """
    compute godunov intercell flux using the exact riemann solver with approximations
    """

    x_over_t = 0 # s = x/t must be 0 for the godunov sampling
    fluxes = np.zeros((3, n_cells+2))
    
    for i in range(n_cells+1):
        d_L = density[i]
        u_L = velocity[i]
        p_L = pressure[i]
        a_L = sound_speed[i]
        d_R = density[i+1]
        u_R = velocity[i+1]
        p_R = pressure[i+1]
        a_R = sound_speed[i+1]

        pm, um = exact_riemann_pu(d_L, u_L, p_L, a_L, d_R, u_R, p_R, a_R)

        # get godunov state values
        dsam, usam, psam = state_variables_sample(x_over_t, um, pm, d_L, u_L, p_L, a_L, d_R, u_R, p_R, a_R)
            
        # calculate intercell flux 1 2 3
        fluxes[0, i] = dsam * usam
        fluxes[1, i] = dsam * usam * usam + psam
        energy = 0.5 * usam * usam * dsam + psam/G8
        fluxes[2, i] = usam * (energy + psam)

    return fluxes


def approximate_riemann_pu(d_L, u_L, p_L, a_L, d_R, u_R, p_R, a_R, tolerance=1e-05):
    """
    computes the pressure (PM) and velocity (UM) in the star region using the exact Riemann solver
    """
    p_start_guess = p_starregion_approximate_guess(d_L, u_L, p_L, a_L, d_R, u_R, p_R, a_R)
    tol = 10
    p_old = p_start_guess
    u_diff = u_R - u_L

    for i in range(MAX_TIMESTEPS):
        # left state flux and its derivative
        f_L, f_Ld = pressure_functions_return(p_old, p_L, a_L, d_L)
        # right state flux and its derivative
        f_R, f_Rd = pressure_functions_return(p_old, p_R, a_R, d_R)
        # newton raphson method to update pressure
        p_new = p_old - (f_L + f_R + u_diff) / (f_Ld + f_Rd)

        # check if tolerance reached
        tol = 2 * np.abs((p_new - p_old) / (p_new + p_old))
        if tol <= tolerance:
            break

        if p_new < 0:
            p_new = tolerance

        p_old = p_new

    u_new = 0.5*(u_L + u_R + f_R - f_L)

    return p_new, u_new


def p_starregion_approximate_guess(d_L, u_L, p_L, a_L, d_R, u_R, p_R, a_R):
    """
    initial guess for the pressure in the Star Region using approximate methods
    """

    cup = 0.25 * (d_L + d_R) * (a_L + a_R) # combined term involving the average of densities and sound speeds
    ppv = 0.5 * (p_L + p_R) + 0.5 * (u_L - u_R) * cup # Calculates an initial guess for the pressure in the Star Region using a formula that combines the average pressures and the difference in velocities scaled by cup
    ppv = max(0.0, ppv)
    p_min = min(p_L, p_R)
    p_max = max(p_L, p_R)
    q_max = p_max / p_min

    if q_max <= 2 and p_min <= ppv and ppv <= p_max:
        # Select PVRS Riemann solver
        pm_start_guess = ppv
    else:
        if (ppv < p_min):
            #select Two-Rarefaction Riemann solverc
            pq = (p_L/p_R)**G1
            um =  (pq * u_L/a_L + u_R/a_R + G4 * (pq - 1.0)) / (pq/a_L + 1.0/a_R)
            ptl = 1.0 + G7 * (u_L - um)/a_L
            ptr = 1.0 + G7 * (um - u_R)/a_R
            pm_start_guess = 0.5 * (p_L * ptl**G3 + p_R*ptr**G3)
        else:
            #select two-shock Riemann solver
            gel = np.sqrt( (G5/d_L) / (G6*p_L + ppv) )
            ger = np.sqrt( (G5/d_R) / (G6*p_R + ppv) )
            pm_start_guess = (gel*p_L + ger*p_R - (u_R -u_L)) / (gel + ger) 

    return pm_start_guess