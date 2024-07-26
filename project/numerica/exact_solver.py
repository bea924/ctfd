import numpy as np
from global_variables import GAMMA, G1, G2, G3, G4, G5, G6, G7, G8
from godunov_approximate_solver import pressure_functions_return, p_starregion_approximate_guess, state_variables_sample


def exact_riemann(n_cells, d_L, u_L, p_L, d_R, u_R, p_R, dx, diaphragm_position, output_time, max_iterations):
    # compute sound speeds
    c_L = np.sqrt(GAMMA * p_L/d_L)
    c_R = np.sqrt(GAMMA * p_R/d_R)

    # test if vacuum still missing

    pm, um = starpu(d_L, u_L, p_L, c_L, d_R, u_R, p_R, c_R, max_iterations)

    density = np.zeros(n_cells+1)
    velocity = np.zeros(n_cells+1)
    pressure = np.zeros(n_cells+1)

    for i in range(n_cells+1): #maybe +1?
        x_position = (i - 0.5)*dx
        s = (x_position - diaphragm_position) /output_time

        # Solution at point (X,T) = ( XPOS - DIAPH1,TIMEOU) is found
        dsam, usam, psam = state_variables_sample(s, um, pm, d_L, u_L, p_L, c_L, d_R, u_R, p_R, c_R)
        density[i] = dsam
        velocity[i] = usam
        pressure[i] = psam

    return density, velocity, pressure




# def starpu(d_local_L, u_local_L, p_local_L, c_local_L, d_local_R, u_local_R, p_local_R, c_local_R, max_iterations, tolerance=1e-06):
#     p_start_guess = p_starregion_approximate_guess(d_local_L, u_local_L, p_local_L, c_local_L, d_local_R, u_local_R, p_local_R, c_local_R)
#     p_old = p_start_guess
#     u_diff = u_local_R - u_local_L

#     for i in range(max_iterations):
#         # left state flux and its derivative
#         f_L, f_Ld = pressure_functions_return(p_old, p_local_L, c_local_L, d_local_L)
#         # right state flux and its derivative
#         f_R, f_Rd = pressure_functions_return(p_old, p_local_R, c_local_R, d_local_R)
#         # newton raphson method to update pressure
#         p_new = p_old - (f_L + f_R + u_diff) / (f_Ld + f_Rd)

#         # check if tolerance reached
#         tol = 2 * np.abs((p_new - p_old) / (p_new + p_old))
#         if tol <= tolerance:
#             break

#         if p_new < 0:
#             p_new = tolerance

#         p_old = p_new

#     u_new = 0.5*(u_local_L + u_local_R + f_R - f_L)

#     return p_new, u_new


def starpu(d_local_L, u_local_L, p_local_L, c_local_L, d_local_R, u_local_R, p_local_R, c_local_R, max_iterations=20, tolerance=1e-05):
    """
    computes the pressure (PM) and velocity (UM) in the star region using the exact Riemann solver
    """

    p_start_guess = p_starregion_approximate_guess(d_local_L, u_local_L, p_local_L, c_local_L, d_local_R, u_local_R, p_local_R, c_local_R)
    tol = 10
    p_old = p_start_guess
    u_diff = u_local_R - u_local_L

    for i in range(max_iterations):
        # left state flux and its derivative
        f_L, f_Ld = pressure_functions_return(p_old, p_local_L, c_local_L, d_local_L)
        # right state flux and its derivative
        f_R, f_Rd = pressure_functions_return(p_old, p_local_R, c_local_R, d_local_R)
        # newton raphson method to update pressure
        p_new = p_old - (f_L + f_R + u_diff) / (f_Ld + f_Rd)

        # check if tolerance reached
        tol = 2 * np.abs((p_new - p_old) / (p_new + p_old))
        if tol <= tolerance:
            break

        if p_new < 0:
            p_new = tolerance

        p_old = p_new

    u_new = 0.5*(u_local_L + u_local_R + f_R - f_L)

    return p_new, u_new