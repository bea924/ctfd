import numpy as np
from riemann_functions.global_variables import GAMMA, G1, G2, G3, G4, G5, G6, G7, G8, MAX_TIMESTEPS


def exact_riemann_solver(n_cells, d_L, u_L, p_L, d_R, u_R, p_R, dx, diaphragm_position, output_time):
    # compute sound speeds
    c_L = np.sqrt(GAMMA * p_L/d_L)
    c_R = np.sqrt(GAMMA * p_R/d_R)

    # test if vacuum still missing

    pm, um = exact_riemann_pu(d_L, u_L, p_L, c_L, d_R, u_R, p_R, c_R)

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


def exact_riemann_pu(d_local_L, u_local_L, p_local_L, c_local_L, d_local_R, u_local_R, p_local_R, c_local_R, tolerance=1e-05):
    """
    computes the pressure (PM) and velocity (UM) in the star region using the exact Riemann solver
    """
    p_start_guess = 1
    tol = 10
    p_old = p_start_guess
    u_diff = u_local_R - u_local_L

    for i in range(MAX_TIMESTEPS):
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


def pressure_functions_return(p, p_K, c_K, d_K):
    """
    computes the pressure functions and their derivatives used in the exact Riemann solver
    - selects if raref or shock
    - p_K : pressure value for specific region
    - p : input pressire value
    """

    if (p <= p_K):
        # raref
        p_ratio = p / p_K # ratio between pressures
        f = G4 * c_K * (p_ratio**G1 - 1)
        fd = (1.0 / (d_K * c_K)) * p_ratio**(-G2)
    else:
        # shock wave
        a_K = G5 / d_K
        b_K = G6 * p_K
        qrt = np.sqrt(a_K / (b_K + p))
        f = (p - p_K) * qrt
        fd = (1.0 - 0.5*(p - p_K)/(b_K + p)) * qrt

    return f, fd


def state_variables_sample(s, um, pm, d_L, u_L, p_L, c_L, d_R, u_R, p_R, c_R):
    """
    sample the solution throughout the wave pattern resulting from a Riemann problem
    determines the state variables (density, velocity, and pressure) at a given position S in the flow based on the computed pressure and velocity in the star region.
    """
    # samples the solution throght the wave
    # input s pm um GAMMA s = x/t
    # output d u p

    if (s <= um):
        # sample left of the contact discontinuity
        if (pm <= p_L):
            # left rarefaction
            sh_L = u_L - c_L
            if (s <= sh_L):
                # sampled point is left data state
                d = d_L
                u = u_L
                p = p_L
            else:
                cm_L = c_L*(pm/p_L)**G1
                stl = um-cm_L
                if (s > stl):
                    #  Sampled point is Star Left state
                    d = d_L*(pm/p_L)**(1.0/GAMMA)
                    u = um
                    p = pm
                else:
                    # Sampled point is inside left fan
                    u = G5*(c_L + G7 * u_L + s)
                    c = G5*(c_L + G7 * (u_L - s))
                    d = d_L*(c/c_L)**G4
                    p = p_L * (c/c_L)**G3
        else:
            # left shock
            pm_L = pm/p_L
            sl = u_L - c_L*np.sqrt(G2 * pm_L + G1)
            if (s <= sl):
                # sampled point is left data state
                d = d_L
                u = u_L
                p = p_L
            else:
                #  Sampled point is Star Left stat
                d = d_L*(pm_L + G6)/(pm_L*G6 + 1) 
                u = um
                p = pm
    else:
        # sample right of the contact discontinuity
        if (pm > p_R):
            # left shock
            pm_R = pm/p_R
            s_R = u_R + c_R*np.sqrt(G2*pm_R + G1)
            if (s >= s_R):
                # sampled point is right data state
                d = d_R
                u = u_R
                p = p_R
            else:
                # Sampled point is Star Right state
                d = d_R*(pm_R+G6)/(pm_R*G6 + 1.0)
                u = um
                p = pm
        else:
            # right rarefaction
            sh_R = u_R + c_R
            if (s >= sh_R):
                # sampled point is right data state
                d = d_R
                u = u_R
                p = p_R
            else:
                cm_R = c_R*(pm/p_R)**G1
                st_R = um + cm_R
                if (s <= st_R):
                    #  Sampled point is Star Right stat
                    d = d_R*((pm/p_R)**(1.0/GAMMA))
                    u = um
                    p = pm
                else:
                    #  Sampled point is inside left fan
                    u = G5*(-c_R + G7*u_R + s)
                    c = G5*(c_R - G7*(u_R - s))
                    d = d_R*(c/c_R)**G4
                    p = p_R*(c/c_R)**G3

    return d, u, p