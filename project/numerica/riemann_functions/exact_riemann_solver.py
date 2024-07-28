import numpy as np
from riemann_functions.global_variables import GAMMA, G1, G2, G3, G4, G5, G6, G7, G8, MAX_TIMESTEPS


def exact_riemann_solver(n_cells, d_L, u_L, p_L, d_R, u_R, p_R, dx, diaphragm_position, output_time):
    """
    solution using exact Riemann solver
    """

    # compute sound speeds
    a_L = np.sqrt(GAMMA * p_L/d_L)
    a_R = np.sqrt(GAMMA * p_R/d_R)

    # test if vacuum still missing

    p_star, u_star = exact_riemann_star_calculate(d_L, u_L, p_L, a_L, d_R, u_R, p_R, a_R)

    density = np.zeros(n_cells+1)
    velocity = np.zeros(n_cells+1)
    pressure = np.zeros(n_cells+1)

    for i in range(n_cells+1): #maybe +1?
        x_position = (i - 0.5)*dx
        s = (x_position - diaphragm_position) /output_time

        d_sample, u_sample, p_sample = state_variables_sample(s, u_star, p_star, d_L, u_L, p_L, a_L, d_R, u_R, p_R, a_R)
        density[i] = d_sample
        velocity[i] = u_sample
        pressure[i] = p_sample

    return density, velocity, pressure


def exact_riemann_star_calculate(d_L, u_L, p_L, a_L, d_R, u_R, p_R, a_R, tolerance=1e-05):
    """
    calculate the star region values for perssure and speed using exact riemann pressure functions L/R
    """
    p_start_guess = 1 # random guess, according to Toro, not too crucial for most applications bc of function behaviour
    tol = 10 # initial tolerance
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

        # pressure not negative
        if p_new < 0:
            p_new = tolerance

        p_old = p_new

    u_new = 0.5*(u_L + u_R + f_R - f_L)

    return p_new, u_new


def pressure_functions_return(p, p_K, c_K, d_K):
    """
    computes the pressure functions and their derivatives used in the exact Riemann solver
    - selects if raref or shock
    - p_K : pressure value for specific region L/R
    - p : input pressire value
    - f : function 
    - fd : function derivative
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


def state_variables_sample(s, u_star, p_star, d_L, u_L, p_L, a_L, d_R, u_R, p_R, a_R):
    """
    samples the solution for a give s=x/t
    return density, velocity, and pressure
    """

    if (s <= u_star):
        # sample left of the contact discontinuity
        if (p_star <= p_L):
            # left rarefaction
            s_head_L = u_L - a_L
            if (s <= s_head_L):
                # sampled point is left data state
                d = d_L
                u = u_L
                p = p_L
            else:
                cm_L = a_L*(p_star/p_L)**G1
                stl = u_star-cm_L
                if (s > stl):
                    #  Sampled point is Star Left state
                    d = d_L*(p_star/p_L)**(1.0/GAMMA)
                    u = u_star
                    p = p_star
                else:
                    # Sampled point is inside left fan
                    u = G5*(a_L + G7 * u_L + s)
                    a = G5*(a_L + G7 * (u_L - s))
                    d = d_L*(a/a_L)**G4
                    p = p_L * (a/a_L)**G3
        else:
            # left shock
            p_star_L = p_star/p_L
            sl = u_L - a_L*np.sqrt(G2 * p_star_L + G1)
            if (s <= sl):
                # sampled point is left data state
                d = d_L
                u = u_L
                p = p_L
            else:
                #  Sampled point is Star Left stat
                d = d_L*(p_star_L + G6)/(p_star_L*G6 + 1) 
                u = u_star
                p = p_star
    else:
        # sample right of the contact discontinuity
        if (p_star > p_R):
            # left shock
            p_star_R = p_star/p_R
            s_R = u_R + a_R*np.sqrt(G2*p_star_R + G1)
            if (s >= s_R):
                # sampled point is right data state
                d = d_R
                u = u_R
                p = p_R
            else:
                # Sampled point is Star Right state
                d = d_R*(p_star_R+G6)/(p_star_R*G6 + 1.0)
                u = u_star
                p = p_star
        else:
            # right rarefaction
            s_head_R = u_R + a_R
            if (s >= s_head_R):
                # sampled point is right data state
                d = d_R
                u = u_R
                p = p_R
            else:
                cm_R = a_R*(p_star/p_R)**G1
                s_tail_R = u_star + cm_R
                if (s <= s_tail_R):
                    #  Sampled point is Star Right stat
                    d = d_R*((p_star/p_R)**(1.0/GAMMA))
                    u = u_star
                    p = p_star
                else:
                    #  Sampled point is inside left fan
                    u = G5*(-a_R + G7*u_R + s)
                    a = G5*(a_R - G7*(u_R - s))
                    d = d_R*(a/a_R)**G4
                    p = p_R*(a/a_R)**G3

    return d, u, p