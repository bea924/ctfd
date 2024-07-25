import numpy as np
from global_variables import GAMMA, G1, G2, G3, G4, G5, G6, G7, G8


def godunov_exact_riemann_solver(n_cells, density, velocity, pressure, sound_speed): # for now only exact riemann RPGODU
    """
    compute godunov intercell flux using the exact riemann solver
    """

    x_over_t = 0 # x/t must be 0 for the godunov
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

        # calls sample for x_over_t=0
        # get godunov state values
        dsam, usam, psam = sample(x_over_t, um, pm, d_local_L, u_local_L, p_local_L, c_local_L, d_local_R, u_local_R, p_local_R, c_local_R)
            
        # calculate intercell flux 1 2 3
        fluxes[0, i] = dsam * usam
        fluxes[1, i] = dsam * usam * usam + psam
        energy = 0.5 * usam * usam * dsam + psam/G8
        fluxes[2, i] = usam * (energy + psam)

    return fluxes


def exact_riemann_solver(d_local_L, u_local_L, p_local_L, c_local_L, d_local_R, u_local_R, p_local_R, c_local_R, max_iterations=20, tolerance=1e-05): #EXACT
    """
    computes the pressure (PM) and velocity (UM) in the star region using the exact Riemann solver
    """

    # p_start_guess = 0 # later implement GUESSP
    p_start_guess = guessp(d_local_L, u_local_L, p_local_L, c_local_L, d_local_R, u_local_R, p_local_R, c_local_R)
    tol = 10
    p_old = p_start_guess
    u_diff = u_local_R - u_local_L

    for i in range(max_iterations):
        # left state flux and its derivative
        f_L, f_Ld = prefun(p_old, p_local_L, c_local_L, d_local_L)
        # right state flux and its derivative
        f_R, f_Rd = prefun(p_old, p_local_R, c_local_R, d_local_R)
        # newton raphson method to update pressure
        p_new = p_old - (f_L + f_R + u_diff) / (f_Ld + f_Rd)

        # check if tolerance reached
        tol = 2 * np.abs((p_new - p_old) / (p_new + p_old))
        if tol <= tolerance:
            break

        if p_new < 0:
            p_new = tolerance

        p_old = p_new

    u_new = 0.5*(u_local_L + u_local_R + f_R - f_L)  # speed (is it maybe wrong? original: 0.5*(u_L + u_R + f_R - f_L)

    return p_new, u_new


def guessp(d_local_L, u_local_L, p_local_L, c_local_L, d_local_R, u_local_R, p_local_R, c_local_R):
    """
    initial guess for the pressure in the Star Region
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
            #select Two-Rarefaction Riemann solver
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


def prefun(p, p_K, c_K, d_K):
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


def sample(s, um, pm, d_L, u_L, p_L, c_L, d_R, u_R, p_R, c_R):
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