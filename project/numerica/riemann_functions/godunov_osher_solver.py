import numpy as np
from riemann_functions.global_variables import GAMMA, G1, G2, G3, G4, G5, G6, G7, G8


def godunov_osher_solver(n_cells, density, velocity, pressure, sound_speed, conserved_var):
    """
    Osher Godunov method, with P ordering
    """

    fluxes_cell = np.zeros((3, n_cells+2)) # the fluxes from within the cell
    fluxes_intercell = np.zeros((3, n_cells+2))

    # Compute fluxes on data and conserved variables in fictitious cells
    for i in range(n_cells+2):
        if (i < 1) or (i > n_cells): # conserved variable was not initialised at borders yet
            conserved_var[0, i] = density[i]
            conserved_var[1, i] = density[i] * velocity[i]
            conserved_var[2, i] = 0.5 * density[i] * velocity[i] * velocity[i] + pressure[i]/G8

        fluxes_cell[0,i] = conserved_var[1,i]
        fluxes_cell[1,i] = conserved_var[1,i] * velocity[i] + pressure[i]
        fluxes_cell[2,i] = velocity[i] *(conserved_var[2,i] + pressure[i])

    for i in range(n_cells+1):
        d_L = density[i]
        u_L = velocity[i]
        p_L = pressure[i]
        a_L = sound_speed[i]

        d_R = density[i+1]
        u_R = velocity[i+1]
        p_R = pressure[i+1]
        a_R = sound_speed[i+1]

        # calculate star region values
        d_star_L, d_star_R, u_star, p_star, a_star_L, a_star_R = star_trrs_calculate(d_L, u_L, p_L, a_L, d_R, u_R, p_R, a_R)

        # Table 12.8
        # column 2
        if ((u_L - a_L) >= 0) and ((u_R + a_R) >= 0):
            
            # column 2, row 2
            if (u_star >= 0) and ((u_star - a_star_L) >= 0):
                fluxes_intercell[0, i] = fluxes_cell[0,i]
                fluxes_intercell[1, i] = fluxes_cell[1,i]
                fluxes_intercell[2, i] = fluxes_cell[2,i]
            
            # column 2, row 3
            if (u_star >= 0) and ((u_star - a_star_L) <= 0):
                d_sonic_L, u_sonic_L, p_sonic_L = sonic_left_calculate(d_L, u_L, p_L, a_L)
                fluxes_sonic_L = flux_calculate(d_sonic_L, u_sonic_L, p_sonic_L)
                fluxes_star_L = flux_calculate(d_star_L, u_star, p_star)
                fluxes_intercell[0, i] = fluxes_cell[0,i] - fluxes_sonic_L[0] + fluxes_star_L[0]
                fluxes_intercell[1, i] = fluxes_cell[1,i] - fluxes_sonic_L[1] + fluxes_star_L[1]
                fluxes_intercell[2, i] = fluxes_cell[2,i] - fluxes_sonic_L[2] + fluxes_star_L[2]

            # column 2, row 4
            if (u_star <= 0) and ((u_star + a_star_R) >= 0):
                d_sonic_L, u_sonic_L, p_sonic_L = sonic_left_calculate(d_L, u_L, p_L, a_L)
                fluxes_sonic_L = flux_calculate(d_sonic_L, u_sonic_L, p_sonic_L)
                fluxes_star_R = flux_calculate(d_star_R, u_star, p_star)
                fluxes_intercell[0, i] = fluxes_cell[0,i] - fluxes_sonic_L[0] + fluxes_star_R[0]
                fluxes_intercell[1, i] = fluxes_cell[1,i] - fluxes_sonic_L[1] + fluxes_star_R[1]
                fluxes_intercell[2, i] = fluxes_cell[2,i] - fluxes_sonic_L[2] + fluxes_star_R[2]

            # column 2, row 5
            if (u_star <= 0) and ((u_star + a_star_R) <= 0):
                d_sonic_L, u_sonic_L, p_sonic_L = sonic_left_calculate(d_L, u_L, p_L, a_L)
                fluxes_sonic_L = flux_calculate(d_sonic_L, u_sonic_L, p_sonic_L)
                d_sonic_R, u_sonic_R, p_sonic_R = sonic_right_calculate(d_R, u_R, p_R, a_R)
                fluxes_sonic_R = flux_calculate(d_sonic_R, u_sonic_R, p_sonic_R)
                fluxes_intercell[0, i] = fluxes_cell[0,i] - fluxes_sonic_L[0] + fluxes_sonic_R[0]
                fluxes_intercell[1, i] = fluxes_cell[1,i] - fluxes_sonic_L[1] + fluxes_sonic_R[1]
                fluxes_intercell[2, i] = fluxes_cell[2,i] - fluxes_sonic_L[2] + fluxes_sonic_R[2]


        # column 3
        if ((u_L - a_L) >= 0) and ((u_R + a_R) <= 0):

            # column 3, row 2
            if (u_star >= 0) and ((u_star-a_star_L)>=0):
                d_sonic_R, u_sonic_R, p_sonic_R = sonic_right_calculate(d_R, u_R, p_R, a_R)
                fluxes_sonic_R = flux_calculate(d_sonic_R, u_sonic_R, p_sonic_R)
                fluxes_intercell[0, i] = fluxes_cell[0,i] - fluxes_sonic_R[0] +  fluxes_cell[0,i+1]
                fluxes_intercell[1, i] = fluxes_cell[1,i] - fluxes_sonic_R[1] +  fluxes_cell[0,i+1]
                fluxes_intercell[2, i] = fluxes_cell[2,i] - fluxes_sonic_R[2] +  fluxes_cell[0,i+1]

            # column 3, row 3
            if (u_star >= 0) and ((u_star-a_star_L)<=0):
                d_sonic_L, u_sonic_L, p_sonic_L = sonic_left_calculate(d_L, u_L, p_L, a_L)
                fluxes_sonic_L = flux_calculate(d_sonic_L, u_sonic_L, p_sonic_L)
                fluxes_star_L = flux_calculate(d_star_L, u_star, p_star)
                d_sonic_R, u_sonic_R, p_sonic_R = sonic_right_calculate(d_R, u_R, p_R, a_R)
                fluxes_sonic_R = flux_calculate(d_sonic_R, u_sonic_R, p_sonic_R)
                fluxes_intercell[0, i] = fluxes_cell[0,i] - fluxes_sonic_L[0] + fluxes_star_L[0] - fluxes_sonic_R[0] + fluxes_cell[0,i+1]
                fluxes_intercell[1, i] = fluxes_cell[1,i] - fluxes_sonic_L[1] + fluxes_star_L[1] - fluxes_sonic_R[1] + fluxes_cell[0,i+1]
                fluxes_intercell[2, i] = fluxes_cell[2,i] - fluxes_sonic_L[2] + fluxes_star_L[2] - fluxes_sonic_R[2] + fluxes_cell[0,i+1]

            # column 3, row 4
            if (u_star <= 0) and ((u_star+a_star_L)>=0):
                d_sonic_L, u_sonic_L, p_sonic_L = sonic_left_calculate(d_L, u_L, p_L, a_L)
                fluxes_sonic_L = flux_calculate(d_sonic_L, u_sonic_L, p_sonic_L)
                fluxes_star_R = flux_calculate(d_star_R, u_star, p_star)
                d_sonic_R, u_sonic_R, p_sonic_R = sonic_right_calculate(d_R, u_R, p_R, a_R)
                fluxes_sonic_R = flux_calculate(d_sonic_R, u_sonic_R, p_sonic_R)
                fluxes_intercell[0, i] = fluxes_cell[0,i] - fluxes_sonic_L[0] + fluxes_star_R[0] - fluxes_sonic_R[0] + fluxes_cell[0,i+1]
                fluxes_intercell[1, i] = fluxes_cell[1,i] - fluxes_sonic_L[1] + fluxes_star_R[1] - fluxes_sonic_R[1] + fluxes_cell[0,i+1]
                fluxes_intercell[2, i] = fluxes_cell[2,i] - fluxes_sonic_L[2] + fluxes_star_R[2] - fluxes_sonic_R[2] + fluxes_cell[0,i+1]

            # column 3, row 5
            if (u_star <= 0) and ((u_star+a_star_L)<=0):
                d_sonic_L, u_sonic_L, p_sonic_L = sonic_left_calculate(d_L, u_L, p_L, a_L)
                fluxes_sonic_L = flux_calculate(d_sonic_L, u_sonic_L, p_sonic_L)
                fluxes_intercell[0, i] = fluxes_cell[0,i] - fluxes_sonic_L[0] + fluxes_cell[0,i+1]
                fluxes_intercell[1, i] = fluxes_cell[1,i] - fluxes_sonic_L[1] + fluxes_cell[0,i+1]
                fluxes_intercell[2, i] = fluxes_cell[2,i] - fluxes_sonic_L[2] + fluxes_cell[0,i+1]


        # column 4
        if ((u_L - a_L) <= 0) and ((u_R + a_R) >= 0):

            # column 4, row 2
            if (u_star >= 0) and ((u_star-a_star_L)>=0):
                d_sonic_L, u_sonic_L, p_sonic_L = sonic_left_calculate(d_L, u_L, p_L, a_L)
                fluxes_sonic_L = flux_calculate(d_sonic_L, u_sonic_L, p_sonic_L)
                fluxes_intercell[0, i] = fluxes_sonic_L[0]
                fluxes_intercell[1, i] = fluxes_sonic_L[1]
                fluxes_intercell[2, i] = fluxes_sonic_L[2]

            # column 4, row 3
            if (u_star >= 0) and ((u_star-a_star_L)<=0):
                fluxes_star_L = flux_calculate(d_star_L, u_star, p_star)
                fluxes_intercell[0, i] = fluxes_star_L[0]
                fluxes_intercell[1, i] = fluxes_star_L[1]
                fluxes_intercell[2, i] = fluxes_star_L[2]

            # column 4, row 4
            if (u_star <= 0) and ((u_star+a_star_L)>=0):
                fluxes_star_R = flux_calculate(d_star_R, u_star, p_star)
                fluxes_intercell[0, i] = fluxes_star_R[0]
                fluxes_intercell[1, i] = fluxes_star_R[1]
                fluxes_intercell[2, i] = fluxes_star_R[2]

            # column 4, row 5
            if (u_star <= 0) and ((u_star+a_star_L)<=0):
                d_sonic_R, u_sonic_R, p_sonic_R = sonic_right_calculate(d_R, u_R, p_R, a_R)
                fluxes_sonic_R = flux_calculate(d_sonic_R, u_sonic_R, p_sonic_R)
                fluxes_intercell[0, i] = fluxes_sonic_R[0]
                fluxes_intercell[1, i] = fluxes_sonic_R[1]
                fluxes_intercell[2, i] = fluxes_sonic_R[2]


        # column 5
        if ((u_L - a_L) <= 0) and ((u_R + a_R) <= 0):

            # column 5, row 2
            if (u_star >= 0) and ((u_star-a_star_L)>=0):
                d_sonic_L, u_sonic_L, p_sonic_L = sonic_left_calculate(d_L, u_L, p_L, a_L)
                fluxes_sonic_L = flux_calculate(d_sonic_L, u_sonic_L, p_sonic_L)
                d_sonic_R, u_sonic_R, p_sonic_R = sonic_right_calculate(d_R, u_R, p_R, a_R)
                fluxes_sonic_R = flux_calculate(d_sonic_R, u_sonic_R, p_sonic_R)
                fluxes_intercell[0, i] = fluxes_sonic_L[0] - fluxes_sonic_R[0] + fluxes_cell[0, i+1]
                fluxes_intercell[1, i] = fluxes_sonic_L[1] - fluxes_sonic_R[1] + fluxes_cell[1, i+1]
                fluxes_intercell[2, i] = fluxes_sonic_L[2] - fluxes_sonic_R[2] + fluxes_cell[2, i+1]

            # column 5, row 3
            if (u_star >= 0) and ((u_star-a_star_L)<=0):
                fluxes_star_L = flux_calculate(d_star_L, u_star, p_star)
                d_sonic_R, u_sonic_R, p_sonic_R = sonic_right_calculate(d_R, u_R, p_R, a_R)
                fluxes_sonic_R = flux_calculate(d_sonic_R, u_sonic_R, p_sonic_R)
                fluxes_intercell[0, i] = fluxes_star_L[0] - fluxes_sonic_R[0] + fluxes_cell[0, i+1]
                fluxes_intercell[1, i] = fluxes_star_L[1] - fluxes_sonic_R[1] + fluxes_cell[1, i+1]
                fluxes_intercell[2, i] = fluxes_star_L[2] - fluxes_sonic_R[2] + fluxes_cell[2, i+1]

            # column 5, row 4
            if (u_star <= 0) and ((u_star+a_star_L)>=0):
                fluxes_star_R = flux_calculate(d_star_R, u_star, p_star)
                d_sonic_R, u_sonic_R, p_sonic_R = sonic_right_calculate(d_R, u_R, p_R, a_R)
                fluxes_sonic_R = flux_calculate(d_sonic_R, u_sonic_R, p_sonic_R)
                fluxes_intercell[0, i] = fluxes_star_R[0] - fluxes_sonic_R[0] + fluxes_cell[0, i+1]
                fluxes_intercell[1, i] = fluxes_star_R[1] - fluxes_sonic_R[1] + fluxes_cell[1, i+1]
                fluxes_intercell[2, i] = fluxes_star_R[2] - fluxes_sonic_R[2] + fluxes_cell[2, i+1]

            # column 5, row 5
            if (u_star <= 0) and ((u_star+a_star_L)<=0):
                fluxes_intercell[0, i] = fluxes_cell[0, i+1]
                fluxes_intercell[1, i] = fluxes_cell[1, i+1]
                fluxes_intercell[2, i] = fluxes_cell[2, i+1]

    return fluxes_intercell
        

def star_trrs_calculate(d_L, u_L, p_L, a_L, d_R, u_R, p_R, a_R):
    """
    calcualte the star region values using TRRS approximation
    """

    # 12.72-73
    p_temp = (p_L/p_R) ** G1
    u_star = (p_temp * u_L/a_L + u_R/a_R + G4*(p_temp-1))/(p_temp/a_L + 1/a_R)
    p_star = 0.5 * (p_L * (1 + G7 * (u_L - u_star)/a_L)**G3 + p_R*(1 + G7 * (u_star - u_R)/a_R)**G3)

    d_star_L = d_L * (p_star/p_L)**(1/GAMMA) # 12.74
    d_star_R = d_R * (p_star/p_R)**(1/GAMMA)

    a_star_L = a_L * (p_star/p_L)**G1 # 12.68
    a_star_R = a_R * (p_star/p_R)**G1

    return d_star_L, d_star_R, u_star, p_star, a_star_L, a_star_R


def flux_calculate(d, u, p):
    """
    calculate flux at current point
    """
    # function F
    flux1 = d*u
    flux2 = d*u*u + p
    flux3 = u * (0.5*d*u*u + p/G8 + p)

    return flux1, flux2, flux3


def sonic_left_calculate(d_L, u_L, p_L, a_L):
    """
    The solution for the left sonic point
    """
    # 12.75
    u_sonic_L = G6*u_L + a_L*G5
    a_star_L = u_sonic_L
    d_sonic_L = d_L * (a_star_L/a_L)**G4 
    p_sonic_L = p_L * (d_sonic_L/d_L)**GAMMA

    return d_sonic_L, u_sonic_L, p_sonic_L


def sonic_right_calculate(d_R, u_R, p_R, a_R):
    """
    The solution for the right sonic point
    """
    # 12.76
    u_sonic_R = G6*u_R - a_R*G5
    a_star_R = -u_sonic_R
    d_sonic_R = d_R * (a_star_R/a_R)**G4
    p_sonic_R = p_R * (d_sonic_R/d_R)**GAMMA

    return d_sonic_R, u_sonic_R, p_sonic_R