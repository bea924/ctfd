import numpy as np
from riemann_functions.global_variables import GAMMA, G8


def godunov_roe_solver(n_cells, density, velocity, pressure, sound_speed, conserved_var, dt, dx, entropy_fix_parameter=-0.1):
    """
    Roe fluxes intercell godunov calculation (Toro ch11)
    with entropy fix
    """

    # if entropy_fix_parameter negative, no entropy fix is done
    fluxes_cell = np.zeros((3, n_cells+2)) # the fluxes from within the cell
    fluxes_intercell = np.zeros((3, n_cells+2))
    right_ev = np.zeros(3) # right eigenvectors

    for i in range(n_cells+2):
        if (i < 1) or (i > n_cells): # conserved variable border cells were not initialised
            conserved_var[0, i] = density[i]
            conserved_var[1, i] = density[i] * velocity[i]
            conserved_var[2, i] = 0.5 * density[i] * velocity[i] * velocity[i] + pressure[i]/G8

        fluxes_cell[0,i] = conserved_var[1,i]
        fluxes_cell[1,i] = conserved_var[1,i] * velocity[i] + pressure[i]
        fluxes_cell[2,i] = velocity[i] *(conserved_var[2,i] + pressure[i])

    for i in range(n_cells+1):
        # obtaining the local left boundary constants per cell
        d_L = density[i]
        u_L = velocity[i]
        p_L = pressure[i]
        a_L = sound_speed[i]
        e_L = conserved_var[2,i]
        h_L = (e_L + p_L) / d_L
        # obtaining the local left boundary constants per cell
        d_R = density[i+1]
        u_R = velocity[i+1]
        p_R = pressure[i+1]
        a_R = sound_speed[i+1]
        e_R = conserved_var[2,i+1]
        h_R = (e_R + p_R) / d_R

        # compute roe averages from equation (11.118) in Toro
        roe_avg = np.sqrt(d_R/d_L) # alternative equivalent calculation proposed by Toro
        d_avg = roe_avg * d_L
        u_avg = (u_L + roe_avg*u_R) / (1 + roe_avg)
        h_avg = (h_L + roe_avg*h_R) / (1 + roe_avg)
        a_avg = np.sqrt(G8 * (h_avg - 0.5*u_avg*u_avg))

        # differences (Delta)
        u_diff = u_R - u_L
        p_diff = p_R - p_L

        # identify wave pattern
        if u_avg > 0:
            # contact wave goes to the right
            eval = u_avg - a_avg # given as lambda 1 in equation (11.107) in Toro
            snew = eval
            ak = (p_diff - d_avg*a_avg*u_diff) / (2* a_avg*a_avg) # given as alpha 1 in equation 11.113 in Toro
            cflm = eval* dt/dx

            if np.abs(cflm)  < entropy_fix_parameter:
                # small left wave speed is identified
                sig = 1
                umm, cmm = starvals(sig, d_L, u_L, e_L, ak, u_avg, a_avg, h_avg)
                sml = u_L - a_L
                smr = umm - cmm

                if (sml < 0) and (smr > 0):
                    # Left wave is a sonic rarefaction, speed is modified
                    snew = sml * (smr - eval)/(smr - sml)
            
            # Compute one-sided intercell flux from left side
            if snew < 0:
                # Compute right eigenvectors given in eq 11.108 in Toro
                right_ev[0] = 1
                right_ev[1] = u_avg - a_avg
                right_ev[2] = h_avg - u_avg*a_avg
                # compute one sided intercell flux using Equations 11.27 - 11.29 in Toro
                fluxes_intercell[0, i] = fluxes_cell[0,i] + snew*ak*right_ev[0]
                fluxes_intercell[1, i] = fluxes_cell[1,i] + snew*ak*right_ev[1]
                fluxes_intercell[2, i] = fluxes_cell[2,i] + snew*ak*right_ev[2]
            else:
                # Compute one-sided intercell flux
                fluxes_intercell[0,i] = fluxes_cell[0,i]
                fluxes_intercell[1,i] = fluxes_cell[1,i]
                fluxes_intercell[2,i] = fluxes_cell[2,i]

        else:
            # Contact wave goes to the right (one of these is wrong)
            eval = u_avg + a_avg # given by lambda in equation (11.107) in Toro
            snew = eval
            ak = (p_diff + d_avg*a_avg*u_diff) / (2* a_avg*a_avg) # solving alpha in equation 11.113 in Toro
            cflm = eval * dt/dx

            if np.abs(cflm)  < entropy_fix_parameter:
                # Small right wave speed is identified
                # Use Roe's Riemann solver to find particle speed UMM and sound speed CMM in start right state
                sig = -1
                umm, cmm = starvals(sig, d_R, u_R, e_R, ak, u_avg, a_avg, h_avg)
                sml = umm + cmm
                smr = u_R + a_R

                if (sml < 0) and (smr > 0):
                    # Right wave is a sonic rarefaction, speed is modified
                    snew = smr * (eval - sml) / (smr - sml)
            
            # Compute one-sided intercell flux from left side
            if snew > 0:
                # Compute right eigenvectors given in eq 11.108 in Toro
                right_ev[0] = 1
                right_ev[1] = u_avg + a_avg
                right_ev[2] = h_avg + u_avg*a_avg
                # compute one sided intercell flux using Equations 11.27 - 11.29 in Toro
                fluxes_intercell[0, i] = fluxes_cell[0,i+1] - snew*ak*right_ev[0]
                fluxes_intercell[1, i] = fluxes_cell[1,i+1] - snew*ak*right_ev[1]
                fluxes_intercell[2, i] = fluxes_cell[2,i+1] - snew*ak*right_ev[2]
            else:
                # Compute one-sided intercell flux
                fluxes_intercell[0, i] = fluxes_cell[0, i+1]
                fluxes_intercell[1, i] = fluxes_cell[1, i+1]
                fluxes_intercell[2, i] = fluxes_cell[2, i+1]

    return fluxes_intercell


def starvals(sig, d_K, u_K, e_K, ak, u_avg, a_avg, h_avg):
    """
    calculate the velocity and sound speed in the star region
    (used for entropy fix Harten and Hyman)
    """

    dmk = d_K + sig*ak
    umm = (d_K*u_K + sig*ak*(u_avg - sig*a_avg)) / dmk
    pm = G8 * (e_K + sig*ak*(h_avg - sig*u_avg*a_avg) - 0.5*dmk*umm*umm)
    cmm = np.sqrt(GAMMA * pm/dmk)

    return umm, cmm