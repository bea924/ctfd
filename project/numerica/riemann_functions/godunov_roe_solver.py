import numpy as np
from riemann_functions.global_variables import GAMMA, G8


def godunov_roe_solver(n_cells, density, velocity, pressure, sound_speed, conserved_var, dt, dx, entropy_fix_parameter=-0.1):
    """
    to compute an intercell Godunov flux using the ROE approximate Riemann solver with entropy fix
    according to Harten and Hyman. See Chap. 11 of Ref. 1 and original references therein
    """

    # if entropy_fix_parameter negative, no entropy fix is done
    fd = np.zeros((3, n_cells+2)) # not sure what this is
    fluxes = np.zeros((3, n_cells+2))
    revec = np.zeros(3) # right eigenvectors

    for i in range(n_cells+2):
        if (i < 1) or (i > n_cells): # i guess filling in where the conserved varaibles are still 0?
            conserved_var[0, i] = density[i]
            conserved_var[1, i] = density[i] * velocity[i]
            conserved_var[2, i] = 0.5 * density[i] * velocity[i] * velocity[i] + pressure[i]/G8

        fd[0,i] = conserved_var[1,i]
        fd[1,i] = conserved_var[1,i] * velocity[i] + pressure[i]
        fd[2,i] = velocity[i] *(conserved_var[2,i] + pressure[i])

    # Solve Riemann problem (i,i+1) and store quantities in I
    for i in range(n_cells+1):
        d_local_L = density[i]
        u_local_L = velocity[i]
        p_local_L = pressure[i]
        c_local_L = sound_speed[i]
        e_local_L = conserved_var[2,i]
        h_local_L = (e_local_L + p_local_L) / d_local_L

        d_local_R = density[i+1]
        u_local_R = velocity[i+1]
        p_local_R = pressure[i+1]
        c_local_R = sound_speed[i+1]
        e_local_R = conserved_var[2,i+1]
        h_local_R = (e_local_R + p_local_R) / d_local_R

        # compute roe averages
        roe_avg = np.sqrt(d_local_R/d_local_L)
        d_local_avg = roe_avg * d_local_L
        u_local_avg = (u_local_L + roe_avg*u_local_R) / (1 + roe_avg)
        h_local_avg = (h_local_L + roe_avg*h_local_R) / (1 + roe_avg)
        c_local_avg = np.sqrt(G8 * (h_local_avg - 0.5*u_local_avg*u_local_avg))

        # compute increments
        u_diff = u_local_R - u_local_L
        p_diff = p_local_R - p_local_L

        # identify wave pattern
        if u_local_avg > 0:
            # contact wave goes to the right
            eval = u_local_avg - c_local_avg
            snew = eval
            ak = (p_diff - d_local_avg*c_local_avg*u_diff) / (2* c_local_avg*c_local_avg)
            cflm = eval* dt/dx

            if np.abs(cflm)  < entropy_fix_parameter:
                # small left wave speed is identified
                sig = 1
                umm, cmm = starvals(sig, d_local_L, u_local_L, e_local_L, ak, u_local_avg, c_local_avg, h_local_avg)
                sml = u_local_L - c_local_L
                smr = umm - cmm

                if (sml < 0) and (smr > 0):
                    # Left wave is a sonic rarefaction, speed is modified
                    snew = sml * (smr - eval)/(smr - sml)
            
            # Compute one-sided intercell flux from left side
            if snew < 0:
                # Compute right eigenvectors
                revec[0] = 1
                revec[1] = u_local_avg - c_local_avg
                revec[2] = h_local_avg - u_local_avg*c_local_avg
                # compute one sided intercell flux
                fluxes[0, i] = fd[0,i] + snew*ak*revec[0]
                fluxes[1, i] = fd[1,i] + snew*ak*revec[1]
                fluxes[2, i] = fd[2,i] + snew*ak*revec[2]
            else:
                # Compute one-sided intercell flux
                fluxes[0,i] = fd[0,i]
                fluxes[1,i] = fd[1,i]
                fluxes[2,i] = fd[2,i]

        else:
            # Contact wave goes to the right (one of these is wrong)
            eval = u_local_avg + c_local_avg
            snew = eval
            ak = (p_diff + d_local_avg*c_local_avg*u_diff) / (2* c_local_avg*c_local_avg)
            cflm = eval * dt/dx

            if np.abs(cflm)  < entropy_fix_parameter:
                # Small right wave speed is identified
                # Use Roe's Riemann solver to find particle speed UMM and sound speed CMM in start right state
                sig = -1
                umm, cmm = starvals(sig, d_local_R, u_local_R, e_local_R, ak, u_local_avg, c_local_avg, h_local_avg)
                sml = umm + cmm
                smr = u_local_R + c_local_R

                if (sml < 0) and (smr > 0):
                    # Right wave is a sonic rarefaction, speed is modified
                    snew = smr * (eval - sml) / (smr - sml)
            
            # Compute one-sided intercell flux from left side
            if snew > 0:
                # Compute right eigenvectors
                revec[0] = 1
                revec[1] = u_local_avg + c_local_avg
                revec[2] = h_local_avg + u_local_avg*c_local_avg
                # compute one sided intercell flux
                fluxes[0, i] = fd[0,i+1] - snew*ak*revec[0]
                fluxes[1, i] = fd[1,i+1] - snew*ak*revec[1]
                fluxes[2, i] = fd[2,i+1] - snew*ak*revec[2]
            else:
                # Compute one-sided intercell flux
                fluxes[0, i] = fd[0, i+1]
                fluxes[1, i] = fd[1, i+1]
                fluxes[2, i] = fd[2, i+1]

    return fluxes


def starvals(sig, d_local_K, u_local_K, e_local_K, ak, u_local_avg, c_local_avg, h_local_avg):
    """
    to compute particle velocity and sound speed in
    appropriate Star state, according to Roe's Riemann
    solver for states, in order to apply entropy fix
    of Harten and Hyman.
    """

    dmk = d_local_K + sig*ak
    umm = (d_local_K*u_local_K + sig*ak*(u_local_avg - sig*c_local_avg)) / dmk
    pm = G8 * (e_local_K + sig*ak*(h_local_avg - sig*u_local_avg*c_local_avg) - 0.5*dmk*umm*umm)
    cmm = np.sqrt(GAMMA * pm/dmk)

    return umm, cmm