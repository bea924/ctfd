import numpy as np
from global_variables import GAMMA, G1, G2, G3, G4, G5, G6, G7, G8


def godunov_osher_solver(n_cells, density, velocity, pressure, sound_speed, conserved_var):
    """
    to compute an intercell Godunov flux using
    the OSHER approximate Riemann solver with
    PHYSICAL or P-ordering of integration paths.
    Details to be found in Chapt. 12 of Ref. 1
    and original referenced therein
    """
    # for now only osher P solution!! maybe add O later but performs worse so not necessary

    fd = np.zeros((3, n_cells+2)) # not sure what this is
    fluxes = np.zeros((3, n_cells+2))

    # Compute fluxes on data and conserved variables in fictitious cells
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

        d_local_R = density[i+1]
        u_local_R = velocity[i+1]
        p_local_R = pressure[i+1]
        c_local_R = sound_speed[i+1]

        # Compute intersection points with P-ordering using two-rarefaction approximation
        dml, dmr, um, pm, cml, cmr = intersp(d_local_L, u_local_L, p_local_L, c_local_L, d_local_R, u_local_R, p_local_R, c_local_R)

        # Case A: Table 12.8, column 2
        if ((u_local_L - c_local_L) >= 0) and ((u_local_R - c_local_R) >= 0):
            
            # case A1
            if (um >= 0) and ((um - cml) >= 0):
                fluxes[0, i] = fd[0,i]
                fluxes[1, i] = fd[1,i]
                fluxes[2, i] = fd[2,i]
            
            # case A2
            if (um >= 0) and ((um - cml) <= 0):
                dsl, usl, psl = sonlef(d_local_L, u_local_L, p_local_L, c_local_L)
                fsl = fluxeval(dsl, usl, psl)
                fml = fluxeval(dml, um, pm)
                fluxes[0, i] = fd[0,i] - fsl[0] + fml[0]
                fluxes[1, i] = fd[1,i] - fsl[1] + fml[1]
                fluxes[2, i] = fd[2,i] - fsl[2] + fml[2]

            # case A3
            if (um <= 0) and ((um + cmr) >= 0):
                dsl, usl, psl = sonlef(d_local_L, u_local_L, p_local_L, c_local_L)
                fsl = fluxeval(dsl, usl, psl)
                fmr = fluxeval(dmr, um, pm)
                fluxes[0, i] = fd[0,i] - fsl[0] + fmr[0]
                fluxes[1, i] = fd[1,i] - fsl[1] + fmr[1]
                fluxes[2, i] = fd[2,i] - fsl[2] + fmr[2]

            # case A4
            if (um <= 0) and ((um + cmr) <= 0):
                dsl, usl, psl = sonlef(d_local_L, u_local_L, p_local_L, c_local_L)
                fsl = fluxeval(dsl, usl, psl)
                dsr, usr, psr = sonrig(d_local_R, u_local_R, p_local_R, c_local_R)
                fsr = fluxeval(dsr, usr, psr)
                fluxes[0, i] = fd[0,i] - fsl[0] + fsr[0]
                fluxes[1, i] = fd[1,i] - fsl[1] + fsr[1]
                fluxes[2, i] = fd[2,i] - fsl[2] + fsr[2]


        # Case B: Table 12.8, column 3
        if ((u_local_L - c_local_L) >= 0) and ((u_local_R + c_local_R) <= 0):

            # case B1
            if (um >= 0) and ((um-cml)>=0):
                dsr, usr, psr = sonrig(d_local_R, u_local_R, p_local_R, c_local_R)
                fsr = fluxeval(dsr, usr, psr)
                fluxes[0, i] = fd[0,i] - fsr[0] +  fd[0,i+1]
                fluxes[1, i] = fd[1,i] - fsr[1] +  fd[0,i+1]
                fluxes[2, i] = fd[2,i] - fsr[2] +  fd[0,i+1]

            # case B2
            if (um >= 0) and ((um-cml)<=0):
                dsl, usl, psl = sonlef(d_local_L, u_local_L, p_local_L, c_local_L)
                fsl = fluxeval(dsl, usl, psl)
                fml = fluxeval(dml, um, pm)
                dsr, usr, psr = sonrig(d_local_R, u_local_R, p_local_R, c_local_R)
                fsr = fluxeval(dsr, usr, psr)
                fluxes[0, i] = fd[0,i] - fsl[0] + fml[0] - fsr[0] + fd[0,i+1]
                fluxes[1, i] = fd[1,i] - fsl[1] + fml[1] - fsr[1] + fd[0,i+1]
                fluxes[2, i] = fd[2,i] - fsl[2] + fml[2] - fsr[2] + fd[0,i+1]

            # case B3
            if (um <= 0) and ((um+cml)>=0):
                dsl, usl, psl = sonlef(d_local_L, u_local_L, p_local_L, c_local_L)
                fsl = fluxeval(dsl, usl, psl)
                fmr = fluxeval(dmr, um, pm)
                dsr, usr, psr = sonrig(d_local_R, u_local_R, p_local_R, c_local_R)
                fsr = fluxeval(dsr, usr, psr)
                fluxes[0, i] = fd[0,i] - fsl[0] + fmr[0] - fsr[0] + fd[0,i+1]
                fluxes[1, i] = fd[1,i] - fsl[1] + fmr[1] - fsr[1] + fd[0,i+1]
                fluxes[2, i] = fd[2,i] - fsl[2] + fmr[2] - fsr[2] + fd[0,i+1]

            # case B4
            if (um <= 0) and ((um+cml)<=0):
                dsl, usl, psl = sonlef(d_local_L, u_local_L, p_local_L, c_local_L)
                fsl = fluxeval(dsl, usl, psl)
                fluxes[0, i] = fd[0,i] - fsl[0] + fd[0,i+1]
                fluxes[1, i] = fd[1,i] - fsl[1] + fd[0,i+1]
                fluxes[2, i] = fd[2,i] - fsl[2] + fd[0,i+1]


        # Case C: Table 12.8, column 4
        if ((u_local_L - c_local_L) <= 0) and ((u_local_R + c_local_R) >= 0):

            # case C1
            if (um >= 0) and ((um-cml)>=0):
                dsl, usl, psl = sonlef(d_local_L, u_local_L, p_local_L, c_local_L)
                fsl = fluxeval(dsl, usl, psl)
                fluxes[0, i] = fsl[0]
                fluxes[1, i] = fsl[1]
                fluxes[2, i] = fsl[2]

            # case C2
            if (um >= 0) and ((um-cml)<=0):
                fml = fluxeval(dml, um, pm)
                fluxes[0, i] = fml[0]
                fluxes[1, i] = fml[1]
                fluxes[2, i] = fml[2]

            # case C3
            if (um <= 0) and ((um+cml)>=0):
                fmr = fluxeval(dmr, um, pm)
                fluxes[0, i] = fmr[0]
                fluxes[1, i] = fmr[1]
                fluxes[2, i] = fmr[2]

            # case C4
            if (um <= 0) and ((um+cml)<=0):
                dsr, usr, psr = sonrig(d_local_R, u_local_R, p_local_R, c_local_R)
                fsr = fluxeval(dsr, usr, psr)
                fluxes[0, i] = fsr[0]
                fluxes[1, i] = fsr[1]
                fluxes[2, i] = fsr[2]


        # Case D: Table 12.8, column 5
        if ((u_local_L - c_local_L) <= 0) and ((u_local_R + c_local_R) <= 0):

            # case D1
            if (um >= 0) and ((um-cml)>=0):
                dsl, usl, psl = sonlef(d_local_L, u_local_L, p_local_L, c_local_L)
                fsl = fluxeval(dsl, usl, psl)
                dsr, usr, psr = sonrig(d_local_R, u_local_R, p_local_R, c_local_R)
                fsr = fluxeval(dsr, usr, psr)
                fluxes[0, i] = fsl[0] - fsr[0] + fd[0, i+1]
                fluxes[1, i] = fsl[1] - fsr[1] + fd[1, i+1]
                fluxes[2, i] = fsl[2] - fsr[2] + fd[2, i+1]

            # case D2
            if (um >= 0) and ((um-cml)<=0):
                fml = fluxeval(dml, um, pm)
                dsr, usr, psr = sonrig(d_local_R, u_local_R, p_local_R, c_local_R)
                fsr = fluxeval(dsr, usr, psr)
                fluxes[0, i] = fml[0] - fsr[0] + fd[0, i+1]
                fluxes[1, i] = fml[1] - fsr[1] + fd[1, i+1]
                fluxes[2, i] = fml[2] - fsr[2] + fd[2, i+1]

            # case D3
            if (um <= 0) and ((um+cml)>=0):
                fmr = fluxeval(dmr, um, pm)
                dsr, usr, psr = sonrig(d_local_R, u_local_R, p_local_R, c_local_R)
                fsr = fluxeval(dsr, usr, psr)
                fluxes[0, i] = fmr[0] - fsr[0] + fd[0, i+1]
                fluxes[1, i] = fmr[1] - fsr[1] + fd[1, i+1]
                fluxes[2, i] = fmr[2] - fsr[2] + fd[2, i+1]

            # case C4
            if (um <= 0) and ((um+cml)<=0):
                fluxes[0, i] = fd[0, i+1]
                fluxes[1, i] = fd[1, i+1]
                fluxes[2, i] = fd[2, i+1]

    return fluxes
        

def intersp(d_L, u_L, p_L, c_L, d_R, u_R, p_R, c_R):
    """
    to compute intersection points for Osher solver with
    PHYSICAL or P-ordering of integration paths.
    Compute: PM, UM, DML, DMR, CML, CMR
    according to two-rarefaction approximation
    """

    pq = (p_L/p_R)**G1
    um = (pq*u_L/c_L + u_R/c_R + G4*(pq-1))/(pq/c_L + 1/c_R)
    ptl = 1 + G7*(u_L - um)/c_L
    ptr = 1 + G7*(um - u_R)/c_R
    pm = 0.5 * (p_L * ptl**G3 + p_R*ptr**G3)

    dml = d_L * (pm/p_L)**(1/GAMMA)
    dmr = d_R * (pm/p_R)**(1/GAMMA)

    cml = c_L * (pm/p_L)**G1
    cmr = c_R * (pm/p_R)**G1

    return dml, dmr, um, pm, cml, cmr


def fluxeval(d, u, p):
    """
    to fluxes FX at values P, U, D
    """

    flux1 = d*u
    flux2 = d*u*u + p
    flux3 = u * (0.5*d*u*u + p/G8 + p)

    return flux1, flux2, flux3


def sonlef(d_L, u_L, p_L, c_L):
    """
    to compute left SONIC state PSL, USL, DSL
    """

    usl = G6*u_L + c_L*G5
    csl = usl
    dsl = d_L * (csl/c_L)**G4
    psl = p_L * (dsl/d_L)**GAMMA

    return dsl, usl, psl


def sonrig(d_R, u_R, p_R, c_R):
    """
    to compute right SONIC state PSR, USR, DSR
    """
    usr = G6*u_R - c_R*G5
    csr = -usr
    dsr = d_R * (csr/c_R)**G4
    psr = p_R * (dsr/d_R)**GAMMA

    return dsr, usr, psr