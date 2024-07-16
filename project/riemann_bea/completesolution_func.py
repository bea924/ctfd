import numpy as np
import sys


def main_riemann_solver(d_L, u_L, p_L, d_R, u_R, p_R, gamma, mpa, domain_length, n_cells, diaph, timeout):
    # mpa: normalising constant
    # diaph: initial discontinuity position
    # n_cells: number of computing cells
    # timeout: output time

    # compute gamma related constants (g1 etc)
    global g1, g2, g3, g4, g5, g6, g7, g8
    # compute gamma constants
    g1 = (gamma - 1.0)/(2.0*gamma)
    g2 = (gamma + 1.0)/(2.0*gamma)
    g3 = 2.0*gamma/(gamma - 1.0)
    g4 = 2.0/(gamma - 1.0)
    g5 = 2.0/(gamma + 1.0)
    g6 = (gamma - 1.0)/(gamma + 1.0)
    g7 = (gamma - 1.0)/2.0
    g8 = gamma - 1.0
    
    # compute sound speeds
    c_L = np.sqrt(gamma*p_L/d_L)
    c_R = np.sqrt(gamma*p_R/d_R)

    # test if there is a vacuum not yet implemented (and if yes stop)
    if (g4*(c_L+c_R) <= (u_R-u_L)):
        sys.exit(1)
    # stop

    # find exact solution
    p_m, u_m = starpu(d_L, u_L, p_L, c_L, d_R, u_R, p_R, c_R, gamma)
    dx = domain_length/n_cells

    # create an array
    output_array = np.zeros((n_cells, 4))  # each cell contains 5 values

    # find each cell's d u p values
    for i in range(n_cells):
        x_pos = (i-0.5)*dx
        s = (x_pos-diaph)/timeout
        d, u, p = sample(s, u_m, p_m, gamma, d_L, u_L, p_L, c_L, d_R, u_R, p_R, c_R)
        output_array[i, 0] = x_pos
        output_array[i, 1] = d
        output_array[i, 2] = u
        output_array[i, 3] = p/mpa
        # output_array[i, 4] = p/d/g8/mpa
        print("pressure:" + str(output_array[i, 3]))

    return output_array


def starpu(d_L, u_L, p_L, c_L, d_R, u_R, p_R, c_R, gamma, maxiter=30, tolerance=1e-6):
    # compute solution for pressure and velocity in star region
    p_start = guessp(d_L, d_R, c_L, c_R, p_L, p_R, u_L, u_R)
    
    p_old = p_start
    u_diff = u_R - u_L

    it = 0
    tol = 1
    while (it < maxiter and tol > tolerance):
        f_L, f_Ld = prefun(p_old, p_L, c_L, d_L, gamma)
        f_R, f_Rd = prefun(p_old, p_R, c_R, d_R, gamma)
        p_new = p_old - (f_L + f_R + u_diff) / (f_Ld + f_Rd)
        tol = 2.0 * np.abs((p_new - p_old) / (p_new + p_old))
        p_old = p_new

    u = 0.5*(u_L + u_R + f_R - f_L)  # speed

    return p_old, u

def guessp(d_L, d_R, c_L, c_R, p_L, p_R, u_L, u_R):
    # provide a guess value for the pressure PM in the star regions
    # according to adaptive Reinmann solver
    quser = 2.0
    
    #compute guess pressure from PVRS
    cup = 0.25*(d_L + d_R)*(c_L + c_R)
    ppv = 0.5*(p_L + p_R) + 0.5*(u_L - u_R)*cup
    ppv = np.maximum(0.0, ppv)
    pmin = min(p_L, p_R)
    pmax = max(p_L, p_R)
    qmax = pmax/pmin
    
    if (qmax <= quser and pmin <= ppv <= pmax):
        #select PVRS
        pm = ppv
    elif (ppv < pmin):
        #select Two-Rarefaction Reimann solver
        pq = (p_L/p_R)**g1
        um =  (pq * u_L/c_L + u_R/c_R + g4 * (pq - 1.0)) / (pq/c_L + 1.0/c_R)
        ptl = 1.0 + g7 * (u_L - um)/c_L
        ptr = 1.0 + g7 * (um - u_R)/c_R
        pm = 0.5 * (p_L * ptl**g3 + p_R*ptr**g3)
    else:
        #select two-shock Reinamnn solver
        gel = np.sqrt( (g5/d_L) / (g6*p_L + ppv) )
        ger = np.sqrt( (g5/d_R) / (g6*p_R + ppv) )
        pm = (gel*p_L + ger*p_R - (u_R -u_L)) / (gel + ger) 
    
    return pm

def prefun(p, p_K, c_K, d_K, gamma):
    # evaluates the pressure functions fl or fr and return f and f derivative
    # selects if raref or shock

    # global g1, g2, g3, g4, g5, g6, g7, g8
    # # compute gamma constants
    # g1 = (gamma - 1.0)/(2.0*gamma)
    # print("g1 is:" +str(g1))
    # g2 = (gamma + 1.0)/(2.0*gamma)
    # g3 = 2.0*gamma/(gamma - 1.0)
    # g4 = 2.0/(gamma - 1.0)
    # g5 = 2.0/(gamma + 1.0)
    # g6 = (gamma - 1.0)/(gamma + 1.0)
    # g7 = (gamma - 1.0)/2.0
    # g8 = gamma - 1.0

    if (p <= p_K):
        # raref
        prat = p/p_K
        f = g4*c_K*(prat**g1 - 1)
        fd = (1.0/(d_K*c_K))*prat**(-g2)
    else:
        # shock wave
        a_K = g5/d_K
        b_K = g6*p_K
        qrt = np.sqrt(a_K/(b_K+p))
        f = (p-p_K)*qrt
        fd = (1.0 - 0.5*(p - p_K)/(b_K + p))*qrt

    return f, fd


def sample(s, um, pm, gamma, d_L, u_L, p_L, c_L, d_R, u_R, p_R, c_R):
    # samples the solution throght the wave
    # input s pm um gamma s = x/t
    # output d u p

    if (s <= um):
        # sample left of the contact discontinuity
        if (pm <= p_L):
            # left rarefaction
            sh_L = u_L-c_L
            if (s <= sh_L):
                # sampled point is left data state
                d = d_L
                u = u_L
                p = p_L
            else:
                cm_L = c_L*(pm/p_L)**g1
                stl = um-cm_L
                if (s > stl):
                    #  Sampled point is Star Left state
                    d = d_L*(pm/p_L)**(1.0/gamma)
                    u = um
                    p = pm
                else:
                    # Sampled point is inside left fan
                    u = g5*(c_L + g7 * u_L + s)
                    c = g5*(c_L + g7 * (u_L - s))
                    d = d_L*(c/c_L)**g4
                    p = p_L * (c/c_L)**g3
        else:
            # left shock
            pm_L = pm/p_L
            sl = u_L - c_L*np.sqrt(g2 * pm_L + g1)
            if (s <= sl):
                # sampled point is left data state
                d = d_L
                u = u_L
                p = p_L
            else:
                #  Sampled point is Star Left stat
                d = d_L*(pm_L + g6)/(pm_L*g6 + 1) 
                u = um
                p = pm
    else:
        # sample right of the contact discontinuity
        if (pm > p_R):
            # left shock
            pm_R = pm/p_R
            s_R = u_R + c_R*np.sqrt(g2*pm_R + g1)
            if (s >= s_R):
                # sampled point is right data state
                d = d_R
                u = u_R
                p = p_R
            else:
                # Sampled point is Star Right state
                d = d_R*(pm_R+g6)/(pm_R*g6 + 1.0)
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
                cm_R = c_R*(pm/p_R)**g1
                st_R = um + cm_R
                if (s <= st_R):
                    #  Sampled point is Star Right stat
                    d = d_R*((pm/p_R)**(1.0/gamma))
                    u = um
                    p = pm
                else:
                    #  Sampled point is inside left fan
                    u = g5*(-c_R + g7*u_R + s)
                    c = g5*(c_R - g7*(u_R - s))
                    d = d_R*(c/c_R)**g4
                    p = p_R*(c/c_R)**g3

    return d, u, p