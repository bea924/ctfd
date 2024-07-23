from configparser import ConfigParser
import numpy as np


def inputfile_read(path: str = "project/numerica/exact.ini"):
    # read file
    config = ConfigParser()
    config.read(path)

    # assign to variables
    domain_length = float(config["General"]["domain_length"])
    diaphragm_position1 = float(config["General"]["diaphragm_position1"])
    n_cells = int(config["General"]["n_cells"])
    gamma = float(config["Constants"]["gamma"])
    output_time = float(config["General"]["output_time"])
    d_initial_L = float(config["Variables"]["d_L"])
    u_initial_L = float(config["Variables"]["u_L"])
    p_initial_L = float(config["Variables"]["p_L"])
    d_initial_M = float(config["Variables"]["d_M"])
    u_initial_M = float(config["Variables"]["u_M"])
    p_initial_M = float(config["Variables"]["p_M"])
    d_initial_R = float(config["Variables"]["d_R"])
    u_initial_R = float(config["Variables"]["u_R"])
    p_initial_R = float(config["Variables"]["p_R"])
    diaphragm_position2 = float(config["General"]["diaphragm_position2"])
    courant = float(config["Constants"]["courant"])
    boundary_L = int(config["Constants"]["boundary_L"])
    boundary_R = int(config["Constants"]["boundary_R"])
    output_frequency = int(config["General"]["output_frequency"])
    max_timesteps = int(config["General"]["max_timesteps"])
    pressure_scaling_factor = float(config["General"]["pressure_scaling_factor"])

    return domain_length, diaphragm_position1, n_cells, gamma, output_time, d_initial_L, u_initial_L, p_initial_L, d_initial_M, u_initial_M, p_initial_M, \
       d_initial_R, u_initial_R, p_initial_R, diaphragm_position2,courant,boundary_L,boundary_R,output_frequency,max_timesteps,pressure_scaling_factor


def gamma_constants_calculate(gamma):
    g1 = (gamma - 1.0)/(2.0*gamma)
    g2 = (gamma + 1.0)/(2.0*gamma)
    g3 = 2.0*gamma/(gamma - 1.0)
    g4 = 2.0/(gamma - 1.0)
    g5 = 2.0/(gamma + 1.0)
    g6 = (gamma - 1.0)/(gamma + 1.0)
    g7 = (gamma - 1.0)/2.0
    g8 = gamma - 1.0
    return g1, g2, g3, g4, g5, g6, g7, g8


def initial_conditions_set(n_cells, diaphragm_position1): # INITIA
    """
    set initial conditions
     - initialize state variables for each cell (d, u, p) and the conserved variabless
    """

    density = np.zeros(n_cells+2)
    velocity = np.zeros(n_cells+2)
    pressure = np.zeros(n_cells+2)
    conserved_var = np.zeros((3, n_cells+2))

    for i in range(1, n_cells+1):
        position_x = (i-0.5)*dx
        if position_x < diaphragm_position1:
            density[i] = d_initial_L
            velocity[i] = u_initial_L
            pressure[i] = p_initial_L
        else:
            density[i] = d_initial_R
            velocity[i] = u_initial_R
            pressure[i] = p_initial_R

        conserved_var[0,i] = density[i]
        conserved_var[1,i] = density[i] * velocity[i]
        conserved_var[2,i] = 0.5 * (density[i]*velocity[i]) * velocity[i] + pressure[i]/g8
    
    return density, velocity, pressure, conserved_var


def boundary_conditions_set(density, velocity, pressure, boundary_L, boundary_R): # BCONDI
    """
    set boundary conditions
    - for d u p at left and right
    - the first and last cells are the border
    """
    if boundary_L: # 0 = transmissive
        density[0] = density[1] # not sure if the dimension is n cells or 3000???
        velocity[0] = velocity[1]
        pressure[0] = pressure[1] 
    else: # 1 = reflective
        density[0] = density[1] 
        velocity[0] = -velocity[1]
        pressure[0] = pressure[1]
    
    if boundary_R: # 0 = transmissive
        density[-1] = density[-2]
        velocity[-1] = velocity[-2]
        pressure[-1] = pressure[-2]
    else: # 1 = reflective
        density[-1] = density[-2] 
        velocity[-1] = -velocity[-2]
        pressure[-1] = pressure[-2]

    return density, velocity, pressure


def cfl_conditions_impose(n_cells, courant, density, velocity, pressure, t, time): # cflcon
    """
    Courant-Friedrichs-Lewy (CFL) condition to determine a stable time step size (dt) for a numerical simulation
    - at each iteration it recalculates a good time step
    """
    # dt i guess recalculated at every iteration
    S_max = 0

    sound_speed = np.zeros(n_cells+2)

    # find S max
    for i in range(n_cells+2):
        sound_speed[i] = np.sqrt(gamma*pressure[i]) / density[i]
        S_current = np.abs(velocity[i]) + sound_speed[i]
        if S_current > S_max:
            S_max = S_current
    
    dt = courant * dx / S_max

    # compensate for the approximate calculation of S_MAX in early steps
    if t <= 5:
        dt = 0.2*dt

    # avoid dt going over max time overshoot
    if (time + dt) > output_time:
        dt = output_time - time
    
    time = time + dt

    return dt, time, sound_speed


def godunov_flux_compute(n_cells): # for now only exact riemann RPGODU
    """
    compute godunov intercell flux using the chose solver (for now only exact riemann)
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

        # calls exact riemann solver, get star region values
        pm, um = exact_riemann_solver(d_local_L, u_local_L, p_local_L, c_local_L, d_local_R, u_local_R, p_local_R, c_local_R)
        
        # calls sample for x_over_t=0
        # get godunov state values
        dsam, usam, psam = sample(x_over_t, um, pm, d_local_L, u_local_L, p_local_L, c_local_L, d_local_R, u_local_R, p_local_R, c_local_R)
            
        # calculate intercell flux 1 2 3
        fluxes[0, i] = dsam * usam
        fluxes[1, i] = dsam * usam * usam + psam
        energy = 0.5 * usam * usam * dsam + psam/g8
        fluxes[2, i] = usam * (energy + psam)

    return fluxes



def update(conserved_var, fluxes, dt, dx, density, velocity, pressure):
    # to update the solution according to the conservative formula and compute physical variables
    for i in range(1, n_cells+1):
        conserved_var[0, i] = conserved_var[0, i] + dt/dx * (fluxes[0, i-1] - fluxes[0, i])
        conserved_var[1, i] = conserved_var[1, i] + dt/dx * (fluxes[1, i-1] - fluxes[1, i])
        conserved_var[2, i] = conserved_var[2, i] + dt/dx * (fluxes[2, i-1] - fluxes[2, i])

    for i in range(1, n_cells+1):
        density[i] = conserved_var[0, i]
        velocity[i] = conserved_var[1, i] / density[i]
        pressure[i] = g8 *(conserved_var[2, i] - 0.5 * conserved_var[1,i]*velocity[i])

    return conserved_var, density, velocity, pressure




#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
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
            pq = (p_local_L/p_local_R)**g1
            um =  (pq * u_local_L/c_local_L + u_local_R/c_local_R + g4 * (pq - 1.0)) / (pq/c_local_L + 1.0/c_local_R)
            ptl = 1.0 + g7 * (u_local_L - um)/c_local_L
            ptr = 1.0 + g7 * (um - u_local_R)/c_local_R
            pm_start_guess = 0.5 * (p_local_L * ptl**g3 + p_local_R*ptr**g3)
        else:
            #select two-shock Riemann solver
            gel = np.sqrt( (g5/d_local_L) / (g6*p_local_L + ppv) )
            ger = np.sqrt( (g5/d_local_R) / (g6*p_local_R + ppv) )
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
        f = g4 * c_K * (p_ratio**g1 - 1)
        fd = (1.0 / (d_K * c_K)) * p_ratio**(-g2)
    else:
        # shock wave
        a_K = g5 / d_K
        b_K = g6 * p_K
        qrt = np.sqrt(a_K / (b_K + p))
        f = (p - p_K) * qrt
        fd = (1.0 - 0.5*(p - p_K)/(b_K + p)) * qrt

    return f, fd


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def sample(s, um, pm, d_L, u_L, p_L, c_L, d_R, u_R, p_R, c_R):
    """
    sample the solution throughout the wave pattern resulting from a Riemann problem
    determines the state variables (density, velocity, and pressure) at a given position S in the flow based on the computed pressure and velocity in the star region.
    """
    # samples the solution throght the wave
    # input s pm um gamma s = x/t
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
    

def output_to_file(path:str = "project/numerica/exact.out"):
    ps_scale = 1
    with open(path, 'w') as file:
        for i in range(1, n_cells + 1):
            xpos = (i - 0.5) * dx
            energy = pressure[i] / density[i] / g8 / ps_scale
            file.write(f"{xpos:14.6f} {density[i]:14.6f} {velocity[i]:14.6f} {pressure[i] / ps_scale:14.6f} {energy:14.6f}\n")






    


##################################################################################################################################
# EXECUTOR
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# read file
domain_length, diaphragm_position1, n_cells, gamma, output_time, d_initial_L, u_initial_L, p_initial_L, d_initial_M, u_initial_M, p_initial_M, \
       d_initial_R, u_initial_R, p_initial_R, diaphragm_position2, courant,boundary_L, boundary_R, output_frequency, max_timesteps, pressure_scaling_factor = inputfile_read()

# calculate some stuff
g1, g2, g3, g4, g5, g6, g7, g8 = gamma_constants_calculate(gamma)
dx = domain_length/n_cells # costant mesh size

# set initial conditions
density, velocity, pressure, conserved_var = initial_conditions_set(n_cells, diaphragm_position1)

# t=0
time = 0
time_difference_tolerance = 1e-06
for t in range(max_timesteps):
    # set coundary conditions BCONDI
    density, velocity, pressure = boundary_conditions_set(density, velocity, pressure, boundary_L, boundary_R)

    # impose courant cfl condition CFLCON
    dt, time, sound_speed = cfl_conditions_impose(n_cells, courant, density, velocity, pressure, t, time)

    # compute intercell fluxes based on method chosen RPGODU
    fluxes = godunov_flux_compute(n_cells)

    # update solution with conservative (godsunov?) UPDATE
    conserved_var, density, velocity, pressure = update(conserved_var, fluxes, dt, dx, density, velocity, pressure)

    # #check if output needed (aka if at the end on the time limit utput?)
    time_difference = np.abs(time - output_time)
    if time_difference < time_difference_tolerance:
        print("done")
        output_to_file()
        break
