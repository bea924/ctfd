from configparser import ConfigParser
import numpy as np


def inputfile_read(path):
    # read file
    config = ConfigParser()
    config.read(path)

    # assign to variables
    domain_length = float(config["General"]["domain_length"])
    diaphragm_position = float(config["General"]["diaphragm_position"])
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
    courant = float(config["Constants"]["courant"])
    boundary_L = int(config["Constants"]["boundary_L"])
    boundary_R = int(config["Constants"]["boundary_R"])
    output_frequency = int(config["General"]["output_frequency"])
    max_timesteps = int(config["General"]["max_timesteps"])
    pressure_scaling_factor = float(config["General"]["pressure_scaling_factor"])
    solver = int(config["General"]["solver"])

    return domain_length, diaphragm_position, n_cells, gamma, output_time, d_initial_L, u_initial_L, p_initial_L, d_initial_M, u_initial_M, p_initial_M, \
       d_initial_R, u_initial_R, p_initial_R, courant, boundary_L, boundary_R, output_frequency, max_timesteps, pressure_scaling_factor, solver


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


def initial_conditions_set(n_cells, diaphragm_position, dx, d_initial_L, u_initial_L, p_initial_L, d_initial_R, u_initial_R, p_initial_R, g8): # INITIA
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
        if position_x <= diaphragm_position:
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
    if boundary_L == 0: # 0 = transmissive
        density[0] = density[1]
        velocity[0] = velocity[1]
        pressure[0] = pressure[1] 
    else: # 1 = reflective
        density[0] = density[1] 
        velocity[0] = -velocity[1]
        pressure[0] = pressure[1]
    
    if boundary_R == 0: # 0 = transmissive
        density[-1] = density[-2]
        velocity[-1] = velocity[-2]
        pressure[-1] = pressure[-2]
    else: # 1 = reflective
        density[-1] = density[-2] 
        velocity[-1] = -velocity[-2]
        pressure[-1] = pressure[-2]

    return density, velocity, pressure


def cfl_conditions_impose(n_cells, dx, courant, density, velocity, pressure, n, time, gamma, output_time): # cflcon
    """
    Courant-Friedrichs-Lewy (CFL) condition to determine a stable time step size (dt) for a numerical simulation
    - at each iteration it recalculates a good time step
    """
    # dt i guess recalculated at every iteration
    S_max = 0

    sound_speed = np.zeros(n_cells+2)

    # find S max
    for i in range(n_cells+2):
        # sound_speed[i] = np.sqrt(gamma*pressure[i]) / density[i] IT WAS LIKE THIS BEFORE AND STILL WORKED!!!
        sound_speed[i] = np.sqrt(gamma*pressure[i] / density[i])
        S_current = np.abs(velocity[i]) + sound_speed[i]
        if S_current > S_max:
            S_max = S_current
    
    dt = courant * dx / S_max

    # compensate for the approximate calculation of S_MAX in early steps
    if n <= 5:
        dt = 0.2*dt

    # avoid dt going over max time overshoot
    if (time + dt) > output_time:
        dt = output_time - time
    
    time = time + dt

    return dt, time, sound_speed


def update(n_cells, conserved_var, fluxes, dt, dx, density, velocity, pressure, g8):
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
   

def output_to_file(n_cells, dx, density, velocity, pressure, g8, path):
    ps_scale = 1
    with open(path, 'w') as file:
        for i in range(1, n_cells + 1):
            xpos = (i - 0.5) * dx
            energy = pressure[i] / density[i] / g8 / ps_scale
            file.write(f"{xpos:14.6f} {density[i]:14.6f} {velocity[i]:14.6f} {pressure[i] / ps_scale:14.6f} {energy:14.6f}\n")



def output_to_file_stats(runtime, path):
    ps_scale = 1
    with open(path, 'w') as file:
        for i in range(len(runtime)):
            file.write(f"{runtime[i]:14.6f}\n")