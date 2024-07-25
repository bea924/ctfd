from configparser import ConfigParser
import numpy as np
import os
from global_variables import GAMMA, G8


def inputfile_read(path, problem_type: str):
    # read file
    config = ConfigParser()
    config.read(path)

    # assign to variables
    domain_length = float(config[problem_type]["domain_length"])
    diaphragm_position = float(config[problem_type]["diaphragm_position"])
    n_cells = int(config[problem_type]["n_cells"])
    GAMMA = float(config[problem_type]["GAMMA"])
    output_time = float(config[problem_type]["output_time"])
    d_initial_L = float(config[problem_type]["d_L"])
    u_initial_L = float(config[problem_type]["u_L"])
    p_initial_L = float(config[problem_type]["p_L"])
    d_initial_R = float(config[problem_type]["d_R"])
    u_initial_R = float(config[problem_type]["u_R"])
    p_initial_R = float(config[problem_type]["p_R"])
    courant = float(config[problem_type]["courant"])
    boundary_L = int(config[problem_type]["boundary_L"])
    boundary_R = int(config[problem_type]["boundary_R"])
    output_frequency = int(config[problem_type]["output_frequency"])
    max_timesteps = int(config[problem_type]["max_timesteps"])
    pressure_scaling_factor = float(config[problem_type]["pressure_scaling_factor"])

    return domain_length, diaphragm_position, n_cells, GAMMA, d_initial_L, u_initial_L, p_initial_L, \
       d_initial_R, u_initial_R, p_initial_R, courant, boundary_L, boundary_R, output_frequency, max_timesteps, pressure_scaling_factor


def initial_conditions_set(n_cells, diaphragm_position, dx, d_initial_L, u_initial_L, p_initial_L, d_initial_R, u_initial_R, p_initial_R): # INITIA
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
        conserved_var[2,i] = 0.5 * (density[i]*velocity[i]) * velocity[i] + pressure[i]/G8
    
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


def cfl_conditions_impose(n_cells, dx, courant, density, velocity, pressure, n, time, output_time): # cflcon
    """
    Courant-Friedrichs-Lewy (CFL) condition to determine a stable time step size (dt) for a numerical simulation
    - at each iteration it recalculates a good time step
    """
    # dt i guess recalculated at every iteration
    S_max = 0

    sound_speed = np.zeros(n_cells+2)

    # find S max
    for i in range(n_cells+2):
        # sound_speed[i] = np.sqrt(GAMMA*pressure[i]) / density[i] IT WAS LIKE THIS BEFORE AND STILL WORKED!!!
        sound_speed[i] = np.sqrt(GAMMA*pressure[i] / density[i])
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


def update(n_cells, conserved_var, fluxes, dt, dx, density, velocity, pressure):
    # to update the solution according to the conservative formula and compute physical variables
    for i in range(1, n_cells+1):
        conserved_var[0, i] = conserved_var[0, i] + dt/dx * (fluxes[0, i-1] - fluxes[0, i])
        conserved_var[1, i] = conserved_var[1, i] + dt/dx * (fluxes[1, i-1] - fluxes[1, i])
        conserved_var[2, i] = conserved_var[2, i] + dt/dx * (fluxes[2, i-1] - fluxes[2, i])

    for i in range(1, n_cells+1):
        density[i] = conserved_var[0, i]
        velocity[i] = conserved_var[1, i] / density[i]
        pressure[i] = G8 *(conserved_var[2, i] - 0.5 * conserved_var[1,i]*velocity[i])

    return conserved_var, density, velocity, pressure
   

def output_to_file(n_cells, dx, density, velocity, pressure, folder_path, filename):
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Now you can safely create your output file in this folder
    file_path = os.path.join(folder_path, filename)

    ps_scale = 1
    with open(file_path, 'w') as file:
        for i in range(1, n_cells + 1):
            xpos = (i - 0.5) * dx
            energy = pressure[i] / density[i] / G8 / ps_scale
            file.write(f"{xpos:14.6f} {density[i]:14.6f} {velocity[i]:14.6f} {pressure[i] / ps_scale:14.6f} {energy:14.6f}\n")



def output_to_file_stats(runtime, path):
    ps_scale = 1
    with open(path, 'w') as file:
        for i in range(len(runtime)):
            file.write(f"{runtime[i]:14.6f}\n")