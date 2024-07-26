from configparser import ConfigParser
import numpy as np
import os
import time
from riemann_functions.global_variables import GAMMA, G8, COURANT, MAX_TIMESTEPS, PRESSURE_SCALING_FACTOR
from riemann_functions.exact_riemann_solver import exact_riemann_solver
from riemann_functions.godunov_approximate_solver import godunov_approximate_riemann_solver
from riemann_functions.laxfried_solver import laxfriedriechs_solver
from riemann_functions.godunov_roe_solver import godunov_roe_solver
from riemann_functions.godunov_osher_solver import godunov_osher_solver


def main_riemann_solver(problem_type, solver, output_time, n_cells, input_file="riemann.ini"):
    print(f"Solving problem {problem_type} with solver {solver}, output time {output_time} and n of cells {n_cells}")

    # read file
    domain_length, diaphragm_position, d_initial_L, u_initial_L, p_initial_L, \
        d_initial_R, u_initial_R, p_initial_R, boundary_L, boundary_R = inputfile_read(path=input_file, problem_type=problem_type)

    output_filename = f"solver{solver}_t{output_time:.3f}_n{n_cells}"
   
    # set initial conditions
    dx = domain_length/n_cells # costant mesh size

    # measure runtime
    start_runtime = time.time()

    if solver == 0: # exact riemann
            start_runtime = time.time()
            density, velocity, pressure = exact_riemann_solver(n_cells, d_initial_L, u_initial_L, p_initial_L, d_initial_R, u_initial_R, p_initial_R, dx, diaphragm_position, output_time)
    
    else: # an approximate solver with godunov
        # set initial conditions
        density, velocity, pressure, conserved_var = initial_conditions_set(n_cells, diaphragm_position, dx, d_initial_L, u_initial_L, p_initial_L, d_initial_R, u_initial_R, p_initial_R)

        simulation_time = 0
        simulation_time_diff_tolerance = 1e-06
        for n in range(MAX_TIMESTEPS):
            # set boundary conditions
            density, velocity, pressure = boundary_conditions_set(density, velocity, pressure, boundary_L, boundary_R)

            # impose COURANT cfl condition
            dt, simulation_time, sound_speed = cfl_conditions_impose(n_cells, dx, COURANT, density, velocity, pressure, n, simulation_time, output_time)

            # compute intercell fluxes based on method chosen
            if solver == 1:
                fluxes = godunov_approximate_riemann_solver(n_cells, density, velocity, pressure, sound_speed)
            elif solver == 2:
                fluxes = godunov_roe_solver(n_cells, density, velocity, pressure, sound_speed, conserved_var, dt, dx)
            elif solver == 3:
                fluxes = godunov_osher_solver(n_cells, density, velocity, pressure, sound_speed, conserved_var)
            elif solver == 4:
                fluxes = laxfriedriechs_solver(n_cells, density, velocity, pressure, sound_speed, conserved_var, dx, dt)

            # update solution with conservative godunov
            conserved_var, density, velocity, pressure = update(n_cells, conserved_var, fluxes, dt, dx, density, velocity, pressure)

            # check if output needed (aka if at the end on the time limit utput?)
            simulation_time_diff = np.abs(simulation_time - output_time)

            if simulation_time_diff < simulation_time_diff_tolerance:
                break

    # save the runtime
    end_runtime = time.time()
    runtime = end_runtime - start_runtime
    runtime_elapsed= runtime

    # output the solution
    output_to_file(n_cells, dx, density, velocity, pressure, folder_path=f"output/{problem_type}", filename=f"{output_filename}.out") 

    # output the solution runtime
    output_to_file_stats(runtime_elapsed, folder_path=f"output/{problem_type}", filename=f"{output_filename}_stats.out") 

    print(f"Done, runtime: {runtime_elapsed}\n")


def inputfile_read(path, problem_type: str):
    # read file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, path)
    config = ConfigParser()
    config.read(file_path)

    # assign to variables
    domain_length = float(config[problem_type]["domain_length"])
    diaphragm_position = float(config[problem_type]["diaphragm_position"])
    d_initial_L = float(config[problem_type]["d_L"])
    u_initial_L = float(config[problem_type]["u_L"])
    p_initial_L = float(config[problem_type]["p_L"])
    d_initial_R = float(config[problem_type]["d_R"])
    u_initial_R = float(config[problem_type]["u_R"])
    p_initial_R = float(config[problem_type]["p_R"])
    boundary_L = int(config[problem_type]["boundary_L"])
    boundary_R = int(config[problem_type]["boundary_R"])

    return domain_length, diaphragm_position, d_initial_L, u_initial_L, p_initial_L, \
       d_initial_R, u_initial_R, p_initial_R, boundary_L, boundary_R


def initial_conditions_set(n_cells, diaphragm_position, dx, d_initial_L, u_initial_L, p_initial_L, d_initial_R, u_initial_R, p_initial_R):
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


def boundary_conditions_set(density, velocity, pressure, boundary_L, boundary_R):
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


def cfl_conditions_impose(n_cells, dx, COURANT, density, velocity, pressure, n, time, output_time):
    """
    COURANT-Friedrichs-Lewy (CFL) condition to determine a stable time step size (dt) for a numerical simulation
    - at each iteration it recalculates a good time step
    """
    # dt i guess recalculated at every iteration
    S_max = 0

    sound_speed = np.zeros(n_cells+2)

    # find S max
    for i in range(n_cells+2):
        sound_speed[i] = np.sqrt(GAMMA*pressure[i] / density[i])
        S_current = np.abs(velocity[i]) + sound_speed[i]
        if S_current > S_max:
            S_max = S_current
    
    dt = COURANT * dx / S_max

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    folder_path = os.path.join(parent_dir, folder_path)

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, filename)

    with open(file_path, 'w') as file:
        for i in range(1, n_cells + 1):
            xpos = (i - 0.5) * dx
            energy = pressure[i] / density[i] / G8 / PRESSURE_SCALING_FACTOR
            file.write(f"{xpos:14.6f} {density[i]:14.6f} {velocity[i]:14.6f} {pressure[i] / PRESSURE_SCALING_FACTOR:14.6f} {energy:14.6f}\n")



def output_to_file_stats(runtime, folder_path, filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    folder_path = os.path.join(parent_dir, folder_path)
    file_path = os.path.join(script_dir, folder_path, filename)

    with open(file_path, 'w') as file:
        file.write(f"{runtime:14.6f}\n")