import numpy as np
import time
from riemann_functions import inputfile_read, initial_conditions_set, boundary_conditions_set, cfl_conditions_impose, update, output_to_file, output_to_file_stats
from godunov_exact_solver import godunov_exact_riemann_solver
from godunov_roe_solver import godunov_roe_solver
from laxfried_solver import laxfriedriechs_solver

##################################
######### USER INPUT #############
##################################
# Choose the problem type: ModifiedSod, StationaryContact
problem_type = "ModifiedSod"
# choose the solver: ExactRiemann [0], LaxFriedrichs [1], Roe [2]
solver = 2
output_time=0.2
##################################
##################################


input_file = "riemann.ini"
# input_file = "project/numerica/riemann.ini" # for debugger

# read file
domain_length, diaphragm_position, n_cells, gamma, d_initial_L, u_initial_L, p_initial_L, \
       d_initial_R, u_initial_R, p_initial_R, courant, boundary_L, boundary_R, output_frequency, max_timesteps, pressure_scaling_factor = inputfile_read(path=input_file, problem_type=problem_type)

output_filename = f"solver{solver}_t{output_time:.2f}"
runtime_elapsed = []

# calculate some stuff
dx = domain_length/n_cells # costant mesh size

# set initial conditions
density, velocity, pressure, conserved_var = initial_conditions_set(n_cells, diaphragm_position, dx, d_initial_L, u_initial_L, p_initial_L, d_initial_R, u_initial_R, p_initial_R)


simtime = 0
time_difference_tolerance = 1e-06
for n in range(max_timesteps):
    start_runtime = time.time()
    # set coundary conditions BCONDI
    density, velocity, pressure = boundary_conditions_set(density, velocity, pressure, boundary_L, boundary_R)

    # impose courant cfl condition CFLCON
    dt, simtime, sound_speed = cfl_conditions_impose(n_cells, dx, courant, density, velocity, pressure, n, simtime, output_time)

    # compute intercell fluxes based on method chosen RPGODU
    if solver == 0:
        fluxes = godunov_exact_riemann_solver(n_cells, density, velocity, pressure, sound_speed)
    elif solver == 1:
        fluxes = laxfriedriechs_solver(n_cells, density, velocity, pressure, sound_speed, conserved_var, dx, dt)
    elif solver == 2:
        fluxes = godunov_roe_solver(n_cells, density, velocity, pressure, sound_speed, conserved_var, dt, dx)

    # update solution with conservative (godsunov?) UPDATE
    conserved_var, density, velocity, pressure = update(n_cells, conserved_var, fluxes, dt, dx, density, velocity, pressure)

    # save the runtime
    end_runtime = time.time()
    runtime = end_runtime - start_runtime
    runtime_elapsed.append(runtime)

    # check if output needed (aka if at the end on the time limit utput?)
    time_difference = np.abs(simtime - output_time)

    if n%10 == 0:
        print(f"{n:} {simtime:14.6f} {output_time:14.6f}")

    if time_difference < time_difference_tolerance:
        # output the solution
        output_to_file(n_cells, dx, density, velocity, pressure, folder_path=f"output/{problem_type}", filename=f"{output_filename}.out") 
        # output_to_file(n_cells, dx, density, velocity, pressure, path=f"project/numerica/output/{output_filename}.out") # for debugger

        # output the solution stats
        output_to_file_stats(runtime_elapsed, path=f"output/{problem_type}/{output_filename}_stats.out") 
        break
