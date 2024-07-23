import numpy as np
from exact_func import inputfile_read, gamma_constants_calculate, initial_conditions_set, boundary_conditions_set, cfl_conditions_impose, godunov_flux_compute, update, output_to_file




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
density, velocity, pressure, conserved_var = initial_conditions_set(n_cells, diaphragm_position1, dx, d_initial_L, u_initial_L, p_initial_L, d_initial_R, u_initial_R, p_initial_R, g8)

# t=0
time = 0
time_difference_tolerance = 1e-06
for n in range(max_timesteps):
    # set coundary conditions BCONDI
    density, velocity, pressure = boundary_conditions_set(density, velocity, pressure, boundary_L, boundary_R)

    # impose courant cfl condition CFLCON
    dt, time, sound_speed = cfl_conditions_impose(n_cells, dx, courant, density, velocity, pressure, n, time, gamma, output_time)

    # compute intercell fluxes based on method chosen RPGODU
    fluxes = godunov_flux_compute(n_cells, gamma, density, velocity, pressure, sound_speed, g8)

    # update solution with conservative (godsunov?) UPDATE
    conserved_var, density, velocity, pressure = update(n_cells, conserved_var, fluxes, dt, dx, density, velocity, pressure, g8)

    # #check if output needed (aka if at the end on the time limit utput?)
    time_difference = np.abs(time - output_time)

    if n%10 == 0:
        print(f"{n:} {time:14.6f} {output_time:14.6f}")

    if time_difference < time_difference_tolerance:
        output_to_file(n_cells, dx, density, velocity, pressure, g8)
        break
