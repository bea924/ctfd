from riemann_functions.main_riemann_functions import main_riemann_solver


##################################
######### USER INPUT #############
##################################
# Choose the problem type: ModifiedSod (0.2), Test2 (0.15), Test3 (0.012), Test5 (0.012)
problem_type = "ModifiedSod"
# choose the solver: ExactRiemann [0], ExactRiemannAppr [1], LaxFriedrichs [2], Roe [3], Osher [4]
solver = 1
output_time = 0.2
n_cells = 100
##################################
##################################

main_riemann_solver(problem_type, solver, output_time, n_cells)


# for solver in [0, 2, 3, 4]:
#     for n_cells in [100, 200, 300, 400, 500, 600, 700, 1000, 1500, 2000]:
#         main_riemann_solver(problem_type, solver, output_time, n_cells)

# for output_time in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
#     main_riemann_solver(problem_type, solver, output_time, n_cells)