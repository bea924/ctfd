from riemann_functions.main_riemann_functions import main_riemann_solver

####################################################################
################### EXAMPLE USER INPUT #############################
####################################################################
# Choose the problem type: ModifiedSod, Test2, Test3, Test5
problem_type = "ModifiedSod"
# choose the solver: ExactRiemann [0], LaxFriedrichs [1], Roe [2], Osher [3], Roe Entropy Fix [4]
solver = 0
# Choose the output time, suggestions: ModifiedSod (0.2), Test2 (0.15), Test3 (0.012), Test5 (0.012)
output_time = 0
# 100 for seeing the errors better, 500 for a better solution
n_cells = 100
####################################################################
####################################################################


# # comparison of the different solvers for n_cells 500 and 100
# problem_type = "ModifiedSod"
# output_time = 0.2
# for solver in [0, 1, 2, 3, 4]:
#     for n_cells in [100, 200, 300, 400, 500, 700, 1000, 1500, 2000]:
#         main_riemann_solver(problem_type, solver, output_time, n_cells)

# problem_type = "Test2"
# output_time = 0.15
# for solver in [0, 1, 2, 3, 4]:
#     for n_cells in [100, 500, 1000, 2000]:
#         main_riemann_solver(problem_type, solver, output_time, n_cells)

# # problem_type = "Test3"
# # output_time = 00.012
# # for solver in [0, 1, 2, 3, 4]:
# #     for n_cells in  [500, 500, 1000, 2000]:
# #         main_riemann_solver(problem_type, solver, output_time, n_cells)

# problem_type = "Test5"
# output_time = 0.012
# for solver in [0, 1, 2, 3, 4]:
#     for n_cells in  [100, 500, 1000, 2000]:
#         main_riemann_solver(problem_type, solver, output_time, n_cells)


# stability
problem_type = "ModifiedSod"
n_cells =500
for solver in [0, 1, 2, 3, 4]:
    for output_time in [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2]:
        main_riemann_solver(problem_type, solver, output_time, n_cells)