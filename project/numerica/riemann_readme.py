from riemann_functions.main_riemann_functions import main_riemann_solver
from plotting.plot_solutions import plot_runtime, plot_2d_solution, plot_solution_validation, plot_solutions_compare, plot_solution_error

####################################################################
################### EXAMPLE USER INPUT #############################
####################################################################
# Choose the problem type: ModifiedSod, Test2, Test3, Test5
problem_type = "ModifiedSod"
# choose the solver: ExactRiemann [0], LaxFriedrichs [1], Roe [2], Osher [3], ExactRiemannAppr [4],
solver = 2
# Choose the output time: ModifiedSod (0.2), Test2 (0.15), Test3 (0.012), Test5 (0.012)
output_time = 0.2
# 100 for seeing the errors better, 500 for a better solution
n_cells = 100
####################################################################
####################################################################

# main_riemann_solver(problem_type, solver, output_time, n_cells)

plot_solution_error(problem_type, solver, n_cells, output_time)