from riemann_functions.main_riemann_functions import main_riemann_solver
from plotting.plot_solutions import plot_runtime, plot_2d_solution, plot_solution_validation, plot_solutions_compare, plot_solution_error, plot_convergence_spatial, plot_convergence_spatial_comparison, \
plot_convergence_spatial_comparison_toexact, plot_solution_ncells_error, plot_solution_stability

####################################################################
################### EXAMPLE USER INPUT #############################
####################################################################
# Choose the problem type: ModifiedSod, Test2, Test3, Test5, BellCurve
problem_type = "ModifiedSod"
# choose the solver: ExactRiemann [0], LaxFriedrichs [1], Roe [2], Osher [3], Roe entropy [4]
solver = 0
# Choose the output time: ModifiedSod (0.2), Test2 (0.15), Test3 (0.012), Test5 (0.012)
output_time = 0.2
# 100 for seeing the errors better, 500 for a better solution
n_cells = 500
####################################################################
####################################################################

# plot_solution_validation(problem_type, solver, n_cells, output_time)


n_cells_list = [100, 200, 300, 400, 500, 700, 1000, 1500, 2000]
solver_list = [1, 2, 3, 4] 



# NCELL ERROR
n_cells_list = [100, 500, 1000, 2000]
solver_list = [1, 2, 3, 4] 
col_n = 0
# plot_solution_ncells_error(problem_type, solver_list, n_cells_list, output_time, col_n)



# ERROR
solver_list = [1, 2, 3, 4]
n_cells = 500
# plot_solution_error(problem_type, solver_list, n_cells, output_time)




# TIME
solver_list = [0, 1, 4]
n_cells =500
# output_time_list =[0.1, 0.2, 0.3, 0.4, 0.5, 1, 2]
output_time_list =[0.05, 0.1, 0.2, 0.3]
plot_solution_stability(problem_type, solver_list, n_cells, output_time_list)


