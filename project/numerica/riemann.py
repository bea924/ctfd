from riemann_functions import main_riemann_solver


##################################
######### USER INPUT #############
##################################
# Choose the problem type: ModifiedSod, StationaryContact
problem_type = "ModifiedSod"
# choose the solver: ExactRiemann [0], LaxFriedrichs [1], Roe [2]
solver = 0
output_time=0.41
##################################
##################################


main_riemann_solver(problem_type, solver, output_time)