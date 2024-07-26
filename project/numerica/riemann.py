from riemann_functions import main_riemann_solver


##################################
######### USER INPUT #############
##################################
# Choose the problem type: ModifiedSod, StationaryContact
problem_type = "ModifiedSod"
# choose the solver: ExactRiemann [0], ExactRiemannAppr [1], LaxFriedrichs [2], Roe [3], Osher [4]
solver = 4
output_time=0.2
##################################
##################################


main_riemann_solver(problem_type, solver, output_time)