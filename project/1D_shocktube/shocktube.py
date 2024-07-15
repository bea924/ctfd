import numpy as np

# initial conditions

# velocity
U_left = 0.0
U_right = 0.0

# pressure
P_left = 100.0
P_right = 1.0

# density
rho_left = 10.0
rho_right = 1.0

# constants
gamma = 1.4
# consider a := (gamma - 1)
# b := (gamma + 1)
a = gamma - 1
b = gamma + 1

# functions

def A(rho):
    return 2.0 / (b*rho)

def B(P):
    return a/b * P

#input
#P_curr: Pressure you are at right now
#P_left: Pressure on the left side of the tube
#P_right: Pressure on the right hand side of the tube
#gamma: Specific heat of the air
#rho: Density of the air (place either right or left) rho_right or rho_left
#output
#f_L and f_R
def f(P_curr, P_left, P_right, gamma, rho_left, rho_right):
    #f left side
    A_L = A(rho_left)
    B_L = B(P_left)
    
    if(P_curr > P_left): #shock
        AB_L = np.sqrt(A_L/(P_curr + B_L))
        f_L = (P_curr - P_left)*AB_L
    else: #rarefaction
        p_ratio = (P_curr/P_left)**(a/(2*gamma))
        f_L = 2*A_L/a * (p_ratio - 1)
    
    A_R = A(rho_right)
    B_R = B(P_right)
    
    if(P_curr > P_right): #shock
        AB_R = np.sqrt(A_R/(P_curr + B_R))
        f_R= (P_curr - P_right)*AB_R
    else: #rarefaction
        p_ratio = (P_curr/P_right)**(a/(2*gamma))
        f_R = 2*A_R/a * (p_ratio - 1)
        
    return f_L, f_R