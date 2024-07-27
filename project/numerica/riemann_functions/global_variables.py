GAMMA = 1.4 # specific heat ratio
COURANT = 0.9 # courant coefficient
MAX_TIMESTEPS = 10000000
PRESSURE_SCALING_FACTOR = 1

# gamma expressions used frequently (inspired by the code in Toro Ch.4)
G1 = (GAMMA - 1.0)/(2.0*GAMMA)
G2 = (GAMMA + 1.0)/(2.0*GAMMA)
G3 = 2.0*GAMMA/(GAMMA - 1.0)
G4 = 2.0/(GAMMA - 1.0)
G5 = 2.0/(GAMMA + 1.0)
G6 = (GAMMA - 1.0)/(GAMMA + 1.0)
G7 = (GAMMA - 1.0)/2.0
G8 = GAMMA - 1.0
