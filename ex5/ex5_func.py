import numpy as np    
    
class SteadyHeat2D:
    def __init__(self, Lx, Ly, dimX, dimY):
        self.l = Lx #lunghezza rettangolo
        self.h = Ly
        self.dimX = dimX #quante divisioni
        self.dimY = dimY

        self.dx = Lx/dimX
        self.dy = Ly/dimY

        self.A = np.identity(self.dimX*self.dimY)
        self.set_inner()
        self.b = np.zeros([self.dimX*self.dimY])
        
    
    # build the linear system
    def set_inner(self):
        for i in range(self.dimX+1, (self.dimX*self.dimY)-self.dimX-1, self.dimX): # the start of each row of inner nodes 
            for j in range(self.dimX-2): # loops through all inner nodes in that row 
                k = i+j
                # builds the matrix like in scicomplab, so each row
                self.A[k][k] = -2 * (1/(self.dx*self.dx) + 1/(self.dy*self.dy)) # central node
                self.A[k][k-1] = 1/(self.dx*self.dx) # side nodes
                self.A[k][k+1] = 1/(self.dx*self.dx)
                self.A[k][k - self.dimX] = 1/(self.dy*self.dy) # upper lower nodes
                self.A[k][k + self.dimX] = 1/(self.dy*self.dy)

    # south
    def set_south(self, bc_type, T_d=0.0, q=0.0, alpha = 0.0, T_inf=0.0):
        if (bc_type=="d"):
            try: 
                self.b[-self.dimX:] = T_d
                for i in range(self.dimX):
                    ii = (self.dimX*self.dimY) - i - 1
                    self.A[ii][ii] = 1
            except:
                print("no T_d value for source boundary type")
        elif (bc_type=="n"):
            try:
                for i in range(self.dimX):
                    ii = (self.dimX*self.dimY)-i-1
                    self.b[ii] = q
                    self.A[ii][ii] = -4/(2*self.dimY)
                    self.A[ii][ii-self.dimX] = 3/(2*self.dimY)
                    self.A[ii][ii-(2*self.dimX)] = 1/(2*self.dimY)
            except:
                print("no q value for flux boundary type")
        elif (bc_type=="r"):
            try:
                for i in range(self.dimX):
                    ii = (self.dimX*self.dimY)-i-1
                    self.b[ii] = alpha * T_inf
                    self.A[ii][ii] = alpha + 3/(2*self.dimY)
                    self.A[ii][ii-self.dimX] = -4/(2*self.dimY)
                    self.A[ii][ii-(2*self.dimX)] = 1/(2*self.dimY)
            except:
                print("no alpha or T_inf value for conjugate boundary type")
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(bc_type))


    # north
    def set_north(self, bc_type, T_d=0.0, q=0.0, alpha = 0.0, T_inf=0.0):
        if (bc_type=="d"):
            try: 
                self.b[:self.dimX] = T_d
                for i in range(self.dimX):
                    ii = (self.dimX*self.dimY) - i - 1
                    self.A[ii][ii] = 1
            except:
                print("no T_d value for source boundary type")
        elif (bc_type=="n"):
            try:
                for i in range(self.dimX):
                    ii = i
                    self.b[ii] = q
                    self.A[ii][ii] = -4/(2*self.dimY)
                    self.A[ii][ii+self.dimX] = 3/(2*self.dimY)
                    self.A[ii][ii+(2*self.dimX)] = 1/(2*self.dimY)
            except:
                print("no q value for flux boundary type")
        elif (bc_type=="r"):
            try:
                for i in range(self.dimX):
                    ii = i
                    self.b[ii] = alpha * T_inf
                    self.A[ii][ii] = alpha + 3/(2*self.dimY)
                    self.A[ii][ii+self.dimX] = -4/(2*self.dimY)
                    self.A[ii][ii+(2*self.dimX)] = 1/(2*self.dimY)
            except:
                print("no alpha or T_inf value for conjugate boundary type")
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(bc_type))


    # west
    def set_west(self, bc_type, T_d=0.0, q=0.0, alpha = 0.0, T_inf=0.0):
        if (bc_type=="d"):
            try: 
                for i in range(self.dimY):
                    ii = i * self.dimX
                    self.b[ii] = T_d
                    self.A[ii][ii] = 1
            except:
                print("no T_d value for source boundary type")
        elif (bc_type=="n"):
            try:
                for i in range(self.dimY):
                    ii = i * self.dimX
                    self.b[ii] = q
                    self.A[ii][ii] = -4/(2*self.dimX)
                    self.A[ii][ii+1] = 3/(2*self.dimX)
                    self.A[ii][ii+2] = 1/(2*self.dimX)
            except:
                print("no q value for flux boundary type")
        elif (bc_type=="r"):
            try:
                for i in range(self.dimY):
                    ii = i * self.dimX
                    self.b[ii] = alpha * T_inf
                    self.A[ii][ii] = alpha + 3/(2*self.dimY)
                    self.A[ii][ii+1] = -4/(2*self.dimY)
                    self.A[ii][ii+2] = 1/(2*self.dimY)
            except:
                print("no alpha or T_inf value for conjugate boundary type")
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(bc_type))


# east
    def set_east(self, bc_type, T_d=0.0, q=0.0, alpha = 0.0, T_inf=0.0):
        if (bc_type=="d"):
            try: 
                for i in range(self.dimY):
                    ii = i * self.dimX + self.dimX -1
                    self.b[ii] = T_d
                    self.A[ii][ii] = 1
            except:
                print("no T_d value for source boundary type")
        elif (bc_type=="n"):
            try:
                for i in range(self.dimY):
                    ii = i * self.dimX + self.dimX -1
                    self.b[ii] = q
                    self.A[ii][ii] = -4/(2*self.dimX)
                    self.A[ii][ii-1] = 3/(2*self.dimX)
                    self.A[ii][ii-2] = 1/(2*self.dimX)
            except:
                print("no q value for flux boundary type")
        elif (bc_type=="r"):
            try:
                for i in range(self.dimY):
                    ii = i * self.dimX + self.dimX -1
                    self.b[ii] = alpha * T_inf
                    self.A[ii][ii] = alpha + 3/(2*self.dimY)
                    self.A[ii][ii-1] = -4/(2*self.dimY)
                    self.A[ii][ii-2] = 1/(2*self.dimY)
            except:
                print("no alpha or T_inf value for conjugate boundary type")
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(bc_type))
       
       

    # solve the linear system
    def solve(self):
        return np.linalg.solve(self.A, self.b)
