import numpy as np
from scipy.sparse import dia_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib.pyplot import spy
from numpy import linalg as la
import matplotlib.pyplot as plt
import pandas as pd
    
class SteadyHeat2Dsparse:
    def __init__(self, Lx, Ly, dimX, dimY):
        self.l = Lx #lunghezza rettangolo
        self.h = Ly
        self.dimX = dimX #quante divisioni
        self.dimY = dimY

        self.dx = Lx/dimX
        self.dy = Ly/dimY

        self.A = None
        self.D_1 = None
        self.R = None

        # self.A = np.identity(self.dimX*self.dimY)
        self.diag = np.zeros([9, self.dimX*self.dimY])
        self.data = np.zeros([9, self.dimX*self.dimY])

        self.set_inner()
        self.b = np.zeros([self.dimX*self.dimY])
        
    
    # build the linear system
    def set_inner(self):
        for i in range(self.dimX+1, (self.dimX*self.dimY)-self.dimX-1, self.dimX): # the start of each row of inner nodes 
            for j in range(self.dimX-2): # loops through all inner nodes in that row 
                k = i+j
                # builds the matrix like in scicomplab, so each row
                # self.A[k][k] = -2 * (1/(self.dx*self.dx) + 1/(self.dy*self.dy)) # central node
                self.diag[4][k] = -2 * (1/(self.dx*self.dx) + 1/(self.dy*self.dy))
                # self.A[k][k-1] = 1/(self.dx*self.dx) # side nodes
                self.diag[3][k] = 1/(self.dx*self.dx)
                # self.A[k][k+1] = 1/(self.dx*self.dx)
                self.diag[5][k] = 1/(self.dx*self.dx)
                # self.A[k][k - self.dimX] = 1/(self.dy*self.dy) # upper lower nodes
                self.diag[1][k] = 1/(self.dy*self.dy)
                # self.A[k][k + self.dimX] = 1/(self.dy*self.dy)
                self.diag[7][k] = 1/(self.dy*self.dy)
                # print(f"Inner node {k}: diag[4][{k}]={self.diag[4][k]}, diag[3][{k}]={self.diag[3][k]}, diag[5][{k}]={self.diag[5][k]}, diag[1][{k}]={self.diag[1][k]}, diag[7][{k}]={self.diag[7][k]}")

    # south
    def set_south(self, bc_type, T_d = 0.0, q = 0.0, alpha = 0.0, T_inf = 0.0):
        if (bc_type=="d"):
            try: 
                self.b[-self.dimX:] = T_d
                for i in range(self.dimX):
                    ii = (self.dimX*self.dimY) - i - 1
                    # self.A[ii][ii] = 1
                    self.diag[4][ii] = 1
            except:
                print("no T_d value for source boundary type")
        elif (bc_type=="n"):
            try:
                for i in range(self.dimX):
                    ii = (self.dimX*self.dimY)-i-1
                    self.b[ii] = q
                    # self.A[ii][ii] = -4/(2*self.dimY)
                    self.diag[4][ii] = -4/(2*self.dimY)
                    # self.A[ii][ii-self.dimX] = 3/(2*self.dimY)
                    self.diag[1][ii] = 3/(2*self.dimY)
                    # self.A[ii][ii-(2*self.dimX)] = 1/(2*self.dimY)
                    self.diag[0][ii] = 1/(2*self.dimY)
            except:
                print("no q value for flux boundary type")
        elif (bc_type=="r"):
            try:
                for i in range(self.dimX):
                    ii = (self.dimX*self.dimY)-i-1
                    self.b[ii] = alpha * T_inf
                    # self.A[ii][ii] = alpha + 3/(2*self.dimY)
                    self.diag[4][ii] = alpha + 3/(2*self.dimX)
                    # self.A[ii][ii-self.dimX] = -4/(2*self.dimY)
                    self.diag[1][ii] = -4/(2*self.dimX)
                    # self.A[ii][ii-(2*self.dimX)] = 1/(2*self.dimY)
                    self.diag[0][ii] = 1/(2*self.dimX)
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
                    ii = i
                    # self.A[ii][ii] = 1
                    self.diag[4][ii] = 1
            except:
                print("no T_d value for source boundary type")
        elif (bc_type=="n"):
            try:
                for i in range(self.dimX):
                    ii = i
                    self.b[ii] = q
                    # self.A[ii][ii] = -4/(2*self.dimY)
                    self.diag[4][ii] = -4/(2*self.dimX)
                    # self.A[ii][ii+self.dimX] = 3/(2*self.dimY)
                    self.diag[7][ii] = -4/(2*self.dimX)
                    # self.A[ii][ii+(2*self.dimX)] = 1/(2*self.dimY)
                    self.diag[8][ii] = -4/(2*self.dimX)
            except:
                print("no q value for flux boundary type")
        elif (bc_type=="r"):
            print("north robin")
            try:
                for i in range(self.dimX):
                    ii = i
                    self.b[ii] = alpha * T_inf
                    # self.A[ii][ii] = alpha + 3/(2*self.dimY)
                    self.diag[4][ii] = alpha + 3/(2*self.dimY)
                    # self.A[ii][ii+self.dimX] = -4/(2*self.dimY)
                    self.diag[7][ii] = -4/(2*self.dimY)
                    # self.A[ii][ii+(2*self.dimX)] = 1/(2*self.dimY)
                    self.diag[8][ii] = 1/(2*self.dimY)
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
                    # self.A[ii][ii] = 1
                    self.diag[4][ii] = 1
            except:
                print("no T_d value for source boundary type")
        elif (bc_type=="n"):
            try:
                for i in range(self.dimY):
                    ii = i * self.dimX
                    self.b[ii] = q
                    # self.A[ii][ii] = -4/(2*self.dimX)
                    self.diag[4][ii] = -4/(2*self.dimX)
                    # self.A[ii][ii+1] = 3/(2*self.dimX)
                    self.diag[5][ii] = 3/(2*self.dimX)
                    # self.A[ii][ii+2] = 1/(2*self.dimX)
                    self.diag[6][ii] = 1/(2*self.dimX)
            except:
                print("no q value for flux boundary type")
        elif (bc_type=="r"):
            try:
                for i in range(self.dimY):
                    ii = i * self.dimX
                    self.b[ii] = alpha * T_inf
                    # self.A[ii][ii] = alpha + 3/(2*self.dimY)
                    self.diag[4][ii] = alpha + 3/(2*self.dimY)
                    # self.A[ii][ii+1] = -4/(2*self.dimY)
                    self.diag[5][ii] = -4/(2*self.dimY)
                    # self.A[ii][ii+2] = 1/(2*self.dimY)
                    self.diag[6][ii] = 1/(2*self.dimY)
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
                    # self.A[ii][ii] = 1
                    self.diag[4][ii] = 1
            except:
                print("no T_d value for source boundary type")
        elif (bc_type=="n"):
            try:
                for i in range(self.dimY):
                    ii = i * self.dimX + self.dimX -1
                    self.b[ii] = q
                    # self.A[ii][ii] = -4/(2*self.dimX)
                    self.diag[4][ii] = -4/(2*self.dimX)
                    # self.A[ii][ii-1] = 3/(2*self.dimX)
                    self.diag[3][ii] = 3/(2*self.dimX)
                    # self.A[ii][ii-2] = 1/(2*self.dimX)
                    self.diag[2][ii] = 1/(2*self.dimX)
            except:
                print("no q value for flux boundary type")
        elif (bc_type=="r"):
            try:
                for i in range(self.dimY):
                    ii = i * self.dimX + self.dimX -1
                    self.b[ii] = alpha * T_inf
                    # self.A[ii][ii] = alpha + 3/(2*self.dimY)
                    self.diag[4][ii] = alpha + 3/(2*self.dimY)
                    # self.A[ii][ii-1] = -4/(2*self.dimY)
                    self.diag[3][ii] = -4/(2*self.dimY)
                    # self.A[ii][ii-2] = 1/(2*self.dimY)
                    self.diag[2][ii] = 1/(2*self.dimY)
            except:
                print("no alpha or T_inf value for conjugate boundary type")
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(bc_type))
       
       

    # solve the linear system
    def solve(self):
        offsets = np.array([-2*self.dimX, -self.dimX, -2, -1, 0, 1, 2, self.dimX, 2*self.dimX])

        for i, offset in enumerate(offsets):
            if offset < 0:
                self.data[i][:offset] = self.diag[i][-offset:]
            elif offset == 0:
                self.data[i] = self.diag[i]
            elif offset > 0:
                self.data[i][offset:] = self.diag[i][:-offset]

        self.A = dia_matrix((self.data, offsets), shape=(self.dimX*self.dimY, self.dimX*self.dimY))
        self.A = csr_matrix(self.A)
        return spsolve(self.A, self.b)
    
    def plot_diag(self):
        plt.figure(figsize=(10, 6))
        plt.imshow(self.diag, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title("Diagonal Array (self.diag)")
        plt.xlabel("Node Index")
        plt.ylabel("Diagonal Index")
        plt.show()
        
    def print_diag(self):
        df_diag = pd.DataFrame(self.diag)
        print(df_diag)

    def print_b(self):
        df_b = pd.DataFrame(self.b, columns=["b"])
        print(df_b)
        
    def print_A(self):
        df_A = pd.DataFrame(self.A)
        print(df_A)

    def solveJacobi(self, threshold=0.001, max_iterations=200):
        residual = 1000000
        iteration = 0

        # Ensure self.diag[4] contains the main diagonal elements of A
        self.D_1 = dia_matrix((1/self.diag[4], 0), shape=(self.dimX*self.dimY, self.dimX*self.dimY))
        self.D_1 = csr_matrix(self.D_1)

        offsets = np.array([-2*self.dimX, -self.dimX, -2, -1, 0, 1, 2, self.dimX, 2*self.dimX])
        self.data = np.zeros((len(offsets), self.dimX * self.dimY))

        for i, offset in enumerate(offsets):
            if offset < 0:
                self.data[i][:offset] = self.diag[i][-offset:]
            elif offset == 0:
                self.data[i] = np.zeros(self.dimX*self.dimY)  # or self.diag[i] if it's supposed to be the main diagonal
            elif offset > 0:
                self.data[i][offset:] = self.diag[i][:-offset]

        self.R = dia_matrix((self.data, offsets), shape=(self.dimX*self.dimY, self.dimX*self.dimY))
        self.R = csr_matrix(self.R)

        x = np.zeros(self.dimX*self.dimY)
        x_new = np.zeros(self.dimX*self.dimY)

        while (iteration < max_iterations):
            T = self.D_1.dot(self.R)
            x_new = self.D_1.dot(self.b) - T.dot(x)
            residual = np.linalg.norm(x_new - x)  # Ensure correct residual calculation
            iteration += 1
            x = x_new

        return x

        

