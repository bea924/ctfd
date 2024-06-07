import numpy as np


def index(i, j, n):
    # Return the index in the computational vector based on the physical indices 'i' and 'j'
    return j+i*(n)

class Coordinate2D():
    def __init__(self, x, y):
        self.x = x
        self.y = y

def calculate_area(ur: Coordinate2D, br: Coordinate2D, bl: Coordinate2D, ul: Coordinate2D):
    # calculate the area of the cell
    # ul (upper left), bl (bottom left), br (bottom right), ur (upper right) are the coordinates of the four vertices of the cell
    # apply Gaussian trapezoidal formula to calculate the areas
    return 0.5 * abs( (ur.x*br.y - br.x*ur.y) + (br.x*bl.y - bl.x*br.y)
                     + (bl.x*ul.y - ul.x*bl.y) + (ul.x*ur.y - ur.x*ul.y) )

def dy(a: Coordinate2D, b: Coordinate2D):
    # Calculate distance between 'a' and 'b' along the y axis
    return b.y - a.y

def dx(a: Coordinate2D, b: Coordinate2D):
    # Calculate distance between 'a' and 'b' along the x axis
    return b.x - a.x

def dist(a: Coordinate2D, b: Coordinate2D):
    # Calculate the euclidean distance between 'a' and 'b'
    return ( (a.x - b.x)**2 + (a.y - b.y)**2 )**0.5

#############################################################################
def formfunction(x, shape: str):
    h1 = 4
    hm = 4
    h2 = 2

    if shape == 'linear':
        return (1-x)*h1/2 + x*h2/2
    
    elif shape == 'quadratic':
        c1 = h2+2*h1/2-2*hm
        c2 = 2*hm - 3*h1/2 - h2/2
        c3 = h1/2
        return c1*x**2 +c2*x + c3
    
    elif shape == 'crazy':
        d1 = 3
        d2 = 4
        return (1-x)*h1/2 + x*h2/2+ np.dot((np.sin(2*np.pi*d1*x)),(1-(1-1/d2)*x))
    
    else:
        raise ValueError('Unknown shape: %s' % shape)

def setUpMesh(n, shape: str, formfunction = formfunction):
    x = np.linspace(0, 10, n)
    y_elevation = np.linspace(10, 3, n)
    Y = np.linspace(0, y_elevation, n)
    X = np.tile(x, (n, 1))
    return X, Y