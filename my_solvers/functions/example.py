import math
import numpy as np

def example_1( X ):
    fX = np.array( [ 0 , 0 ] )
    fX[0] = X[0]**2 - X[1] + 1
    fX[1] = X[0] - math.cos(math.pi*X[1]/2)

    return fX