import math
import numpy as np

def example_1( X: float ):
    fX = np.zeros(2)
    fX[0] = X[0]**2 - X[1] + 1
    fX[1] = X[0] - math.cos(math.pi*X[1]/2)

    return fX

def example_2( X: float ):
    fX = np.zeros(3)
    fX[0] = X[0]**2 + X[1]**2 + X[2]**2 -1
    fX[1] = X[0] + X[1] + X[2]
    fX[2] = X[0] - X[1]**2

    return fX

def example_3( X: float ):
    fX = np.zeros(3)
    A = np.array([[1, 3, -2], [1, 2, 5], [-4, 3, 1]])
    b = np.array([[8],[10],[5]])
    tmp = A.dot(X)-b
    fX[0] = tmp[0][0]
    fX[1] = tmp[1][0]
    fX[2] = tmp[2][0]

    return fX