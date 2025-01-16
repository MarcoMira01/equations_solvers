# Implementation of Steepest Descent algorithm for linear systems
# in the form of Ax = b, with
# - A a positive defined symmetric (square) matrix of dimension n*n
# - b a vector of dimension n*1

# import sys
# sys.path.append('../core/')

from my_solvers.core.Solver import Solver
from typing                 import Callable
from numpy.linalg           import norm, eigvals
from numpy                  import isscalar
from numpy                  import allclose
from numpy                  import all
from math                   import log

# from multipledispatch       import dispatch
# from typing                 import overload
# from multimethods import multimethod

class Solver_SD_LS( Solver ):

    # =================================== #
    # Constructor
    # =================================== #
    def __init__( self , 
                  max_iter: int   = 2e2   , 
                  tol:      float = 1e-5  ,
                  info:     bool  = False ,
                  step:     float = 1e-3  ,
                  res_upd:  int   = 50  ) -> None:
        
        # Parent constructor
        super().__init__( max_iter = max_iter , tol = tol , 
                          info = info , step = step )
        
        # Additional settings
        self.__residual_update = res_upd   

    # =========================== #
    # Set methods
    # =========================== #
    # Override set step adaptability
    def set_step_adaptability( self , step_adpt: bool ) -> None:
        self.__step_adpt = step_adpt

    # =========================== #
    # Get methods
    # =========================== #
    # Override get step adaptability
    def get_step_adaptability( self , step_adpt: bool ) -> None:
        return self.__step_adpt
    
    # =========================== #
    # Solver implementation
    # =========================== #
    def solve( self , A: float , b: float , X0: float ):
        # Check input consistency for a correct calculation of steepest descent
        self.__matrix_consistency( A , b , X0 )

        # Steepest descent implementation
        i      = 0                               # index initialization
        res    = b - A.dot(X0)                   # initial residual
        num    = (res.T).dot(res)                # the num of steplength update is equal to the norm-2 of residual
        den    = ((res.T).dot(A)).dot(res)
        delta0 = num                            
        X      = X0

        # Main iteration: 
        # stop when ||res||^2 < tol*||res_0||^2 -> norm2 of residual < fraction of initial residual
        # OR 
        # max iter number reached
        while (num > self._tol*delta0) and (i < self._max_iter):
            # Update the step length alpha
            alpha = num/den

            # Update solution
            X = X + alpha*res

            # Update residual
            if (i % self.__residual_update == 0):  # use expression to reduce numerical errors accumulation
                res = b - A.dot(X)
            else:
                res = res - alpha*(A.dot(res))
            
            # Update the rest
            num = (res.T).dot(res)
            den = ((res.T).dot(A)).dot(res)

            i = i+1

        return X
    
    # =========================== #
    # Line search step-size update
    # =========================== #
    

    # =========================== #
    # Matrix consistency
    # =========================== #
    def __matrix_consistency( self , A: float , b: float , X0: float ):
        # Check the correct dimensions of A,B, and X0
        nrA = A.shape[0]
        ncA = A.shape[1]
        nb  = b.shape[0]
        nX  = X0.shape[0]

        # Check if A is square
        if nrA != ncA:
            raise SystemExit('Error: A is not square')
        
        # Check if A is symmetric
        if not allclose( A , A.T , rtol=1e-05 , atol=1e-08 ):
            raise SystemExit('Error: A is not symmetric')
        
        # Check if A is positive definite
        if not all(eigvals(A) > 0):
            raise SystemExit('Error: A is not positive definite')
        
        # Check dimension mismatch between matrix and vectors
        if (nrA != nb) or (nrA != nX) or (nb != nX):
            raise SystemExit('Error: dimension mismatch between A,b, and x')
        
