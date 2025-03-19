# Implementation of Steepest Descent algorithm for linear systems
# in the form of Ax = b, with
# - A a positive defined symmetric (square) matrix of dimension n*n
# - b a vector of dimension n*1

# import sys
# sys.path.append('../core/')

from my_solvers.core.Solver import Solver
from numpy.linalg           import eigvals
from numpy                  import allclose
from numpy                  import all
from numpy                  import random

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
                  info:     bool  = False  ) -> None:
        
        # Parent constructor
        super().__init__( max_iter = max_iter , tol = tol , 
                          info = info )
        
        # Additional definitions
        self.__step = 0           # here step is an internal variable automatically managed by the routine!

    # =========================== #
    # Set methods
    # =========================== #
    # No additional (abstract) set methods to be implemented 

    # =========================== #
    # Get methods
    # =========================== #
    def get_solver_step(self) -> float:
        return self.__step

    # Override get step adaptability
    def get_step_adaptability( self , step_adpt: bool ) -> bool:
        return True
    
    # =========================== #
    # Solver implementation
    # =========================== #
    def solve( self , A: float , b: float , X0: float , method: str = "lin_search" , BT_factor:float = 5e-2 , res_upd: int = 50 ):
        # Check input consistency for a correct calculation of steepest descent
        self.__matrix_consistency( A , b , X0 )

        # Initialize random generator if random step size is selected
        rng = random.default_rng() 

        # Initialization at step 0
        X_im1     = X0
        res_im1   = b - A.dot(X_im1)
        res_sqr_0 = (res_im1.T).dot(res_im1)
        res_sqr   = res_sqr_0
        p_im1     = res_im1

        i = 1        

        # Main iteration: 
        # stop when ||res||^2 < tol*||res_0||^2 -> norm2 of residual < fraction of initial residual
        # OR 
        # max iter number reached
        while (res_sqr > self._tol*res_sqr_0) and (i < self._max_iter):
        
            match method:
                case "lin_search":
                    if (i % res_upd == 0):  # use expression to reduce numerical errors accumulation
                        res_im1 = b - A.dot(X_im1)
                    alpha_i , res_i = self.__solve_ELS( A , res_im1 )
                    descent = res_im1

                case "conj_grad":                    
                    alpha_i , res_i , p_i = self.__solve_CG( A , res_im1  , p_im1 )
                    descent = p_im1
                    p_im1   = p_i

                case "back_track_iter":
                    alpha_i , res_i = self.__solve_BT( A , b , X_im1 , BT_factor )
                    descent = res_im1

                case "barzilai_borwein_short":
                    if (i > 1):
                        alpha_i , res_i = self.__solve_BB_short( A , b , X_im1 , X_im2 )
                        X_im2 = X_im1
                    else:
                        # First iteration
                        alpha_i , res_i = self.__solve_ELS( A , res_im1 )
                        X_im2 = X_im1                        
                    descent = res_i

                case "barzilai_borwein_long":
                    if (i > 1):
                        alpha_i , res_i = self.__solve_BB_long( A , b , X_im1 , X_im2 )
                        X_im2 = X_im1
                    else:
                        # First iteration
                        alpha_i , res_i = self.__solve_ELS( A , res_im1 )
                        X_im2 = X_im1                        
                    descent = res_i

                case "barzilai_borwein_alt":
                    if (i > 1):
                        # Alternate between short and long BB steps
                        if (i % 2==0):
                            alpha_i , res_i = self.__solve_BB_short( A , b , X_im1 , X_im2 )                            
                        else:
                            alpha_i , res_i = self.__solve_BB_long( A , b , X_im1 , X_im2 )
                        X_im2 = X_im1
                    else:
                        # First iteration
                        alpha_i , res_i = self.__solve_ELS( A , res_im1 )
                        X_im2 = X_im1                        
                    descent = res_i

                case "random_step":  
                    res_im1 = b - A.dot(X_im1)                            # Always overwrite the residual with exact calculation
                    alpha_i , res_i = self.__solve_ELS( A , res_im1 )
                    alpha_i = rng.uniform( low = 0 , high = 2 )*alpha_i   # Randomize the step
                    res_i   = res_im1                                     # Overwrite the residual with the previous one for the exit condition
                    descent = res_im1

                # If an exact match is not confirmed, this last case will be used if provided
                case _:
                    raise SystemExit('Error: not expected solver method')

            res_sqr = (res_i.T).dot(res_i)

            X_i     = X_im1 + alpha_i*descent
            X_im1   = X_i
            res_im1 = res_i

            i = i+1  

            self.__step = alpha_i        

        return X_i

    # =========================== #
    # Exact linear seach
    # =========================== #
    def __solve_ELS( self , A: float , res: float ):

        # Compute step alpha
        num   = (res.T).dot(res)
        den   = (res.T).dot(A.dot(res))
        alpha = num/den

        # Update residual
        res_new = res - alpha*(A.dot(res))

        return alpha, res_new
    
    # =========================== #
    # Conjugate gradient method
    # =========================== #
    def __solve_CG( self , A: float , res: float , p: float ):

        # Compute step alpha
        num   = (res.T).dot(res)
        den   = (p.T).dot(A.dot(p))
        alpha = num/den

        # Update residual
        res_new = res - alpha*A.dot(p)

        # Update p
        num  = (res_new.T).dot(res_new)
        den  = (res.T).dot(res)
        beta = num/den 
        p_new = res_new + beta*p
        
        return alpha , res_new , p_new    
    
    # =========================== #
    # Backtracking method
    # =========================== #
    def __solve_BT( self , A: float , b: float , X: float , scaling: float ):
            
        alpha = 1
        entry = 1
        k     = 1

        # Iterative backtracking
        while (entry == 1):

            f_grad = A.dot(X) - b
            X_temp = X - alpha*f_grad

            # Computation of right side term for while entry condition                
            f_grad_sqr = (f_grad.T).dot(f_grad)
            f_X = 0.5*((X.T).dot(A)).dot(X) - (b.T).dot(X)
            right_term = f_X - (alpha/2)*f_grad_sqr

            # Computation of left side term for while entry condition                
            f_X_temp = 0.5*((X_temp.T).dot(A)).dot(X_temp) - (b.T).dot(X_temp)
            left_term = f_X_temp

            if (left_term > right_term):
                # Update alpha
                alpha = scaling*alpha
            else:
                entry = 0

            k = k+1

        res = b - A.dot(X)

        return alpha , res
    
    # =========================== #
    # Barzilai-Borwein method
    # =========================== #
    # Short step
    def __solve_BB_short( self , A: float , b: float , X_im1: float , X_im2: float ):
            
        delta_X = X_im1 - X_im2
        delta_g = A.dot( delta_X )

        alpha = ((delta_X.T).dot(delta_g))/((delta_g.T).dot(delta_g))
        
        res = b - A.dot(X_im1)
        
        return alpha , res
    
    # Long step
    def __solve_BB_long( self , A: float , b: float , X_im1: float , X_im2: float ):
            
        delta_X = X_im1 - X_im2
        delta_g = A.dot( delta_X )

        alpha = ((delta_X.T).dot(delta_X))/((delta_X.T).dot(delta_g))

        res = b - A.dot(X_im1)
        
        return alpha , res

    # =========================== #
    # Random step
    # =========================== #
    # def __solve_RandS( self , A: float , res: float ):
        
    #     rng = random.default_rng()
    #     theta = rng.uniform( low = 0 , high = 2 )


    #     # Compute step alpha
    #     num   = (res.T).dot(res)
    #     den   = (res.T).dot(A.dot(res))
    #     alpha = theta*(num/den)

    #     # Update residual
    #     res_new = res - alpha*(A.dot(res))

    #     return alpha, res_new

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