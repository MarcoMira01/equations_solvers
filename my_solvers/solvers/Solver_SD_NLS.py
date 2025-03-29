# Implementation of Steepest Descent algorithm for non linear systems
# in the form of f(x) = 0

# import sys
# sys.path.append('../core/')

from my_solvers.core.Solver import Solver
from numpy                  import random
from numpy                  import power
from numdifftools           import Gradient as ndGradient

# from multipledispatch       import dispatch
# from typing                 import overload
# from multimethods import multimethod

class Solver_SD_NLS( Solver ):

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
        self.__n_iter = 0         # number of iterations to converge to the solution

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
    def get_step_adaptability(self) -> bool:
        return True
    
    def get_n_iter(self) -> int:
        return self.__n_iter
    
    # =========================== #
    # Solver implementation
    # =========================== #
    def solve( self , hfun , X0: float , method = 'back_track_iter' , BT_factor:float = 5e-2 , beta_mth = 'PR' ):

        # Initialize random generator if random step size is selected
        rng = random.default_rng() 

        # Initialization at step 0
        F_X0  = hfun(X0)
        X_im1 = X0
        J_X0  = 0.5*(F_X0.T).dot(F_X0)   # obj_fun = 0.5||F(x)||^2
        J_Xi  = J_X0
        p_im1 = -(ndGradient(hfun)(X0).T).dot(hfun(X0))

        i = 1        

        # Main iteration: 
        # stop when 0.5||F(x)||^2 < tol*0.5||F(x0)||^2
        # OR 
        # max iter number reached
        while (J_Xi > self._tol*J_X0) and (i < self._max_iter):
            
            match method:
                case 'back_track_iter':
                    alpha_i , descent_i = self.__solve_BT( hfun , X_im1 , BT_factor )

                # NOT WORKING
                # case 'conj_grad':
                #     if (i > 1):
                #         descent_i = self.__solve_CG( hfun , X_im1 , X_im2 , p_im1 , beta_mth )
                #         alpha_i   = self.__solve_BT_p( hfun , X_im1 , BT_factor , descent_i )
                #         X_im2 = X_im1
                #     else:
                #         # First iteration
                #         alpha_i , descent_i = self.__solve_BT( hfun , X_im1 , BT_factor )
                #         X_im2 = X_im1
                
                case "barzilai_borwein_short":
                    if (i > 1):
                        alpha_i , descent_i = self.__solve_BB_short( hfun , X_im1 , X_im2 )
                        X_im2 = X_im1
                    else:
                        # First iteration
                        alpha_i , descent_i = self.__solve_BT( hfun , X_im1 , BT_factor )
                        X_im2 = X_im1

                case "barzilai_borwein_long":
                    if (i > 1):
                        alpha_i , descent_i = self.__solve_BB_long( hfun , X_im1 , X_im2 )
                        X_im2 = X_im1
                    else:
                        # First iteration
                        alpha_i , descent_i = self.__solve_BT( hfun , X_im1 , BT_factor )
                        X_im2 = X_im1
                    
                case "barzilai_borwein_alt":
                    if (i > 1):
                        # Alternate between short and long BB steps
                        if (i % 2==0):
                            alpha_i , descent_i = self.__solve_BB_short( hfun , X_im1 , X_im2 )                         
                        else:
                            alpha_i , descent_i = self.__solve_BB_long( hfun , X_im1 , X_im2 )
                        X_im2 = X_im1
                    else:
                        # First iteration
                        alpha_i , descent_i = self.__solve_BT( hfun , X_im1 , BT_factor )
                        X_im2 = X_im1

                case "random_step":  
                    alpha_i , descent_i = self.__solve_BT( hfun , X_im1 , BT_factor )
                    alpha_i = rng.uniform( low = 0 , high = 2 )*alpha_i   # Randomize the step

                # If an exact match is not confirmed, this last case will be used if provided
                case _:
                    raise SystemExit('Error: not expected solver method')

            X_i     = X_im1 + alpha_i*descent_i
            X_im1   = X_i
            p_im1   = descent_i

            F_Xi = hfun(X_i)
            J_Xi = 0.5*(F_Xi.T).dot(F_Xi)

            i = i+1  

            self.__step = alpha_i     

        self.__n_iter = i   

        return X_i
    
    # =========================== #
    # Backtracking method
    # =========================== #
    def __solve_BT( self , hfun , X: float , scaling: float ):
            
        alpha = 1
        entry = 1
        k     = 1

        J_X     = 0.5*(hfun(X).T).dot(hfun(X))
        grad_JX = (ndGradient(hfun)(X).T).dot(hfun(X))

        # Iterative backtracking
        while (entry == 1):

            # Computation of right side term for while entry condition
            right_term = J_X - (alpha/2)*((grad_JX).T).dot(grad_JX)

            # Computation of left side term for while entry condition     
            Xtmp = X - alpha*grad_JX
            left_term = 0.5*(hfun(Xtmp).T).dot(hfun(Xtmp))

            if (left_term > right_term):
                # Update alpha
                alpha = scaling*alpha
            else:
                entry = 0

            k = k+1
        
        descent = -grad_JX

        return alpha , descent
    
    # =========================== #
    # Barzilai-Borwein method
    # =========================== #
    # Short step
    def __solve_BB_short( self , hfun , X_im1: float , X_im2: float ):
            
        delta_X = X_im1 - X_im2

        grad_JXm2 = (ndGradient(hfun)(X_im2).T).dot(hfun(X_im2))
        grad_JXm1 = (ndGradient(hfun)(X_im1).T).dot(hfun(X_im1))
        delta_g = grad_JXm1 - grad_JXm2

        alpha = ((delta_X.T).dot(delta_g))/((delta_g.T).dot(delta_g))
        
        descent = -grad_JXm1

        return alpha , descent
    
    # Long step
    def __solve_BB_long( self , hfun , X_im1: float , X_im2: float ):
            
        delta_X = X_im1 - X_im2

        grad_JXm2 = (ndGradient(hfun)(X_im2).T).dot(hfun(X_im2))
        grad_JXm1 = (ndGradient(hfun)(X_im1).T).dot(hfun(X_im1))
        delta_g = grad_JXm1 - grad_JXm2

        alpha = ((delta_X.T).dot(delta_X))/((delta_X.T).dot(delta_g))

        descent = -grad_JXm1

        return alpha , descent