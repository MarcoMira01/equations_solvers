# import sys
# sys.path.append('../core/')

from my_solvers.core.Solver import Solver
from typing                 import Callable
from numpy.linalg           import norm
from numpy                  import isscalar
from multipledispatch       import dispatch
from typing                 import overload
from multimethods import multimethod

class Solver_Han( Solver ):

    # =================================== #
    # Constructor
    # =================================== #
    def __init__( self , 
                  max_iter: int = 2e2     , 
                  tol: float      = 1e-5  ,
                  info: bool      = False ,
                  step: float     = 1e-3  ,
                  step_adpt: bool = False ,
                  e: float = 0.5 ) -> None:
        
        # Parent constructor
        super().__init__( max_iter = max_iter , tol = tol , 
                          info = info , step = step )

        # Additional settings
        self.__step_adpt = step_adpt
        self.__eps       = e

    # ====================== #
    # Set methods
    # ====================== #
    # Override set step adaptability
    def set_step_adaptability( self , step_adpt: bool ) -> None:
        self.__step_adpt = step_adpt

    # ====================== #
    # Get methods
    # ====================== #
    # Override get step adaptability
    def get_step_adaptability( self , step_adpt: bool ) -> None:
        return self.__step_adpt
    
    # ====================== #
    # Solver implementation
    # ====================== #
    def solve( self , fcn_hndl: Callable , X0: float ):
        if not self.__step_adpt:
            X = self.__solver_fixed_step( fcn_hndl , X0 )
            return X
        
    def __solver_fixed_step( self , fcn_hndl: Callable , X0: float ):

        # 0) Define local parameters for easy of reading
        # h = self._step
        if isscalar(self._step):
            h = [ self._step , 2*self._step , 4*self._step ]
        else:
            h = self._step
        e = self.__eps

        # 1) Compute parameter omega
        w = h[0]/( h[0] + e )

        # 2) Evaluate function at starting point X0
        #    and compute ancillary variable z0
        Z0  = h[0]*fcn_hndl( X0 )

        # 3) Compute new point x amd evaluate function
        # NOTE: Xn1 = X_(n+1)
        Xn1_0  = X0 - Z0
        rho = norm( fcn_hndl( Xn1_0 ) )

        # 4) Start the iterative computation
        Zn = Z0
        Xn = X0
        for i in range(3):
            n = 0
            while ( n < self._max_iter ) and ( rho > self._tol ):
                
                Zn1 = w*( e*fcn_hndl( Xn1_0 ) + Zn )
                Xn1 = Xn - Zn1

                n     = n+1
                Xn    = Xn1
                Zn    = Zn1
                Xn1_0 = Xn - Zn
                rho = norm( fcn_hndl(Xn1_0) )

            h[i] = h[i]*2
            w = h[i]/( h[i] + e )

            if ( n >= self._max_iter ):
                print("Solver stopped since max number of iterations reached")

        return Xn1_0