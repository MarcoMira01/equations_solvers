# import sys
# sys.path.append('../core/')

from my_solvers.core.Solver import Solver
from typing       import Callable
from numpy        import array
from numpy.linalg import norm

class Solver_Han( Solver ):

    # =================================== #
    # Constructor
    # =================================== #
    def __init__( self , 
                step_adpt: bool = False , 
                e: float = 2e-2 ) -> None:
        
        # Parent constructor
        super().__init__()

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

        # 1) Compute parameter omega
        w = self._step/( self._step + self.__eps )

        # 2) Evaluate function at starting point X0
        #    and compute ancillary variable z0
        fX = fcn_hndl( X0 )
        Z  = w*fX

        # 3) Compute new point x amd evaluate function
        X   = X0 - Z
        fX  = fcn_hndl( X )
        rho = norm( fX )

        # 4) Start the iterative computation
        i = 1
        while ( i <= self._max_iter ) and ( rho > self._tol ):
            
            Znew = w*( self.__eps*fcn_hndl( X ) + Z )
            Xnew = X-Znew

            i   = i+1
            X   = Xnew - Znew
            fX  = fcn_hndl( X )
            rho = norm( fX )

        if ( i >= self._max_iter ):
            print("Solver stopped since max number of iterations reached")

        return X