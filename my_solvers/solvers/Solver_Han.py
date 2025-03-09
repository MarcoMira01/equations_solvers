# import sys
# sys.path.append('../core/')

from my_solvers.core.Solver import Solver
from typing                 import Callable
from numpy.linalg           import norm
from numpy                  import isscalar
from math                   import log

# from multipledispatch       import dispatch
# from typing                 import overload
# from multimethods import multimethod

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
                          info = info )

        # Additional settings
        self.__step      = step
        self.__step_adpt = step_adpt
        self.__eps       = e

    # ====================== #
    # Set methods
    # ====================== #
    def set_solver_step( self , step: float ) -> None:
        self.__step = step

    # Override set step adaptability
    def set_step_adaptability( self , step_adpt: bool ) -> None:
        self.__step_adpt = step_adpt

    # ====================== #
    # Get methods
    # ====================== #
    def get_solver_step( self ) -> float:
        return self.__step

    # Override get step adaptability
    def get_step_adaptability( self ) -> bool:
        return self.__step_adpt
    
    # ====================== #
    # Solver implementation
    # ====================== #
    def solve( self , fcn_hndl: Callable , X0: float , step_max: float = 1e3 ):
        if not self.__step_adpt:
            X = self.__solver_fixed_step( fcn_hndl , X0 )
        else:
            X = self.__solver_var_step( fcn_hndl , X0 , step_max )
        return X
    
    # Implementation of Han solver in the case of fixed step
    # - If the step is a scalar, then we follow the base approach of
    #   iterating three times the process by doubling each time the
    #   step value
    # - Otherwise, if it is a vector, we iterate over it
    def __solver_fixed_step( self , fcn_hndl: Callable , X0: float ):

        # 0) Define local parameters for easy of reading
        # h = self.__step
        if isscalar(self.__step):
            h = [ self.__step , 2*self.__step , 4*self.__step ]
        else:
            h = self.__step
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
    
    # Implementation of Han solver with automatic step size control
    # - The step set at the beginning is used as starting value
    # - If the step was set as a vector, then only the first value is considered
    def __solver_var_step( self , fcn_hndl: Callable , X0: float , step_max: float ):
        
        # 0) Define local parameters for easy of reading
        if not isscalar(self.__step):
            h = self.__step[0]
        else:
            h = self.__step
        h_max = step_max
        e = self.__eps

        # 1) Compute parameter omega
        w = h/( h + e )

        # 2) Evaluate function at starting point X0
        #    and compute ancillary variable z0
        Z0  = h*fcn_hndl( X0 )

        # 3) Compute new point x amd evaluate function
        # NOTE: Xn1 = X_(n+1)
        Xn1_0  = X0 - Z0
        rho = norm( fcn_hndl( Xn1_0 ) )

        # 4) Start the iterative computation
        Zn    = Z0
        Xn    = X0
        rho_n = rho
        h_n   = h
        n     = 0

        always_en = False
        while ( n < self._max_iter ) and ( rho_n > self._tol ):
            
            Zn1 = w*( e*fcn_hndl( Xn1_0 ) + Zn )
            Xn1 = Xn - Zn1

            n     = n+1
            Xn    = Xn1
            Zn    = Zn1
            Xn1_0 = Xn - Zn
            rho_n1 = norm( fcn_hndl(Xn1_0) )

            # Only for the implicit case?
            # Limit the change up to 50%
            sigma_n = log(rho_n1)-log(rho_n)
            h_n1    = h_n
            if sigma_n > -0.5 or always_en:
                h_n1 = round( min( h_max , h_n*rho_n/rho_n1 , 1.5*h_n ) , 4 )
            
            h_n   = h_n1
            rho_n = rho_n1

            w = h_n/( h_n + e )

            if ( n >= self._max_iter ):
                print("Solver stopped since max number of iterations reached")

        return Xn1_0