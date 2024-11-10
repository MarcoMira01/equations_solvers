import sys
sys.path.append('../core/')

from equations_solvers.my_solvers.core.Solver import Solver
from typing import Callable

class Solver_Han( Solver ):

    # =================================== #
    # Constructor
    # =================================== #
    def __init__( self , 
                step_adpt: bool = False) -> None:
        
        # Parent constructor
        super().__init__()

        # Additional settings
        self.__step_adpt = step_adpt

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
        pass