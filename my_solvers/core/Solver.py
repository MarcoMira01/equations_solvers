from abc import ABC, abstractmethod
from typing import Callable 

class Solver(ABC):
    
    # =================================== #
    # Protected attributes
    # =================================== #
    # _max_iter              # max number of iterations
    # _tol                   # tolerance value for the solution
    # _info                  # boolean: switch on/off additional info in output
    # _step                  # step used by the solver for computing the solution
    # _step_adpt             # boolean: switch on/off automatic step variation

    # =================================== #
    # Constructor
    # =================================== #
    def __init__( self , 
                  max_iter: int = 2e2     , 
                  tol: float      = 1e-5  ,
                  info: bool      = False  ) -> None:
        
        self._max_iter  = max_iter
        self._tol       = tol
        self._info      = info        

    # =================================== #
    # Public methods
    # =================================== #
    # ====================== #
    # Set methods
    # ====================== #
    def set_max_iter( self , max_iter: int ) ->None:
        """ Set the maximum number of iterations (or function evaluations)
            that the solver can perform to find a solution """
        self._max_iterations = max_iter
    
    def set_tolerance( self , tol: float ) ->None:
        """ Set the tolerance value for considering a value as zero 
            of the system """
        self._tol = tol

    def set_out_info( self , info: bool ) -> None:
        """ Enable/disable the output of additional information at the end
            of the procedure """
        self._info = info
    
    # ====================== #
    # Get methods
    # ====================== #
    def get_max_iter( self ) ->None:
        """ Get the maximum number of iterations (or function evaluations)
            that the solver can perform to find a solution """
        return self._max_iter
    
    def get_tolerance( self ) ->None:
        """ Get the tolerance value for considering a value as zero 
            of the system """
        return self._tol

    def get_out_info( self ) -> None:
        """ Get if the output of additional information at the end
            of the procedure is switched on/off """
        return self._info

    @abstractmethod
    def get_solver_step( self ) -> None:
        """ Get the step that the solver uses to calculate the roots """
        return NotImplementedError

    @abstractmethod
    def get_step_adaptability( self ) -> None:
        """ Get if auto-update of solver step is switched on or off """
        raise NotImplementedError
    
    # ====================== #
    # Solver implementation
    # ====================== #
    @abstractmethod
    def solve( self , fcn_hndl: Callable , X0: float ):
        """ Compute the root of the function passed in input starting from initial point X0 """
        raise NotImplementedError