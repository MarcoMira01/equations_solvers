from abc import ABC, abstractmethod

class Solver(ABC):
    
    # ======================= #
    # Protected attributes
    # ======================= #
    # _max_iterations              # max number of iterations
    # _tolerance                   # tolerance value for the solution
    # _output_info                 # boolean: switch on/off additional info in output
    # _solver_step                 # step used by the solver for computing the solution
    # _step_variation              # boolean: switch on/off automatic step variation

    # ======================= #
    # Constructor
    # ======================= #
    def __init__( self , 
                  max_iter: float = 2e2   , 
                  tol: float      = 1e-5  ,
                  info: bool      = False ,
                  step: float     = 1e-3  ,
                  step_adpt: bool = False ) -> None:
        
        self._max_iterations = max_iter
        self._tolerance      = tol
        self._output_info    = info
        self._solver_step    = step
        self._step_variation = step_adpt

    # ======================= #
    # Public methods
    # ======================= #
    # Set methods
    @abstractmethod
    def set_max_iterations(self, max_iter) ->None:
        """ Set the maximum number of iterations (or function evaluetions)
            that the solver can perform to find a solution """
        raise NotImplementedError
    
    