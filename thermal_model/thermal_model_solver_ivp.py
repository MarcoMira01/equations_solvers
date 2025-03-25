# This funtion implements a solver tailored for the thermal model
# For the moment, only the linearized model is taken into cosideration
# Possible future developments include the implementation of nonlinear
# solvers
#
# Different methods can be selected to change the computation strategy:
#  - BE_linear: simulate the model via Backward Euler method integration
#  - BE_linear_SD_xxx: use of steepest descent algorithm with xxx step update:
#       - lin_search
#       - conj_grad
#       - back_track_iter
#       - barzilai_borwein_short
#       - barzilai_borwein_long
#       - barzilai_borwein_alt
#       - random_step

import numpy as np
from thermal_model.thermal_matrix_continuous import thermal_matrix_continuous
from thermal_model.thermal_model_linearized  import thermal_model_linearized
from my_solvers.solvers.Solver_SD_LS         import Solver_SD_LS

def thermal_model_solver_ivp( t_span , Delta_t , T0 , Ta , qext , method ):

    sol = Solver_SD_LS( max_iter = 1e3 , tol = 1e-8 )

    n_iter_avg = 0

    # Matrices of the linearized thermal model
    C, A_cnv, A_cnv_tld, A_cnd, A_rad, B_ext = thermal_matrix_continuous( )

    # Vector of simulation time
    sim_time = np.arange( t_span[0] , t_span[1]+Delta_t , Delta_t )

    # Inizialize temperature vector
    T_sim = np.zeros( (len(T0),len(sim_time)) )
    T_sim[:,0] = T0
    i = 1
    for t in sim_time[1:]:
        
        # Previous temperature
        T = T_sim[:,i-1]
        T3 = np.power( T , 3 )
        T4 = np.power( T , 4 )

        # Interpolate inputs at current t
        Ta_t   = np.interp( np.round(t,4) , Ta[:,0]   , Ta[:,1]   )
        qext_t = np.interp( np.round(t,4) , qext[:,0] , qext[:,1] )

        Ta_t_vec  = Ta_t*np.ones( len(T) )
        Ta_t_vec4 = np.power( Ta_t_vec , 4 )
        
        f_t  = -A_rad.dot(T4) + (A_cnd-A_cnv).dot(T) + A_rad.dot(Ta_t_vec4) + A_cnv_tld.dot(Ta_t_vec) + B_ext.dot(qext_t)
        
        tmp_A = 4*A_rad.dot(np.diag(T3)) - A_cnd + A_cnv
        A = C + Delta_t*tmp_A

        if (method == 'BE_linear'):
            T_new = T + Delta_t*np.linalg.inv(A).dot(f_t)
            n_iter_avg = n_iter_avg + 1
        else:
            b = Delta_t*f_t + A.dot(T)
            T_new = sol.solve( (A.T).dot(A) , (A.T).dot(b) , T , method = method )
            n_iter_avg = n_iter_avg + sol.get_n_iter()/(len(sim_time)-1)

        T_sim[:,i] = T_new

        i = i+1

    return sim_time , T_sim , n_iter_avg
