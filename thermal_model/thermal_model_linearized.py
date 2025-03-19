# Implementation of nonlinear continuous thermal model
# The additional inputs are
#  - T_a: evironment temperature
#  - q_ext: heat in input
#
# T_a and q_ext must be 2D vectors with respect to time ( T_a = [ Ta_time , Ta_value ] )
# The function resample the correct value with respect to the simulation time t

import numpy as np
from thermal_model.thermal_matrix_continuous import thermal_matrix_continuous
from thermal_model.thermal_model_nonlinear import thermal_model_nonlinear

def thermal_model_linearized( t , Delta_t , T , T_a , q_ext ):
    
    C, A_cnv, A_cnv_tld, A_cnd, A_rad, B_ext = thermal_matrix_continuous( )

    Ta_smp   = np.interp( t , T_a[:,0]   , T_a[:,1]   )
    qext_smp = np.interp( t , q_ext[:,0] , q_ext[:,1] )

    T3 = np.power( T , 3 )
    T4 = np.power( T , 4 )

    Ta_vec  = Ta_smp*np.ones( len(T) )
    Ta_vec4 = np.power( Ta_vec , 4 )

    f_t    = -A_rad.dot(T4) + (A_cnd-A_cnv).dot(T) + A_rad.dot(Ta_vec4) + A_cnv_tld.dot(Ta_vec) + B_ext.dot(qext_smp)
    # dTdt = thermal_model_nonlinear( t , T , T_a , q_ext )
    
    tmp_A = 4*A_rad.dot(T3) - A_cnd + A_cnv
    # Cinv = np.linalg.inv(C)
    I = np.ones(A_rad.shape)
    A = C + Delta_t*tmp_A
    
    T_new = T + Delta_t*np.linalg.inv(A).dot(f_t)

    return T_new