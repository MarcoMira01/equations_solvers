import numpy as np
from thermal_model.thermal_model_linearized import thermal_model_linearized

def thermal_model_solver_ivp( t_span , Delta_t , T0 , Ta , qext ):

    sim_time = np.arange( t_span[0] , t_span[1]+Delta_t , Delta_t )

    T_sim = np.zeros( (len(T0),len(sim_time)) )
    T_sim[:,0] = T0
    i = 1
    for t in sim_time[1:]:

        Ta_smp   = np.interp( np.round(t,4) , Ta[:,0]   , Ta[:,1]   )
        qext_smp = np.interp( np.round(t,4) , qext[:,0] , qext[:,1] )

        T_sim_tmp = thermal_model_linearized( Delta_t , T_sim[:,i-1] , Ta_smp , qext_smp ) 
        T_sim[:,i] = T_sim_tmp

        i = i+1

    return sim_time , T_sim