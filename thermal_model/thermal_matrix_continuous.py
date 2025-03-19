# This function returns the matrices of the continuous nonlinear model
#
# We consider a bar of fixed length 0.3m (30cm)
# The bar is divided into three main sections of length 0.1m (10cm) with different thermal coefficients
# The simulation is performed via finite-difference-method by defining 300 silces of length Delta_x = 0.01m (1cm)
# Other dimensions of the bar are:
#  - height = 0.01m (1cm)
#  - depth  = 0.04m (4cm)
#
# Section 1: aluminium
#  - specific heat capacity: c1   = 897    J*Kg^-1*K^-1
#  - density:                rho1 = 2.7    Kg*m^-3
#  - convection coeff.:      h1   = 5      W*m^-2*K^-1
#  - emissivity:             emi1 = 0.04   -
#  - conduction coeff.:      k1   = 237    W*m^-1*K^-1
# 
# Section 2: iron
#  - specific heat capacity: c2   = 449    J*Kg^-1*K^-1
#  - density:                rho2 = 7.87   Kg*m^-3
#  - convection coeff.:      h2   = 5      W*m^-2*K^-1
#  - emissivity:             emi2 = 0.25   -
#  - conduction coeff.:      k2   = 79     W*m^-1*K^-1
#
# Section 3: silver
#  - specific heat capacity: c3   = 223    J*Kg^-1*K^-1
#  - density:                rho3 = 10.5   Kg*m^-3
#  - convection coeff.:      h3   = 5      W*m^-2*K^-1
#  - emissivity:             emi3 = 0.03   -
#  - conduction coeff.:      k3   = 406    W*m^-1*K^-1

import numpy as np
from scipy.constants import Stefan_Boltzmann as sb

def thermal_matrix_continuous( ):
    
    #--------------------------------------------------------#
    # Bar dimensions
    #--------------------------------------------------------#
    N_section = 100
    H = 0.01
    D = 0.04
    Delta_x = 0.01

    N_sec_inp = 100

    #--------------------------------------------------------#
    # Thermal coefficients definition
    #--------------------------------------------------------#
    # specific heat capacity
    c = np.array( [ 897 , 449 , 223 ] )

    # density
    rho = np.array( [ 2.7 , 7.87 , 10.5 ] )

    # convective coefficient
    h = np.array( [ 5 , 5 , 5 ] )

    # emissivity
    emi = np.array( [ 0.04 , 0.25 , 0.03 ] )

    # conductivity
    k = np.array( [ 237 , 79 , 406 ] )

    #--------------------------------------------------------#
    # Matrices creation
    #--------------------------------------------------------#
    # Heat capacities: C = c*rho*In
    tmp = np.multiply(c,rho)
    tmp1 = tmp[0]*np.ones(N_section)
    tmp2 = tmp[1]*np.ones(N_section)
    tmp3 = tmp[2]*np.ones(N_section)
    tmp4 = np.concatenate( ( tmp1 , tmp2 , tmp3 ) )
    C = np.diag( tmp4 ) 

    # Convective term
    alpha_cv  = 2*h*( (H+D)/(H*D) )
    tmp1      = alpha_cv[0]*np.ones(N_section)
    tmp2      = alpha_cv[1]*np.ones(N_section)
    tmp3      = alpha_cv[2]*np.ones(N_section)
    tmp4      = np.concatenate( ( tmp1 , tmp2 , tmp3 ) )
    A_cnv     = np.diag( tmp4 ) 
    A_cnv_tld = A_cnv - np.diag( np.concatenate( ( [2*h[0]/Delta_x] , np.zeros(3*N_section-2) , [2*h[2]/Delta_x] ) ) )

    # Conductive term
    alpha_cd = k/( np.power(Delta_x,2) )
    # Main diagonal
    tmp1      = alpha_cd[0]*np.concatenate( ( [2*( (h[0]*Delta_x)/k[0] - 1 )] , -2*np.ones(N_section-1) ) )
    tmp2      = alpha_cd[1]*(-2)*np.ones(N_section)
    tmp3      = alpha_cd[2]*np.concatenate( ( -2*np.ones(N_section-1) , [2*( (h[2]*Delta_x)/k[2] - 1 )] ) )
    Atmp_1    = np.diag( np.concatenate( ( tmp1 , tmp2 , tmp3 ) ) )
    # Upper diagonal
    tmp1      = alpha_cd[0]*np.concatenate( ( [2] , np.ones(N_section-1) ) )
    tmp2      = alpha_cd[1]*np.ones(N_section)
    tmp3      = alpha_cd[2]*np.ones(N_section-1)
    Atmp_2    = np.diag( np.concatenate( ( tmp1 , tmp2 , tmp3 ) ) , k= 1 )
    #Lower diagonal
    tmp1      = alpha_cd[0]*np.ones(N_section-1)
    tmp2      = alpha_cd[1]*np.ones(N_section)
    tmp3      = alpha_cd[2]*np.concatenate( ( np.ones(N_section-1) , [2] ) )
    Atmp_3    = np.diag( np.concatenate( ( tmp1 , tmp2 , tmp3 ) ) , k=-1 )
    # Resultant
    A_cnd     = Atmp_1 + Atmp_2 + Atmp_3


    # Radiative term
    alpha_r = 2*(emi*sb)*( (H+D)/(H*D) )
    tmp1    = alpha_r[0]*np.ones(N_section)
    tmp2    = alpha_r[1]*np.ones(N_section)
    tmp3    = alpha_r[2]*np.ones(N_section)
    tmp4    = np.concatenate( ( tmp1 , tmp2 , tmp3 ) )
    A_rad   = np.diag( tmp4 )

    # Input term
    B_ext = (1/H)*np.concatenate( ( np.ones( N_sec_inp ) , np.zeros( 3*N_section - N_sec_inp ) ) )

    return C, A_cnv, A_cnv_tld, A_cnd, A_rad, B_ext