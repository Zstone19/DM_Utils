import numpy as np
from numba import njit, prange




def get_r_bounds(xline, xcon, r_input, t_input):
    rmin_set = 0
    
    
    tspan_data = xline.max() - xline.min()
    tspan_con = xcon.max() - xcon.min()
    tcad_data = max( np.diff(xline).max(), np.diff(xcon).max() )
    
    tset = tspan_con + (xcon[0] - xline[0])
    tset = max( 2*tcad_data, tset )
    
    
    
    dt = xcon[0] - tset
    if (r_input > 0):
        dt = max( dt , xline[0] - r_input*2 )
    elif (tset > 0):
        dt = max( dt, xcon[0] - t_input )
    


    rmax_set = tspan_data/2
    
    rmax_set = min( rmax_set, (xline[0] - xcon[0] + tset)/2  )
    if r_input > 0:
        rmax_set = min( rmax_set, r_input )
        
    return rmin_set, rmax_set





############################################################################################################
################################################# CLOUDS ###################################################
############################################################################################################

@njit(fastmath=True)
def theta_sample_inner(gamma, theta_opn_cos1, theta_opn_cos2):
    return np.arccos( theta_opn_cos1 + (theta_opn_cos2 - theta_opn_cos1) * np.random.random_sample()**(1./gamma) )

@njit(fastmath=True)
def theta_sample_outer(gamma, theta_opn_cos1, theta_opn_cos2):
    a1 = np.arccos(theta_opn_cos1)
    a2 = np.arccos(theta_opn_cos2)
    return a2 + (a1-a2)* (  1. - np.random.random_sample()**(1./gamma)  )




@njit(fastmath=True)
def generate_clouds(model_params, n_cloud_per_core, rcloud_max_set, rcloud_min_set, n_v_per_cloud):
    
    #Param order:
    #mbh, mu, beta, F, inc, cos_theta_opn, kappa, gamma, xi, f_ellip, f_flow, sigr_circ, sigth_circ, sigr_rad, sigth_rad, theta_e, sig_turb


    #Constants
    GRAVITY = 6.672e-8
    SOLAR_MASS = 1.989e33
    CVAL = 2.9979e10
    CM_PER_LD = CVAL*8.64e4
    VEL_UNIT = np.sqrt( GRAVITY * 1.0e6 * SOLAR_MASS / CM_PER_LD ) / 1.0e5
    C_UNIT = CVAL/1.0e5/VEL_UNIT


    #Arrays
    cloud_weights = np.zeros(n_cloud_per_core)
    cloud_taus = np.zeros(n_cloud_per_core)
    cloud_coords = np.zeros((n_cloud_per_core, 3))
    cloud_pcoords = np.zeros((n_cloud_per_core, 2))
    cloud_vels = np.zeros((n_cloud_per_core, n_v_per_cloud, 3))
    cloud_vels_los = np.zeros((n_cloud_per_core, n_v_per_cloud))
    
    
    mu = np.exp(model_params[0])                                
    beta = model_params[1]
    F = model_params[2]
    inc = np.arccos(model_params[3])                            #Inc [rad]
    cos_theta_opn = np.cos(model_params[4] * np.pi/180)         #Cos(theta_opn)
    kappa = model_params[5]                                     
    gamma = model_params[6]
    xi = model_params[7]
    mbh = np.exp(model_params[8])                               #MBH [1e6 solar masses]
    
    f_ellip = model_params[9]
    f_flow = model_params[10]
    sigr_circ = np.exp(model_params[11])
    sigth_circ = np.exp(model_params[12])
    sigr_rad = np.exp(model_params[13])
    sigth_rad = np.exp(model_params[14])
    theta_e = model_params[15] * np.pi/180
    sig_turb = np.exp(model_params[16])
    
    
    a = 1./beta/beta
    s = mu/a
    Rs = 3e10*mbh / CM_PER_LD
    Rin = mu*F + Rs
    sigma = (1. - F)*s

    sin_inc_comp = np.cos(inc)
    cos_inc_comp = np.sin(inc)   
    
    for j in prange(n_cloud_per_core):
        coords = np.zeros(3)   #Cartesian
        pcoords = np.zeros(2)  #Polar
        Lvec = np.zeros(2)     #Angular momentum vec (polar)
        vels = np.zeros(3)
        pvels = np.zeros(2)

        
        Lvec[0] = 2*np.pi*np.random.random_sample()                     #L_phi
        Lvec[1] = theta_sample_outer( gamma, cos_theta_opn, 1. )        #L_theta
        
        nc = 0
        pcoords[0] = rcloud_max_set + 1.
        while ( (pcoords[0] > rcloud_max_set) or (pcoords[0] < rcloud_min_set) ):
            if nc > 1000:
                raise Exception('Cloud generation failed')

            rnd = np.random.standard_gamma(a)
            pcoords[0] = Rin + rnd*sigma    #r
            nc += 1
    
    
    
    
    
        #############################
        # Get cloud coords
    
        coords_b = np.zeros(3)
        
        #Azimuthal positions of clouds
        pcoords[1] = 2*np.pi*np.random.random_sample()   #phi
        cloud_pcoords[j] = pcoords
        
        #Polar -> Cartesian (in disk)
        coords[0] = pcoords[0] * np.cos(pcoords[1])   #x_disk
        coords[1] = pcoords[0] * np.sin(pcoords[1])   #y_disk
        coords[2] = 0.                                #z_disk
        
        #Right-handed framework
        coords_b[0] = np.cos(Lvec[1])*np.cos(Lvec[0])*coords[0] + np.sin(Lvec[0])*coords[1]
        coords_b[1] = -np.cos(Lvec[1])*np.sin(Lvec[0])*coords[0] + np.cos(Lvec[0])*coords[1]
        coords_b[2] = np.sin(Lvec[1])*coords[0]

        zb0 = coords_b[2]
        rnd_xi = np.random.random_sample()
        if (rnd_xi < 1. - xi) and (zb0 < 0.):
            coords_b[2] = -coords_b[2]
            
            
        #Counter-rotate around y-axis (LOS is x-axis)
        coords[0] = coords_b[0]*cos_inc_comp + coords_b[2]*sin_inc_comp
        coords[1] = coords_b[1]
        coords[2] = -coords_b[0]*sin_inc_comp + coords_b[2]*cos_inc_comp
        
        weight = .5 + kappa*(coords[0]/pcoords[0])
        cloud_weights[j] = weight
        
        dis = pcoords[0] - coords[0]
        cloud_taus[j] = dis
        
        cloud_coords[j] = coords        
        
        #############################
        # Get cloud vels

        v_kep = np.sqrt(mbh/pcoords[0])
        
        for k in prange(n_v_per_cloud):
            rnd = np.random.random_sample()
            
            if rnd < f_ellip:
                rho_v = v_kep*( np.random.standard_normal()*sigr_circ + 1. )
                theta_v = np.pi*( np.random.standard_normal()*sigth_circ + .5 )
            else:
                if f_flow <= .5:
                    rho_v = v_kep*( np.random.standard_normal()*sigr_rad + 1. )                  #Inflow
                    theta_v = np.pi*( np.random.standard_normal()*sigth_rad + 1. ) + theta_e
                else:
                    rho_v = v_kep*( np.random.standard_normal()*sigr_rad + 1. )                  #Outflow
                    theta_v = np.pi*np.random.standard_normal()*sigth_rad + theta_e
                    
        
            pvels[0] = np.sqrt(2.)*rho_v*np.cos(theta_v)
            pvels[1] = rho_v*np.abs(np.sin(theta_v))
            
            
            
            
            vels_b = np.zeros(3)
            
            vels[0] = pvels[0]*np.cos(pcoords[1]) - pvels[1]*np.sin(coords[1])
            vels[1] = pvels[0]*np.sin(pcoords[1]) + pvels[1]*np.cos(coords[1])
            vels[2] = 0.
            
            vels_b[0] = np.cos(Lvec[1])*np.cos(Lvec[0])*vels[0] + np.sin(Lvec[0])*vels[1]
            vels_b[1] = -np.cos(Lvec[1])*np.sin(Lvec[0])*vels[0] + np.cos(Lvec[0])*vels[1]
            vels_b[2] = np.sin(Lvec[1])*vels[0]
            
            if (rnd_xi < 1. - xi) and (zb0 < 0.):
                vels_b[2] = -vels_b[2]
                
            vels[0] = vels_b[0]*cos_inc_comp + vels_b[2]*sin_inc_comp
            vels[1] = vels_b[1]
            vels[2] = -vels_b[0]*sin_inc_comp + vels_b[2]*cos_inc_comp
            cloud_vels[j,k] = vels
            
            #Define LOS velocity - positive is receding w.r.t. observer
            v = -vels[0]
            
            #Add turbulent velocity
            v += np.random.standard_normal()*sig_turb*v_kep
            
            #Make velocity stay physical
            if np.abs(v) >= C_UNIT:
                v = .9999*C_UNIT * np.sign(v)
                
            #Relativistic effects
            g = np.sqrt( (1. + v/C_UNIT) / (1. - v/C_UNIT) ) / np.sqrt( 1. - Rs/pcoords[0] )
            v = (g - 1.)*C_UNIT
            
            cloud_vels_los[j,k] = v


    
    return cloud_weights, cloud_taus, cloud_coords, cloud_pcoords, cloud_vels, cloud_vels_los


############################################################################################################
############################################# TRANSFER FUNCTION ############################################
############################################################################################################

def gkern(l=5, sig=1.):
    """
    
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    kernel = np.exp(-0.5 * np.square(ax) / np.square(sig))
    return kernel / np.sum(kernel)


def generate_tfunc(cloud_taus, cloud_vels, cloud_weights, ntau, psi_v, EPS):
    
    bin_offset = .5
    n_cloud_per_core = len(cloud_taus)
    n_v_per_cloud = cloud_vels.shape[1]
    
    tau_min = cloud_taus.min()
    tau_max = cloud_taus.max()
    dtau = (tau_max - tau_min)/(ntau - 1)
    
    psi_tau = np.array([ tau_min + j*dtau for j in range(ntau) ])
    dv = np.diff(psi_v)[0]
    psi2d = np.zeros((ntau, len(psi_v)))
    
    #Fill Psi 2D
    for j in range(n_cloud_per_core):
        idt = int(  (cloud_taus[j] - tau_min)//dtau  )
        
        for k in range(n_v_per_cloud):
            v_offset = cloud_vels[j,k] + bin_offset*dv
            
            if (v_offset < psi_v[0]) or (v_offset >= psi_v[-1]):
                continue
    
            idv = int(  (v_offset - psi_v[0])//dv  )
            psi2d[idt, idv] += cloud_weights[j]
            
    #Normalize
    Anorm = np.sum(psi2d)*dtau*dv
    Anorm += EPS
    psi2d /= Anorm
    
    
    ngauss = 30
    alpha = (ngauss-1)/2/2
    sig_gauss = (len(psi_v) - 1.)/2/alpha
    kernel = gkern(len(psi_v), sig_gauss)
    
    #Smooth
    for j in range(len(psi_v)):
        # psi2d[:,j] = convolve(psi2d[:,j], Gaussian1DKernel(stddev=sig_gauss))
        psi2d[:,j] = np.convolve(psi2d[:,j], kernel, mode='same') 
    
    return psi_tau, psi_v, psi2d




############################################################################################################
############################################## LIGHT CURVES ################################################
############################################################################################################


# def reconstruct_line2d(model_params, vel_line, nt, line_center):
#     GRAVITY = 6.672e-8
#     SOLAR_MASS = 1.989e33
#     CVAL = 2.9979e10
#     CM_PER_LD = CVAL*8.64e4
#     VEL_UNIT = np.sqrt( GRAVITY * 1.0e6 * SOLAR_MASS / CM_PER_LD ) / 1.0e5
#     C_UNIT = CVAL/1.0e5/VEL_UNIT
    

    
#     nv = len(vel_line)
#     dv = (vel_line[-1] - vel_line[0])/(nv-1) * line_center/C_UNIT
#     ylc_recon = np.zeros(nt)
    
    
    
    
    
#     return t_recon, wl_recon, spec_recon