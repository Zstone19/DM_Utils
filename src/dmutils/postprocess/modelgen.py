import numpy as np
import sys
from numba import njit, prange

from .modelmath import (calculate_cont_from_model, 
                        calculate_cont_rm, calculate_line2d_from_model)


def get_rset_tset(xline, xcon, r_input, t_input):
    rmin_set = 0
    
    
    tspan_data_con = xcon.max() - xcon.min()
    tspan_data = xline.max() - xcon.min()    
    tcad_data = np.max( [np.diff(xline).max(), np.diff(xcon).max(), tspan_data] )
    
    tset = tspan_data_con + (xcon[0] - xline[0])
    tset = max( 2*tcad_data, tset )
    
    
    
    dt = xcon[0] - tset
    if (r_input > 0):
        dt = max( dt , xline[0] - r_input*2 )
    elif (t_input > 0):
        dt = max( dt, xcon[0] - t_input )
    
    tset = xcon[0] - dt



    rmax_set = tspan_data/2
    
    rmax_set = min( rmax_set, (xline[0] - xcon[0] + tset)/2  )
    if r_input > 0:
        rmax_set = min( rmax_set, r_input )
        
    return rmin_set, rmax_set, tset


class DM_Data:
    
    def __init__(self, bp, paramfile_inputs):
        
        #All fluxes and velocities will be scaled
        #Use VEL_UNIT to convert to km/s
        #Use cont_scale and line_scale to convert to input flux units
        
        ################################
        #Constants
        self.GRAVITY = 6.672e-8
        self.SOLAR_MASS = 1.989e33
        self.CVAL = 2.9979e10
        self.CM_PER_LD = self.CVAL*8.64e4
        self.VEL_UNIT = np.sqrt( self.GRAVITY * 1.0e6 * self.SOLAR_MASS / self.CM_PER_LD ) / 1.0e5
        self.C_UNIT = self.CVAL/1.0e5/self.VEL_UNIT
        
        self.EPS = sys.float_info.epsilon
        
        
        self.z = float(paramfile_inputs['redshift'])
        self.central_wl = float(paramfile_inputs['linecenter'])
        self.ncloud = int(paramfile_inputs['ncloudpercore'])
        self.vel_per_cloud = int(paramfile_inputs['nvpercloud'])
        self.nrecon_cont = int(paramfile_inputs['nconrecon'])
        self.nrecon_line = int(paramfile_inputs['nlinerecon'])
        self.nrecon_vel = int(paramfile_inputs['nvelrecon'])
        
        self.flag_linecenter = int(paramfile_inputs['flaglinecenter'])
        self.flag_trend = int(paramfile_inputs['flagtrend'])
        self.flag_trend_diff = int(paramfile_inputs['flagtrenddiff'])
        self.flag_inst_res = int(paramfile_inputs['flaginstres'])
        self.flag_nl = int(paramfile_inputs['flagnarrowline'])
        
        self.inst_res = float(paramfile_inputs['instres'])
        self.inst_res_err = float(paramfile_inputs['instreserr'])
        self.r_input = float(paramfile_inputs['rcloudmax'])
        self.t_input = float(paramfile_inputs['timeback'])
        
            #In observed-frame
        self.xcont, self.ycont, self.yerr_cont = bp.data['con_data'].T
        self.xline = bp.data['line2d_data']['time']
        self.xcont_out = self.xcont.copy()
        self.xline_out = self.xline.copy()
        
            #Put in rest-frame
        self.xcont /= (1+self.z)
        self.xline /= (1+self.z)

        

        self.ycont_out = self.ycont.copy()
        self.yerr_cont_out = self.yerr_cont.copy()
    
        
        self.r_input = float(paramfile_inputs['rcloudmax'])
        self.t_input = float(paramfile_inputs['timeback'])
        self.rmin, self.rmax, self.timeback = get_rset_tset(self.xline, self.xcont, self.r_input, self.t_input)

        self.ntau = len(bp.results['tau_rec'][0])
        self.wl_vals = bp.data['line2d_data']['profile'][0,:,0]/(1+self.z)
        self.vel_line = self.C_UNIT*( self.wl_vals - self.central_wl )/self.central_wl
        self.line2d = bp.data['line2d_data']['profile'][:,:,1]
        self.line2d_err = bp.data['line2d_data']['profile'][:,:,2] 
        
        ################################
        # Get nparams
        self.nq = 1 + self.flag_trend
        self.ntrend = self.nq
        self.ndrw = 3
        self.nresp = 2
        
        if self.flag_trend_diff > 0:
            self.ndifftrend = 1
        else:
            self.ndifftrend = 0
            
        self.nvar = self.ndrw + self.ntrend + self.nresp + self.ndifftrend
        
        self.nnlr = 0
        if self.flag_nl >= 2:
            self.nnlr = 3
        
        self.nres = 1
        if self.flag_inst_res > 1:
            self.nres = len(self.xline)
            
        self.nlinecenter = 0
        if self.flag_linecenter > 0:
            self.nlinecenter = 1
        elif self.flag_linecenter < 0:
            self.nlinecenter = len(self.xline)
            
            
            
        ################################
        #Get xcont_recon
        tspan = self.xcont.max() - self.xcont.min()
        xcont_recon_min = self.xcont.min() - self.timeback - 10.
        
        xcont_recon_max = self.xcont.max() + max(.05*tspan, 20.)
        xcont_recon_max = max( xcont_recon_max, self.xline.max() + 10. )

        dt = (xcont_recon_max - xcont_recon_min)/(self.nrecon_cont-1)
        self.xcont_recon = np.array([ xcont_recon_min + j*dt for j in range(self.nrecon_cont) ])

        
        ################################
        #Get xline_recon
        xline_recon_min = self.xline.min() - min( .1*(self.xline.max() - self.xline.min()), 10. )
        if self.t_input <= 0.:
            xline_recon_min = max( xline_recon_min, xcont_recon_min + self.timeback )

        xline_recon_max = self.xline.max() + min( .1*(self.xline.max() - self.xline.min()), 10. )
        xline_recon_max = min( xline_recon_max, xcont_recon_max - 1. )
        
        dt = (xline_recon_max - xline_recon_min)/(self.nrecon_line-1)
        self.xline_recon = np.array([ xline_recon_min + j*dt for j in range(self.nrecon_line) ])
        
        
        ################################
        # Get pow_xcont, xcont_med
        tspan_cont = self.xcont.max() - self.xcont.min()    
        tspan = self.xline.max() - self.xcont.min()
        self.xcont_med = .5*(self.xcont.max() + self.xcont.min())
        
        self.pow_xcont = np.zeros(self.ndifftrend)
        for i in range(self.ndifftrend):
            self.pow_xcont[i] = ( (self.xcont.max() - self.xcont_med)**(i+2) - (self.xcont.min() - self.xcont_med)**(i+2) )/(i+2)/tspan_cont

        
        ################################
        # Get idx_resp
        
        #For P14 model
        self.nblrmodel = 17
        self.nblr = self.nblrmodel + self.nnlr + self.nres + self.nlinecenter + 1
        
        self.nparams = self.nrecon_cont + self.nblr + self.nvar
        self.idx_resp = self.nblr + self.ndrw + self.ntrend
        self.idx_difftrend = self.idx_resp + self.nresp
        self.idx_linecenter = self.nblrmodel + self.nnlr + self.nres

        
        ################################
        # Get line LC
        
        dv = np.diff(self.vel_line)[0]
        
        self.yline = np.zeros(len(self.xline))
        self.yerr_line = np.zeros(len(self.xline))
        
        
        self.line_err_mean = np.mean(self.line2d_err)
        for i in range(len(self.xline)):
            self.yline[i] = self.line2d[i,0]/2.
            self.yerr_line[i] = self.line2d_err[i,0]*self.line2d_err[i,0]/2. 
            
            for j in range(1, len(self.vel_line)):
                self.yline[i] += self.line2d[i,j]
                self.yerr_line[i] += self.line2d_err[i,j]*self.line2d_err[i,j]   
                
            self.yline[i] += self.line2d[i,-1]/2.
            self.yerr_line[i] += self.line2d_err[i,-1]*self.line2d_err[i,-1]/2.
            
            self.yline[i] *= dv
            self.yerr_line[i] = np.sqrt(self.yerr_line[i])*dv
            
            
        ################################
        # Rescale light curves
        
        self.cont_err_mean = np.mean(self.yerr_cont)
        
        cont_avg = np.mean(self.ycont)
        self.cont_scale = 1/cont_avg
        
        line_avg = np.mean(self.yline)
        self.line_scale = 1/line_avg
        
        self.ycont *= self.cont_scale
        self.yerr_cont *= self.cont_scale
        self.cont_err_mean *= self.cont_scale
        
        self.yline *= self.line_scale
        self.yerr_line *= self.line_scale
        self.line2d *= self.line_scale
        self.line2d_err *= self.line_scale
        self.line_err_mean *= self.line_scale
            
        ################################
        # Extend velocities
        
        self.nvel_data_incr = 5
        self.nvel_data_ext = len(self.vel_line) + 2*self.nvel_data_incr

        dv = np.diff(self.vel_line)[0]
        self.vel_line_ext = np.zeros(self.nvel_data_ext)
        for i in range(self.nvel_data_incr+1):
            self.vel_line_ext[i] = self.vel_line[0] - (self.nvel_data_incr - i)*dv
            self.vel_line_ext[-1-i] = self.vel_line[-1] + (self.nvel_data_incr - i)*dv

        for i in range(len(self.vel_line)):
            self.vel_line_ext[i+self.nvel_data_incr] = self.vel_line[i]


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

@njit(fastmath=True)
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

def get_cont_line2d_recon(model_params, data, ycont_recon, yerr_cont_recon):

    ################################
    # Run
    ycont_rm, line2D_recon = reconstruct_line2d(model_params, data, ycont_recon)    

    ################################
    # Get line LCs
    
    dv = np.diff(data.vel_line_ext)[0]
    
    yline_recon = np.zeros(len(data.xline_recon))        
    for i in range(len(data.xline_recon)):
        yline_recon[i] = line2D_recon[i,0]/2.
        
        for j in range(1, len(data.vel_line_ext)):
            yline_recon[i] += line2D_recon[i,j]
            
        yline_recon[i] += line2D_recon[i,-1]/2.
        yline_recon[i] *= dv
        
    
    
    #Put light curve times into observed frame
    xcont_recon = data.xcont_recon.copy()
    xline_recon = data.xline_recon.copy()
    xcont_recon *= (1+data.z)
    xline_recon *= (1+data.z)
    
    #Rescale light curves
    ycont_recon_out = ycont_recon.copy()
    yerr_cont_recon_out = yerr_cont_recon.copy()
    ycont_recon_out /= data.cont_scale
    yerr_cont_recon_out /= data.cont_scale
    ycont_rm /= data.cont_scale
    line2D_recon /= data.line_scale
    yline_recon /= data.line_scale
    
    #Rescale velocities
    vel_line_ext_out = data.vel_line_ext.copy()
    vel_line_ext_out *= data.VEL_UNIT
    
    return xcont_recon, ycont_recon_out, yerr_cont_recon_out, ycont_rm, xline_recon, yline_recon, vel_line_ext_out, line2D_recon
    

def reconstruct_line2d(model_params, data, ycont_recon):
    
    #Reconstruct continuum
    ycont_rm = calculate_cont_rm(model_params, data.xcont_recon, ycont_recon, 
                                 data.pow_xcont, data.xcont_med, 
                                 data.idx_resp, data.flag_trend_diff, data.ndifftrend)
    
    
    #Get transfer function (at data pts)
    cloud_weights, cloud_taus, _, _, _, cloud_vels_los = generate_clouds(model_params, data.ncloud, data.rmax, data.rmin, data.vel_per_cloud)
    psi_tau, _, psi2D = generate_tfunc(cloud_taus, cloud_vels_los, cloud_weights, data.ntau, data.vel_line_ext, data.EPS)

    #Get line2D
    line2D_recon = calculate_line2d_from_model(model_params, data.vel_line_ext, psi_tau, psi2D,
                                               data.xline_recon, data.xcont_recon, ycont_rm,
                                               data.flag_inst_res, data.inst_res, data.inst_res_err,
                                               data.nblrmodel, data.nnlr)
    
    
    return ycont_rm, line2D_recon
