import numpy as np
import scipy.linalg as sla
from scipy.interpolate import splev, splrep
from scipy.signal import fftconvolve
from numba import njit

###################################################################################################
# SEMISEPARABLE

@njit(fastmath=True)
def compute_invC_semiseparable(x, a1, c1, sigma, syserr):
    n = len(x)
    
    D = np.zeros(n)
    W = np.zeros(n)
    phi = np.zeros(n)
    
    for i in range(1, n):
        phi[i] = np.exp( -c1*(x[i]-x[i-1]) )    

    S = 0.
    A = sigma[0]*sigma[0] + syserr*syserr + a1
    
    D[0] = A
    W[0] = 1./D[0] 
    for i in range(1, n):
        S = phi[i]*phi[i] * (S + D[i-1]*W[i-1]*W[i-1])
        A = sigma[i]*sigma[i] + syserr*syserr + a1
        
        D[i] = A - a1*a1*S
        W[i] = 1./D[i] * (1. - a1*S)  
    
    return phi, D, W


@njit(fastmath=True)
def matmultvec_invC(y, a1, phi, D, W):
    #z = invC y
    n = len(y)
    z = np.zeros(n)
    
    f = 0.
    z[0] = y[0]
    for i in range(1, n):
        f = phi[i] * (f + W[i-1]*z[i-1])
        z[i] = y[i] - a1*f
        
    g = 0.
    for i in range(n-2, -1, -1):
        g = phi[i+1] * (g + a1*z[i+1])
        z[i] = z[i]/D[i] - W[i]*g
        
    return z



@njit(fastmath=True)
def matmult_invC(Y, a1, phi, D, W):
    #Z = invC Y
    n, m = Y.shape
    Z = np.zeros_like(Y)

    for j in range(m):
        f = 0.
        Z[0,j] = Y[0,j]
        
        for i in range(1, n):
            f = phi[i] * (f + W[i-1]*Z[i-1,j])
            Z[i,j] = Y[i,j] - a1*f
            
            
    for j in range(m):
        g = 0.
        Z[-1,j] = Z[-1,j]/D[n-1]
        
        for i in range(n-2, -1, -1):
            g = phi[i+1] * (g + a1*Z[i+1,j])
            Z[i,j] = Z[i,j]/D[i] - W[i]*g


    return np.atleast_2d(Z)


###################################################################################################
# UTILITY 

def inverse_pomat(A):
    n = A.shape[0]
    A1, info = sla.lapack.dpotrf(A)
    A2, info = sla.lapack.dpotri(A1)
    
    Aout = A2.copy()
    for i in range(n):
        for j in range(i):
            Aout[i,j] = Aout[j,i]
            
    return Aout


def chol_decomp_L(A):
    n = A.shape[0]
    A1, info = sla.lapack.dpotrf(A, lower=1)
    
    Aout = A1.copy()
    for i in range(n):
        for j in range(i+1, n):
            Aout[i,j] = 0.
            
    return Aout


@njit(fastmath=True)
def get_Larr(xcon, nq):
    Larr = np.zeros((len(xcon), nq))
    for i in range(len(xcon)):
        for j in range(nq):
            Larr[i,j] = xcon[i]**j    

    return np.atleast_2d(Larr)


@njit(fastmath=True)
def get_covar_Pmat(xcont, yerr_cont, sigma, tau, alpha, syserr):
    
    PSmat = np.zeros((len(xcont), len(xcont)))
    PCmat = np.zeros((len(xcont), len(xcont)))
    
    for i in range(len(xcont)):
        t1 = xcont[i]
        
        for j in range(i):
            t2 = xcont[j]
            PSmat[i,j] = sigma*sigma * np.exp( - (np.abs(t1-t2)/tau)**alpha )
            PSmat[j,i] = PSmat[i,j]
            
            PCmat[i,j] = PSmat[i,j]
            PCmat[j,i] = PCmat[i,j]
            
        
        PSmat[i,i] = sigma*sigma
        
        err2 = yerr_cont[i]*yerr_cont[i] + syserr*syserr 
        PCmat[i,i] = PSmat[i,i] + err2
    
    return PSmat, PCmat


@njit(fastmath=True)
def get_covar_Umat(xcont_recon, xcont, sigma, tau, alpha):
    USmat = np.zeros((len(xcont_recon), len(xcont)))
                     
    for i in range(len(xcont_recon)):
        t1 = xcont_recon[i]
        
        for j in range(len(xcont)):
            t2 = xcont[j]
            USmat[i,j] = sigma*sigma*np.exp(- (np.abs(t1-t2)/tau)**alpha )                     

    return USmat


@njit(fastmath=True)
def get_covar_Pmat_recon(xcont_recon, sigma, tau, alpha):
    PSmat = np.zeros((len(xcont_recon), len(xcont_recon)))
    
    for i in range(len(xcont_recon)):
        t1 = xcont_recon[i]
        
        for j in range(i):
            t2 = xcont_recon[j]
            PSmat[i,j] = sigma*sigma * np.exp( - (np.abs(t1-t2)/tau)**alpha )
            PSmat[j,i] = PSmat[i,j]
    
    return PSmat
    






@njit(fastmath=True)
def get_Sboth_semisep(xcont_recon, xcont, sigma, tau, alpha):
    USmat = np.zeros((len(xcont_recon), len(xcont)))
    
    for i in range(len(xcont_recon)):
        t1 = xcont_recon[i]
        
        for j in range(len(xcont)):
            t2 = xcont[j]
            USmat[i,j] = sigma*sigma * np.exp( - (np.abs(t1-t2)/tau)**alpha )            

    return USmat



@njit(fastmath=True)
def get_Srecon_semisep(xcont_recon, sigma, tau, alpha):
    PSmat = np.zeros((len(xcont_recon), len(xcont_recon)))
    
    for i in range(len(xcont_recon)):
        t1 = xcont_recon[i]
        
        for j in range(i):
            t2 = xcont_recon[j]
            PSmat[i,j] = sigma*sigma * np.exp( - (np.abs(t1-t2)/tau)**alpha )            
            PSmat[j,i] = PSmat[i,j]


    return PSmat




###################################################################################################
# MAIN FUNCTIONS


#NEED TO CHECK THAT THIS WORKS BC ERROR IS ALWAYS NAN
def calculate_cont_from_model(model_params, xcont, ycont, yerr_cont, 
                              xcont_recon, nq, nvar, cont_err_mean):
    
    nrecon_cont = len(xcont_recon)
    sys_err = ( np.exp(model_params[0]) - 1. )*cont_err_mean
    tau = np.exp(model_params[2])
    sigma = np.exp(model_params[1]) * np.sqrt(tau)
    alpha = 1.
    
    #Get model params
    uq = np.atleast_1d(model_params[3:3+nq])   #[sigmad] = uq
    us = model_params[nvar:nvar+nrecon_cont] #timeseries... = us
    
    Larr = get_Larr(xcont, nq)
    Larr_recon = get_Larr(xcont_recon, nq)
    
    
    
    
    
    _, C = get_covar_Pmat(xcont, yerr_cont, sigma, tau, alpha, sys_err)  #(ndat, ndat)
    invC = np.linalg.inv(C)                                              #(ndat, ndat)
    
    S_both = get_covar_Umat(xcont_recon, xcont, sigma, tau, alpha)       #(nrecon, ndat)
    S_recon = get_covar_Pmat_recon(xcont_recon, sigma, tau, alpha)       #(nrecon, nrecon)

    Cq = np.linalg.inv(Larr.T @ invC @ Larr)                             #(nq, nq)
    dCq = np.linalg.cholesky(Cq)                                         #(nq, nq)
    
    yq = (dCq @ uq) + (Cq @ Larr.T @ invC @ ycont)                       #(nq,)
    y_detrend = ycont - (Larr @ yq)                                      #(ndat,)
    
    Q = S_recon - (S_both @ invC @ S_both.T)                             #(nrecon, nrecon)
    yerr_recon2 = np.diag(Q)                                             #(nrecon,)
    
    try:
        dQ = np.linalg.cholesky(Q)                                                #(nrecon, nrecon)
    except:
        dQ = chol_decomp_L(Q)

    yrecon = (S_both @ invC @ y_detrend) + (dQ @ us) + (Larr_recon @ yq) #(nrecon,)
    
    return yrecon, yerr_recon2





def calculate_cont_from_model_semiseparable(model_params, xcont, ycont, yerr_cont, 
                                            xcont_recon, nq, nvar, cont_err_mean):
    
    nrecon_cont = len(xcont_recon)
    sys_err = ( np.exp(model_params[0]) - 1. )*cont_err_mean
    tau = np.exp(model_params[2])
    sigma = np.exp(model_params[1]) * np.sqrt(tau)
    alpha = 1.
    
    #Get model params
    uq = np.atleast_1d(model_params[3:3+nq])   #[sigmad] = uq
    us = model_params[nvar:nvar+nrecon_cont] #timeseries... = us
    
    Larr = get_Larr(xcont, nq)
    Larr_recon = get_Larr(xcont_recon, nq)
    
    
    
    
    
    phi, D, W = compute_invC_semiseparable(xcont, sigma*sigma, 1./tau, yerr_cont, sys_err)
    S_both = get_Sboth_semisep(xcont_recon, xcont, sigma, tau, alpha)    #(nrecon, ndat)
    S_recon = get_Srecon_semisep(xcont_recon, sigma, tau, alpha)         #(nrecon, nrecon)
    Amat = matmult_invC(S_both.T, sigma*sigma, phi, D, W)                #(ndat, nrecon)
    
    invCq = Larr.T @ matmult_invC(Larr, sigma*sigma, phi, D, W)          #(nq, nq)
    Cq = np.linalg.inv(invCq)                                            #(nq, nq)
    dCq = np.linalg.cholesky(Cq)                                         #(nq, nq)
    
    ycq = matmultvec_invC(ycont, sigma*sigma, phi, D, W)                 #(nq,)
    yq = (dCq @ uq) + (Cq @ Larr.T @ ycq)                                #(nq,)
    y_detrend = ycont - (Larr @ yq)                                      #(ndat,)
    
    E2 = S_both @ Amat
    Q = S_recon - E2                                                     #(nrecon, nrecon)
    yerr_recon2 = np.diag(Q)                                             #(nrecon,)
    
    try:
        dQ = np.linalg.cholesky(Q)                                       #(nrecon, nrecon)
    except:
        dQ = chol_decomp_L(Q)

    Amat = matmult_invC(S_both.T, sigma*sigma, phi, D, W)
    yrecon = (Amat.T @ y_detrend) + (dQ @ us) + (Larr_recon @ yq)       #(nrecon,)
    
    return yrecon, yerr_recon2



@njit(fastmath=True)
def calculate_cont_rm(model_params, 
                      xcont_recon, ycont_recon,
                      pow_xcont, xcont_med,
                      idx_resp, flag_trend_diff, nparams_difftrend):
    
    ycont_rm = np.zeros(len(xcont_recon))
    
    A = np.exp(model_params[idx_resp])
    Ag = model_params[idx_resp+1]
    
    if flag_trend_diff > 0:
        
        tmp = 0.
        for m in range(1, nparams_difftrend+1):
            tmp += model_params[idx_resp+m-1] * pow_xcont[m-1]
            
        a0 = -tmp
        
        
        
        for i in range(len(xcont_recon)):
            
            ftrend = a0
            tmp = 1.
            for m in range(1, nparams_difftrend+1):
                tmp *= xcont_recon[i] - xcont_med
                ftrend += model_params[idx_resp+m-1] * tmp
                
            fcont = ycont_recon[i] + ftrend
            if fcont > 0.:
                ycont_rm[i] = A * ( fcont**(1.+Ag) )
            else:
                ycont_rm[i] = 0.    
            
        
    else:
        
        for i in range(len(xcont_recon)):
            fcont = ycont_recon[i]
            if fcont > 0.:
                ycont_rm[i] = A * ( fcont**(1.+Ag) )
            else:
                ycont_rm[i] = 0.
                
                
    return ycont_rm




def interp_cont_rm(tp, spl, xcont_recon, ycont_rm):
    if tp < 0.:
        return ycont_rm[0]    
    elif tp < xcont_recon[-1]:
        return splev(tp, spl)        
    else:
        return ycont_rm[-1]
    
    
def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    kernel = np.exp(-0.5 * np.square(ax) / np.square(sig))
    return kernel / np.sum(kernel)
    
    
def line_gaussian_smooth_2d(line2D_recon, model_params, dv,
                            flag_inst_res, inst_res, inst_res_err,
                            nblr_model, nnlr):
    

    line2D_recon_smooth = np.zeros_like(line2D_recon)
    nl, nv = line2D_recon.shape
    
    
    if flag_inst_res <= 1:
        sigv = inst_res + model_params[nblr_model + nnlr] * inst_res_err
        sigv = max(0., sigv)
        
        for i in range(nl):
            line2D_recon_smooth[i] = fftconvolve(line2D_recon[i], gkern(nv, sigv/dv), mode='same')
        
    else:
        for i in range(nl):
            sigv = inst_res[i] + model_params[nblr_model + nnlr + i] * inst_res_err[i]
            sigv = max(0., sigv)
            
            line2D_recon_smooth[i] = fftconvolve(line2D_recon[i], gkern(nv, sigv/dv), mode='same')         

    
    return line2D_recon_smooth





def line_gaussian_smooth_2dfft(line2D_recon, model_params,
                               psi_v_recon, line2d_recon_time, 
                               flag_inst_res, inst_res, inst_res_err,
                               nblr_model, nnlr,
                               flag_linecenter, linecenter_err, idx_linecenter):
    

    line2D_recon_smooth = np.zeros_like(line2D_recon)
    nl, nv = line2D_recon.shape
    
    npad = int( min(nv*.1, 20) )
    nd_fft = int(nv + 2*npad)
    nd_fft_cal = int(nd_fft/2 + 1)
    
    
    dv = np.diff(psi_v_recon)[0]
    
    if flag_inst_res <= 1:
        sigv = inst_res + model_params[nblr_model + nnlr] * inst_res_err
        sigv = max(0., sigv)        
        
        resp_fft0 = np.zeros((nd_fft_cal, 2))
        resp_fft0[:,0] = np.exp( -2. * np.pi*np.pi * (sigv/dv)*(sigv/dv) * np.arange(0, nd_fft_cal)*np.arange(0, nd_fft_cal) /nd_fft/nd_fft ) / nd_fft

        for i in range(nl):
            if flag_linecenter == 0:
                linecenter = 0.
            elif flag_linecenter > 0:
                linecenter = model_params[idx_linecenter] * linecenter_err
            elif flag_linecenter < 0:
                linecenter = model_params[idx_linecenter + j] * linecenter_err
                
            resp_fft = np.zeros((nd_fft_cal, 2))
            resp_fft[:,0] = resp_fft[:,0] * np.cos( 2.*np.pi*linecenter/dv * np.arange(0, nd_fft_cal)/nd_fft )
            resp_fft[:,1] = -resp_fft[:,0] * np.sin( 2.*np.pi*linecenter/dv * np.arange(0, nd_fft_cal)/nd_fft )

            real_data = np.zeros(nd_fft)
            real_data[npad:-npad] = line2D_recon[i]
            data_fft = np.fft.fft(real_data, n=nd_fft)
            
            
            conv_fft = np.zeros((nd_fft_cal, 2))
            conv_fft[:,0] = np.real(data_fft)*resp_fft[:,0] - np.imag(data_fft)*resp_fft[:,1]
            conv_fft[:,1] = np.real(data_fft)*resp_fft[:,1] + np.imag(data_fft)*resp_fft[:,0]
             
            real_conv = np.real(  np.fft.ifft(conv_fft, n=nd_fft)  )
            line2D_recon_smooth = real_conv[npad:-npad]            

        
        
    else:
        for i in range(nl):
            sigv = inst_res[i] + model_params[nblr_model + nnlr + i] * inst_res_err[i]
            sigv = max(0., sigv)
            
            
            
    
    return line2D_recon_smooth
    


@njit(fastmath=True)
def calculate_line2d_from_model_spl(psi_v, psi_tau, psi2D,
                                line2d_recon_time,
                                xcont_recon, ycont_rm):
    
    #Velocity of tfunc and line2d must be the same
    spl = splrep(xcont_recon, ycont_rm)

    line2D_recon = np.zeros( ( len(line2d_recon_time), len(psi_v) ) )
    dt = psi_tau[1] - psi_tau[0]
    nl = len(line2d_recon_time)
    nv = len(psi_v)
        
    for i in range(nl):
        tc_vals = line2d_recon_time[i] - psi_tau

        fcon_rm_vals = np.full( len(tc_vals), np.nan )
        fcon_rm_vals[ tc_vals < 0. ] = ycont_rm[0]
        fcon_rm_vals[ tc_vals > xcont_recon[-1] ] = ycont_rm[-1]
        
        good_mask = ( (tc_vals >= 0.) & (tc_vals <= xcont_recon[-1]) )
        fcon_rm_vals[good_mask] = splev(tc_vals[good_mask], spl)
            
        for j in range(nv):            
            line2D_recon[i,j] = np.sum( np.multiply(psi2D[:,j], fcon_rm_vals) ) * dt
    
    
    return line2D_recon




@njit(fastmath=True)
def calculate_line2d_from_model(psi_v, psi_tau, psi2D,
                                line2d_recon_time,
                                xcont_recon, ycont_rm):
    
    #Velocity of tfunc and line2d must be the same

    line2D_recon = np.zeros( ( len(line2d_recon_time), len(psi_v) ) )
    dt = psi_tau[1] - psi_tau[0]
    nl = len(line2d_recon_time)
    nv = len(psi_v)
        
    for i in range(nl):
        tc_vals = line2d_recon_time[i] - psi_tau

        fcon_rm_vals = np.full( len(tc_vals), np.nan )
        fcon_rm_vals[ tc_vals < 0. ] = ycont_rm[0]
        fcon_rm_vals[ tc_vals > xcont_recon[-1] ] = ycont_rm[-1]
        
        good_mask = ( (tc_vals >= 0.) & (tc_vals <= xcont_recon[-1]) )
        fcon_rm_vals[good_mask] = np.interp(tc_vals[good_mask], xcont_recon, ycont_rm)
            
        for j in range(nv):            
            line2D_recon[i,j] = np.sum( np.multiply(psi2D[:,j], fcon_rm_vals) ) * dt
    
    
    return line2D_recon
