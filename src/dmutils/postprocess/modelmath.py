import numpy as np
import scipy.linalg as sla
from scipy.interpolate import splev, splrep
from scipy.signal import fftconvolve
from numba import njit

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
    return Larr


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
    



###################################################################################################
# MAIN FUNCTIONS

def calculate_cont_from_model(model_params, xcont, ycont, yerr_cont, 
                              xcont_recon, nq, nvar):
    
    nrecon_cont = len(xcont_recon)
    cont_err_mean = np.mean(yerr_cont)
    sys_err = ( np.exp(model_params[0]) - 1. )*cont_err_mean
    tau = np.exp(model_params[2])
    sigma = np.exp(model_params[1]) * np.sqrt(tau)
    alpha = 1.
    
    #Get model params
    yfit = np.atleast_1d(model_params[3:3+nq])   #[sigmad]
    yfit2 = model_params[nvar:nvar+nrecon_cont] #timeseries...
    
    Larr = get_Larr(xcont, nq)
    Larr_recon = get_Larr(xcont_recon, nq)
    
    
    
    
    
    PSmat, PCmat = get_covar_Pmat(xcont, yerr_cont, sigma, tau, alpha, sys_err) #(ndat, ndat)
    USmat = get_covar_Umat(xcont_recon, xcont, sigma, tau, alpha)               #(nrecon, ndat)
    
    PCmat = inverse_pomat(PCmat)                                             #(ndat, ndat)
    
    # inv(Cq) = L^T inv(C) L
    ybuf = PCmat @ Larr                                                      #(ndat, nq)
    Cq = Larr.T @ ybuf                                                       #(nq, nq)
    
    #yq = L^T inv(C) y
    ybuf = PCmat @ ycont                                                     #(ndat,)
    yq = Larr.T @ ybuf                                                       #(nq,)

    #ybuf = Cq yq
    Cq = inverse_pomat(Cq)                                                   #(nq, nq)
    ybuf = Cq @ yq                                                           #(nq,)
    
    Cq = chol_decomp_L(Cq)                                                   #(nq, nq)
    yq = Cq @ yfit                                                            #(nq,)
    yq += ybuf                                                               #(nq,)
    
    ybuf = Larr @ yq                                                         #(ndat,)
    y = ycont - ybuf                                                         #(ndat,)
    
    ybuf = PCmat @ y                                                         #(ndat,)
    yrecon = USmat @ ybuf                                                    #(nrecon,)
    
    PEmat1 = USmat @ PCmat                                                   #(nrecon, ndat)
    PEmat2 = PEmat1 @ USmat.T                                                #(nrecon, nrecon)

    PSmat = get_covar_Pmat_recon(xcont_recon, sigma, tau, alpha)                #(nrecon, nrecon)
    PQmat = PSmat - PEmat2                                                   #(nrecon, nrecon)
    yerr_recon = np.sqrt(np.diag(PQmat))                                     #(nrecon,)

    PQmat = chol_decomp_L(PQmat)                                             #(nrecon, nrecon)
    yu = PQmat @ yfit2                                                        #(nrecon,)
    
    yuq = Larr_recon @ yq                                                    #(nrecon,)
    yrecon += yu + yuq                                                       #(nrecon,)
    
    return yrecon, yerr_recon



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
    
    
def line_gaussian_smooth_2d(psi_v, xline_recon, line2D_recon, 
                            model_params,
                            flag_inst_res, inst_res, inst_res_err,
                            nblr_model, nnlr):
    

    line2D_recon_smooth = np.zeros_like(line2D_recon)
    
    if flag_inst_res <= 1:
        sigv = inst_res + model_params[nblr_model + nnlr] * inst_res_err
        sigv = max(0., sigv)
        
        for i in range(len(xline_recon)):
            line2D_recon_smooth[i] = fftconvolve(line2D_recon[i], gkern(len(psi_v), sigv), mode='same')
        
    else:
        for i in range(len(xline_recon)):
            sigv = inst_res[i] + model_params[nblr_model + nnlr + i] * inst_res_err[i]
            sigv = max(0., sigv)
            
            line2D_recon_smooth[i] = fftconvolve(line2D_recon[i], gkern(len(psi_v), sigv), mode='same')         

    
    return line2D_recon_smooth
    


def calculate_line2d_from_model(model_params, psi_v, psi_tau, psi2D,
                                xline_recon,
                                xcont_recon, ycont_rm,
                                flag_inst_res, inst_res, inst_res_err, 
                                nblrmodel, nnlr):
    
    #Velocity of tfunc and line2d must be the same

    
    spl = splrep(xcont_recon, ycont_rm)
    
    line2D_recon = np.zeros( ( len(xline_recon), len(psi_v) ) )
    dt = psi_tau[1] - psi_tau[0]
    
    nl = len(xline_recon)
    nv = len(psi_v)
    for j in range(nl):
        tl = xline_recon[j]
        
        for k in range(len(psi_tau)):
            tau = psi_tau[k]
            tc = tl - tau            
            fcon_rm = interp_cont_rm(tc, spl, xcont_recon, ycont_rm)

            for i in range(nv):
                line2D_recon[j,i] += psi2D[k,i] * fcon_rm


    line2D_recon *= dt
    
    
    #Smooth line profile
    line2D_recon_smooth = line_gaussian_smooth_2d(psi_v, xline_recon, line2D_recon, 
                                                  model_params,
                                                  flag_inst_res, inst_res, inst_res_err,
                                                  nblrmodel, nnlr)
    
    return line2D_recon_smooth
