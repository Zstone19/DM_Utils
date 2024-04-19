import numpy as np
import scipy.linalg as sla
from scipy.interpolate import splev, splrep
from numba import njit

###################################################################################################
# DRW MATH

@njit(fastmath=True)
def compute_semiseparable_drw(xcont, a1, c1, sigma, syserr):
    
    phi = np.zeros(len(xcont))
    for i in range(len(xcont)):
        phi[i] = np.exp( -c1 * (xcont[i] - xcont[i-1]) )
        

    S = 0.
    A = sigma[0]*sigma[0] + syserr*syserr + a1
    
    D = np.zeros(len(xcont))
    W = np.zeros(len(xcont))
    
    D[0] = A
    W[0] = 1./D[0]
    for i in range(len(xcont)):
        S = phi[i]*phi[i] * ( S + D[i-1]*W[i-1]*W[i-1] )
        A = sigma[i]*sigma[i] + syserr*syserr + a1
        D[i] = A - a1*a1*S
        W[i] = 1./D[i] * (1. - a1*S)    
    
    return W, D, phi


@njit(fastmath=True)
def multiply_mat_semiseparable_drw(Larr, W, D, phi, a1):
    Z = np.zeros_like(Larr)    
    n = Larr.shape[0]
    m = Larr.shape[1]
    
    for j in range(m):
        f = 0.
        Z[0,j] = Larr[0,j]
        
        for i in range(1,n):
            f = phi[i] * (f + W[i-1] * Z[i-1,j])


    for j in range(m):
        g = 0.
        Z[n-1,j] = Z[n-1,j]/D[n-1]
        
        for i in range(n-2,-1,-1):
            g = phi[i+1] * (g + a1*Z[i+1,j])
            Z[i,j] = Z[i,j]/D[i] - W[i]*g


    return Z


@njit(fastmath=True)
def multiply_matvec_semiseparable_drw(y, W, D, phi, a1):
    z = np.zeros_like(y)
    n = len(z)
    
    f = 0.
    z[0] = y[0]
    for i in range(n):
        f = phi[i] * (f + W[i-1] * z[i-1])
        z[i] = y[i] - a1*f
        
    g = 0.
    z[n-1] = z[n-1]/D[n-1]
    for i in range(n-2,-1,-1):
        g = phi[i+1] * (g + a1*z[i+1])
        z[i] = z[i]/D[i] - W[i]*g
        
    return z


@njit(fastmath=True)
def multiply_mat_transposeB_semiseparable_drw(Y, W, D, phi, a1):
    m = Y.shape[0]
    n = Y.shape[1]
    
    Y_flat = np.hstack(Y)
    Z_flat = np.zeros_like(Y_flat)
    
    for j in range(m):
        f = 0.
        Z_flat[0*m + j] = Y_flat[0 + j*n]
        
        for i in range(1,n):
            f = phi[i] * (f + W[i-1] * Z_flat[(i-1)*m + j])
            Z_flat[i*m + j] = Y_flat[i + j*n] - a1*f
    
    
    for j in range(m):
        g = 0.
        Z_flat[(n-1)*m + j] = Z_flat[(n-1)*m + j]/D[n-1]
        
        for i in range(n-2,-1,-1):
            g = phi[i+1] * (g + a1*Z_flat[(i+1)*m + j])
            Z_flat[i*m + j] = Z_flat[i*m + j]/D[i] - W[i]*g
       
 
    Z = Z_flat.reshape((n,m))

    return Z


###################################################################################################
# UTILITY

@njit(fastmath=True)
def get_Smat_both(sigma, tau, alpha, xcont, xcont_recon):
    Smat = np.zeros((len(xcont_recon), len(xcont)))
    
    
    for i in range(len(xcont_recon)):
        t1 = xcont_recon[i]
        
        for j in range(len(xcont)):
            t2 = xcont[j]
            
            Smat[i,j] = sigma*sigma*np.exp(- (np.abs(t1-t2)/tau)**alpha )
    
    return Smat


@njit(fastmath=True)
def get_Smat_recon(sigma, tau, alpha, xcont_recon):
    Smat = np.zeros((len(xcont_recon), len(xcont_recon)))
    
    for i in range(len(xcont_recon)):
        t1 = xcont_recon[i]
        
        for j in range(i):
            t2 = xcont_recon[j]
            Smat[i,j] = sigma*sigma * np.exp( - (np.abs(t1-t2)/tau)**alpha )
            Smat[j,i] = Smat[i,j]
    
    return Smat


@njit(fastmath=True)
def get_Larr(xcon, nq):
    Larr = np.zeros((len(xcon), nq))
    for i in range(len(xcon)):
        for j in range(nq):
            Larr[i,j] = xcon[i]**j    
    return Larr


@njit(fastmath=True)
def get_covar_Umat(sigma, tau, alpha, xcont_recon, xcont):
    nrecon_cont = len(xcont_recon)
    Umat = np.zeros((nrecon_cont, nrecon_cont))
    
    for i in range(nrecon_cont):
        t1 = xcont_recon[i]
        
        for j in range(len(xcont)):
            t2 = xcont[j]
            
            Umat[i,j] = sigma*sigma*np.exp(- (np.abs(t1-t2)/tau)**alpha )

    return Umat


@njit(fastmath=True)
def get_covar_Pmat(sigma, tau, alpha, xcont_recon):
    nrecon_cont = len(xcont_recon)
    PSmat = np.zeros((nrecon_cont, nrecon_cont))

    for i in range(nrecon_cont):
        t1 = xcont_recon[i]
        
        for j in range(i):
            t2 = xcont_recon[j]
            PSmat[i,j] = sigma*sigma * np.exp( - (np.abs(t1-t2)/tau)**alpha )
            PSmat[j,i] = PSmat[i,j]
    
    return PSmat



###################################################################################################
# MAIN FUNCTIONS

def calculate_cont_from_model_semiseparable(model_params, xcont, ycont, yerr_cont, 
                                            xcont_recon, nq, nvar):
    
    
    
    #Original
    #Lbuf = inv(C) L
    #inv(Cq) = L^T Lbuf
    #yq = Lbuf^T ycont     (yq=uq)
    
    #Cq = inv(inv(Cq))     
    #ybuf = Cq yq          (ybuf)
    
    #Cq = decomp(Cq)
    #Cq = Cq [sigmad]
    #yq += ybuf            (yq=q)
    
    #y = ycont - L yq
    
    #PEmat1 = inv(C) S_both^T
    #yrecon =  PEmat1^T y           (yrecon=shat)
    
    #PEmat2 = S_both PEmat1
    
    #PQmat = S_recon - PEmat2          (PQmat=Q)
    #yerr_recon = sqrt(diag(PQmat))
    
    #PQmat = decomp(PQmat)
    #y = PQmat [timseries_fit]

    #ybuf = L_recon yq
    #yrecon += y + ybuf
    
    
        
    
    nrecon_cont = len(xcont_recon)
    cont_err_mean = np.mean(yerr_cont)
    sys_err = ( np.exp(model_params[0]) - 1. )*cont_err_mean
    tau = np.exp(model_params[2])
    sigma = np.exp(model_params[1]) * np.sqrt(tau)
    
    #Get model params
    yfit = np.atleast_1d(model_params[3:3+nq])   #[sigmad]
    yfit2 = model_params[nvar:nvar+nrecon_cont] #timeseries...

    #Get all matrices
    Larr = get_Larr(xcont, nq)
    Larr_recon = get_Larr(xcont_recon, nq)
    S_both = get_Smat_both(sigma, tau, 1., xcont, xcont_recon)
    S_recon = get_Smat_recon(sigma, tau, 1., xcont_recon)
    W, D, phi = compute_semiseparable_drw(xcont, sigma*sigma, 1./tau, yerr_cont, sys_err)     #W, D, phi -> C
    
    
    
    
    Lbuf = multiply_mat_semiseparable_drw(Larr, W, D, phi, sigma*sigma)
    assert Lbuf.shape == (len(xcont), nq)
    Cqinv = Larr.T @ Lbuf
    Cq = np.linalg.inv(Cqinv)
    yq = Larr.T @ ycont

    ybuf = Cq @ yq

    Cq = np.linalg.cholesky(Cq)
    Cq = Cq @ yfit
    yq += ybuf
        
    y = ycont - Larr @ yq
    
    PEmat1 = multiply_mat_transposeB_semiseparable_drw(S_both, W, D, phi, sigma*sigma)
    assert PEmat1.shape == ( len(xcont), nrecon_cont )
    yrecon = PEmat1.T @ y
    
    PEmat2 = S_both @ PEmat1
    
    PQmat = S_recon - PEmat2
    yerr_recon = np.sqrt(np.diag(PQmat))
    
    PQmat = np.linalg.cholesky(PQmat)
    y = PQmat @ yfit2
    
    ybuf = Larr_recon @ yq
    yrecon += y + ybuf
    
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
        spl = splrep(xcont_recon, ycont_rm)
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
            line2D_recon_smooth[i] = np.convolve(line2D_recon[i], gkern(len(psi_v), sigv), mode='same')
        
    else:
        for i in range(len(xline_recon)):
            sigv = inst_res[i] + model_params[nblr_model + nnlr + i] * inst_res_err[i]
            sigv = max(0., sigv)
            
            line2D_recon_smooth[i] = np.convolve(line2D_recon[i], gkern(len(psi_v), sigv), mode='same')         

    
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
