from re import X
import numpy as np
import scipy.linalg as sla
from scipy.interpolate import splev, splrep


###################################################################################################
# MATH

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

def multiply_mat_MN_transposeA(A, B):
    return np.dot(A, B.T)

def multiply_matvec_MN_transposeA(A, x):
    return np.dot(A.T, x)

def inverse_pomat(A):
    A1 = sla.lapack.dpotrf(A, lower=0)
    A2 = sla.lapack.dpotri(A1, lower=0)
    
    Aout = A2.copy()
    
    for i in range(A.shape[0]):
        for j in range(i):
            Aout[i,j] = A2[j,i]
            
    return Aout

def multiply_mat_MN(A, B):
    return np.dot(A, B)
    
def Chol_decomp_L(A):            
    return np.linalg.cholesky(A)

def multiply_matvec(A, x):
    return np.dot(A, x)

def multiply_matvec_MN(A, x):
    return np.dot(A, x)

def multiply_mat_transposeB_semiseparable_drw(Y, W, D, phi, a1):
    Z = np.zeros_like(Y)
    n = Y.shape[0]
    m = Y.shape[1]
    
    for j in range(m):
        f = 0.
        Z[0,j] = Y[n,0]
        
        for i in range(1,n):
            f = phi[i] * (f + W[i-1] * Z[i-1,j])
            Z[i,j] = Y[j,i] - a1*f
    
    
    for j in range(m):
        g = 0.
        Z[n-1,j] = Z[n-1,j]/D[n-1]
        
        for i in range(n-2,-1,-1):
            g = phi[i+1] * (g + a1*Z[i+1,j])
            Z[i,j] = Z[i,j]/D[i] - W[i]*g
        
    return Z


###################################################################################################
# UTILITY

def get_Larr(xcon, nq):
    Larr = np.zeros((len(xcon), nq))
    for i in range(len(xcon)):
        for j in range(nq):
            Larr[i,j] = xcon[i]**j    
    return Larr


def get_covar_Umat(sigma, tau, alpha, xcont_recon, xcont):
    nrecon_cont = len(xcont_recon)
    Umat = np.zeros((nrecon_cont, nrecon_cont))
    
    for i in range(nrecon_cont):
        t1 = xcont_recon[i]
        
        for j in range(len(xcont)):
            t2 = xcont[j]
            
            Umat[i,j] = sigma*sigma*np.exp(- (np.abs(t1-t2)/tau)**alpha )

    return Umat

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



###################################################################################################
# MAIN FUNCTIONS

def calculate_cont_from_model_semiseparable(model_params, xcont, ycont, yerr_cont, 
                                            xcont_recon, nq, nvar):
    
    cont_err_mean = np.mean(yerr_cont)
    nrecon_cont = len(xcont_recon)
    
    sys_err = ( np.exp(model_params[0]) - 1. )*cont_err_mean
    tau = np.exp(model_params[2])
    sigma = np.exp(model_params[1]) * np.sqrt(tau)
    
    sigma2 = sigma*sigma
    alpha = 1.
    
    
    USmat = get_covar_Umat(sigma, tau, alpha, xcont_recon, xcont)
    W, D, phi = compute_semiseparable_drw(xcont, sigma2, 1./tau, yerr_cont, sys_err)
    
    Larr = get_Larr(xcont, nq)
    Lbuf = multiply_mat_semiseparable_drw(Larr, W, D, phi, sigma2)
    Cq = multiply_mat_MN_transposeA(Larr, Lbuf)
    
    yq = multiply_matvec_MN_transposeA(Lbuf, ycont)
    
    Cq = inverse_pomat(Cq)
    ybuf = multiply_mat_MN(Cq, yq)
    
    Cq = Chol_decomp_L(Cq)
    yq = multiply_matvec(Cq, model_params[3])
    
    yq += ybuf
    
    ybuf = multiply_matvec_MN(Larr, yq)
    y = ycont - ybuf
    
    PEmat1 = multiply_mat_transposeB_semiseparable_drw(USmat, W, D, phi, sigma2)
    
    y_cont_recon = multiply_mat_MN_transposeA(USmat, y)
    
    PEmat2 = multiply_mat_MN(USmat, PEmat1)
    
    PSmat = get_covar_Pmat(sigma, tau, alpha, xcont_recon)
    
    PQmat = PSmat - PEmat2
    yerr_cont_recon = np.sqrt(np.diag(PQmat))
    
    PQmat = inverse_pomat(PQmat)
    y = multiply_matvec(PQmat, model_params[nvar])
    
    Larr_recon = get_Larr(xcont_recon, nq)
    ybuf = multiply_matvec_MN(Larr_recon, yq)
    
    y_cont_recon += y + ybuf
    
    return y_cont_recon, yerr_cont_recon





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
