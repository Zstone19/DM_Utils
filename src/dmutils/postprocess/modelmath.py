import numpy as np
import scipy.linalg as sla


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
        Z[0,j] = Y[0,n]
    
    
    #NEED TO FINISH!!!!!!!!!!!!!!!!!!!!!!
    
    return


###################################################################################################
# UTILITY

def get_Larr(xcon, nq):
    Larr = np.zeros((len(xcon), nq))
    for i in range(len(xcon)):
        for j in range(nq):
            Larr[i,j] = xcon[i]**j    
    return Larr

def get_covar_Umat(sigma, tau, alpha, nrecon_cont, xcont):
    Umat = np.zeros((nrecon_cont, nrecon_cont))
    
    for i in range(nrecon_cont):
        t1 = xcont[i]
        
        for j in range(len(xcont)):
            t2 = xcont[j]
            
            Umat[i,j] = sigma*sigma*np.exp(- (np.abs(t1-t2)/tau)**alpha )

    return Umat



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
                                            nrecon_cont, nq):
    cont_err_mean = np.mean(yerr_cont)
    
    sys_err = ( np.exp(model_params[0]) - 1. )*cont_err_mean
    tau = np.exp(model_params[2])
    sigma = np.exp(model_params[1]) * np.sqrt(tau)
    
    sigma2 = sigma*sigma
    alpha = 1.
    
    
    USmat = get_covar_Umat(sigma, tau, alpha, nrecon_cont, xcont)
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
    
    PEmat1 = multiply_mat_transposeB_semiseparable_drw(USmat, W, D, phi, len(xcont), nrecon_cont, sigma2, PEmat1)