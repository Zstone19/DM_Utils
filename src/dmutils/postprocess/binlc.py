import numpy as np
import astropy.constants as const

from dmutils.postprocess.lags import get_lag_dists
from dmutils.postprocess.result import val2latex


import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
import palettable


############################################################################################################
############################################################################################################
############################################################################################################


def get_psi1d(res, wl_bins):
    
    """Generates the 1D transfer function from a Result object for each of the wavelength bins given.

    Parameters:
        res (Result): Result object from BRAINS
        wl_bins (list): List of wavelength bins to use in the analysis

    Returns:
        tau_vals (array): Array of tau values
        psi1d (array): Array of psi values
    """
    
    nbin = len(wl_bins)-1
    
    #--- Extract data
    wl = res.bp.data['line2d_data']['profile'][0,:,0]/(1+res.z)
    psi2d = res.bp.results['tran2d_rec']
    tau_vals = res.bp.results['tau_rec']
    
    #--- Get 1D psi samples
    psi1d_samples = np.zeros((psi2d.shape[0], psi2d.shape[1], nbin))    
        
    for i in range(nbin):
        ind1 = np.argwhere(wl > wl_bins[i]).T[0][0]
        ind2 = np.argwhere(wl <= wl_bins[i+1]).T[0][-1]            
        psi1d_samples[:,:,i] = np.sum(psi2d[:,:,ind1:ind2], axis=2)
        
    #--- Get psi1d
    psi1d = np.zeros((3, psi2d.shape[1], nbin))
    for i in range(nbin):
        psi1d[:,:,i] = np.percentile(psi1d_samples[:,:,i], [16, 50, 84], axis=0)    

    return tau_vals[0], psi1d



def get_binned_lcs(res, wl_bins):
    
    """Generates the binned light curves from a Result object for each of the wavelength bins given.

    Parameters:
        res (Result): Result object from BRAINS
        wl_bins (list): List of wavelength bins to use in the analysis

    Returns:
        xl (array): Array of time values
        yl (array): Array of line light curve values
        yerrl (array): Array of line light curve error values
        yrec (array): Array of reconstructed line light curve values
    """
    
    nbin = len(wl_bins)-1
    c = const.c.cgs.value
    npos = res.bp.results['tran2d_rec'].shape[0]   
    
    line_err_idx = np.nonzero(res.bp.para_names['name'] == 'sys_err_line')[0][0]      

    prof = res.bp.data['line2d_data']['profile'][:,:,1]
    prof_err = res.bp.data['line2d_data']['profile'][:,:,2]
    wl_vals = res.bp.data['line2d_data']['profile'][0,:,0].copy()/(1+res.z)
    vel_vals = (c/1e5)*( wl_vals - res.central_wl )/res.central_wl #km/s
    dV = (vel_vals[1] - vel_vals[0])/res.bp.VelUnit #km/s
    
    
    xl = res.bp.data['line2d_data']['time'] 
    yl = np.zeros( (len(xl), nbin) )
    yerrl = np.zeros( (len(xl), nbin) )
    yrec = np.zeros( (npos, len(xl), nbin) )

    for i in range(nbin):    
        ind1 = np.argwhere(wl_vals > wl_bins[i]).T[0][0]
        ind2 = np.argwhere(wl_vals <= wl_bins[i+1]).T[0][-1]
        
    
        #--- Get input line light curve
        line_lc = np.sum(prof[:, ind1:ind2], axis=1)*dV
        line_lc_err = np.sqrt( np.sum(prof_err[:, ind1:ind2]**2, axis=1) )*dV
        
        line_mean_err = np.mean(line_lc_err)
        syserr_line = (np.exp(np.median(res.bp.results['sample'][:, line_err_idx])) - 1.0) * line_mean_err

        yl[:,i] = line_lc*res.central_wl*res.bp.VelUnit/(c/1e5)
        yerrl[:,i] = np.sqrt(line_lc_err**2 + (syserr_line**2)*(dV**2))
        
        
        #--- Get reconstructed line light curve
        yrec[:,:,i] = np.sum(res.bp.results['line2d_rec'][:,:,ind1:ind2], axis=2)*dV * res.central_wl*res.bp.VelUnit/(c/1e5)
        
    return xl, yl, yerrl, yrec


def get_cont_lc(res):
    
    """Generates the continuum light curve from a Result object.

    Parameters:
        res (Result): Result object from BRAINS

    Returns:
        xc (array): Array of time values
        yc (array): Array of continuum light curve values
        yerrc (array): Array of continuum light curve error values
        xrec (array): Array of reconstructed continuum time values
        yrec (array): Array of reconstructed continuum light curve values
    """
    
    con_err_idx = np.nonzero(res.bp.para_names['name'] == 'sys_err_con')[0][0]
    
    xc, yc, yerr_in = res.bp.data['con_data'].T
    xrec = res.bp.results['con_rec'][0,:,0]
    yrec = res.bp.results['con_rec'][:,:,1]
        
    con_mean_err = np.mean(yerr_in)
    syserr_con = (np.exp(np.median(res.bp.results['sample'][:, con_err_idx])) - 1.0) * con_mean_err
    
    yerrc = np.sqrt(yerr_in**2 + (syserr_con**2))
    
    return xc, yc, yerrc, xrec, yrec

############################################################################################################
############################################################################################################
############################################################################################################

def plot_binned_lcs(res, wl_bins,
                    psi_xlim=None, lag_nbin=25,
                    show=True, output_fname=None):
    
    
    ### Plotting style ###
    cmap = ListedColormap( palettable.cartocolors.qualitative.Vivid_10.mpl_colors )
    colors = cmap.colors
    
    #### Get data ####
    tau, psi1d = get_psi1d(res, wl_bins)
    xl, yl, yerrl, yrecl = get_binned_lcs(res, wl_bins)
    xc, yc, yerrc, xrecc, yrecc = get_cont_lc(res)
    downsampled_posteriors = get_lag_dists(res, wl_bins)
    
    
    #### Plot ####
    Nrow = len(wl_bins)-1
    Ncol = 3

    fig = plt.figure(figsize=(4*Ncol, 1.5*Nrow))
    gs = gridspec.GridSpec(Nrow+1, Ncol, figure=fig, hspace=0, wspace=.05, width_ratios=[1,4,1])

    for i in range(Nrow):
        
        if i == 0:
            ax1 = fig.add_subplot(gs[i,0])
            ax2 = fig.add_subplot(gs[i,1])
            ax3 = fig.add_subplot(gs[i,2])
            
        else:
            ax1 = fig.add_subplot(gs[i,0], sharex=ax1)
            ax2 = fig.add_subplot(gs[i,1], sharex=ax2)
            ax3 = fig.add_subplot(gs[i,2], sharex=ax3, sharey=ax3)

        
        
        #--- Plot lag dist
        vals, bins = np.histogram(downsampled_posteriors[i], bins=lag_nbin, range=[0,150], density=False)
        vals = vals/np.max(vals) * .7
        
        bin_centers = []
        for j in range(len(bins)-1):
            bin_centers.append( (bins[j]+bins[j+1])/2. )
        
        ax3.bar(bin_centers, vals, color=colors[i%9], width=bins[1]-bins[0], fill=True, alpha=.7)
        ax3.axvline(np.median(downsampled_posteriors[i]), color='k', ls='--', lw=1.5)
        

        
        #--- Plot Psi
        scale = psi1d[2,:,i].max()*1.5
        ax1.plot(tau, psi1d[1,:,i]/scale, color='k')
        ax1.fill_between(tau, psi1d[0,:,i]/scale, psi1d[2,:,i]/scale, color=colors[i%9], alpha=0.4)
        
        #--- Plot LC
        ax2.errorbar(xl-xl[0], yl[:,i], yerr=yerrl[:,i], fmt='o', ms=4, color=colors[9], mec='k', mew=.5)
        for j in range(yrecl.shape[0]):
            ax2.plot(xl-xl[0], yrecl[j,:,i], color=colors[i%9], alpha=0.05, lw=.5)
        
        ax1.set_xlim(psi_xlim)
        ax3.set_xlim(0, 150)
        ax3.set_ylim(0, 1) 
        for a in [ax1, ax2, ax3]:
            a.set_yticklabels([])
            a.tick_params('both', which='major', length=5)
            a.tick_params('both', which='minor', length=2)
            
            
        ax1.tick_params('y', which='both', length=0)
        ax2.tick_params('y', which='both', length=0)
        ax3.tick_params('y', which='both', length=0)
            
        
        ax2.tick_params('x', labelsize=0)
        ax3.tick_params('x', labelsize=0)
        if i != Nrow - 1:
            ax1.tick_params('x', labelsize=0)
            ax3.tick_params('x', labelsize=0)
        else:
            ax1.tick_params('x', labelsize=10)
            ax3.tick_params('x', labelsize=10)
        
        
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(3))
        
        ax3.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax3.yaxis.set_major_locator(ticker.MaxNLocator(3))    
        
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(10))
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(3))    
        
        if i == Nrow-1:
            ax1.set_xlabel(r'$\tau$ [d]', fontsize=14) 
            ax3.set_xlabel(r'$\tau$ [d]', fontsize=14)
            
        ymax = np.max([yl[:,i].max(), yrecl[:,:,i].max()])*1.1
        ax2.set_ylim(top=ymax)
            
            
        txt = r'$\rm ' + '{:.0f}'.format(wl_bins[i]) + '-' + '{:.0f}'.format(wl_bins[i+1]) + r'\; \AA $'
        ax2.text(.01, .7, txt, fontsize=11, transform=ax2.transAxes, ha='left', va='bottom') 
        
        txt = val2latex(downsampled_posteriors[i], 0)
        ax3.text(.97, .7, txt, fontsize=10, transform=ax3.transAxes, ha='right', va='bottom')
        
        if i == 0:
            ax1.set_title(r'$\rm \psi(\tau)$', fontsize=14)
            ax2.set_title('Flux', fontsize=14)
            ax3.set_title('Lag', fontsize=14)

            
    #--- Plot continuum
    ax = fig.add_subplot(gs[-1,1], sharex=ax2)

    ax.errorbar(xc-xl[0], yc, yerr=yerrc, fmt='o', ms=4, color=colors[9], mec='k', mew=.5)
    for j in range(yrecc.shape[0]):
        ax.plot(xrecc-xl[0], yrecc[j,:], color=colors[Nrow % 9], alpha=0.05, lw=.5)
        
    xmax = np.max([xl.max(), xrecc.max()])-xl[0]+100
    ax.set_xlim(left=-100, right=xmax)

    ax.tick_params('y', labelsize=0)
    ax.tick_params('both', which='major', length=5)
    ax.tick_params('both', which='minor', length=2)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))

    ymax = np.max([yc.max(), yrecc[:,:].max()])*1.25
    ax.set_ylim(top=ymax)

    ax.text(.01, .7, 'Continuum', fontsize=11, transform=ax.transAxes, ha='left', va='bottom') 
    ax.set_xlabel(r'Time [d]', fontsize=14)
    
    if output_fname is not None:
        plt.savefig(output_fname, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
        
    plt.close()
    plt.cla()
    plt.clf()
    
    return
