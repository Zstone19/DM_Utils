import numpy as np
import astropy.constants as const

from pypetal.weighting.utils import get_weights, get_bounds

import matplotlib.pyplot as plt

##############################################################################################################
##############################################################################################################
#Utility functions

def get_rms_spectrum(res):
    
    flux = res.bp.data['line2d_data']['profile'][:,:,1]
    mean_prof = np.mean(flux, axis=0)

    rms_prof = np.zeros_like(mean_prof)
    for i in range(flux.shape[0]):
        rms_prof += (flux[i] - mean_prof)**2
        
    rms_prof = np.sqrt(rms_prof/flux.shape[0])

    return rms_prof   



def get_lag_dists(res, wl_bins):
    
    c = const.c.cgs.value

    wl = res.bp.data['line2d_data']['profile'][0,:,0]/(1+res.z)
    flux = res.bp.data['line2d_data']['profile'][:,:,1]
    vel_vals = (c/1e5)*( wl - res.central_wl )/res.central_wl
    
    #--------------------------------------------------
    
    psi2d = res.bp.results['tran2d_rec']
    tau_vals = res.bp.results['tau_rec']

    wl_bin_centers = []
    for i in range(len(wl_bins)-1):
        wl_bin_centers.append( (wl_bins[i] + wl_bins[i+1])/2.0 )

    nbins = len(wl_bin_centers)
    lag_post = np.zeros( (nbins, psi2d.shape[0]) )
    downsampled_posterior = []

    xc, yc, _ = res.bp.data['con_data'].T
    xl = res.bp.data['line2d_data']['time']
    yl = np.zeros( (nbins, len(xl) ) )

    for i in range(nbins):
        ind_l = np.argwhere( wl > wl_bins[i] )[0][0]
        ind_r = np.argwhere( wl <= wl_bins[i+1] )[-1][0]
        
        new_psi2d = psi2d[:,:,ind_l:ind_r]
        new_vel = vel_vals[ind_l:ind_r]
        dV = new_vel[1] - new_vel[0]
        
        #Get line lc
        line_lc = np.sum(flux[:,ind_l:ind_r], axis=1)*dV
        yl[i] = line_lc*res.central_wl*res.bp.VelUnit/(c/1e5)

        #Get lag posterior
        sum2_arr = new_psi2d.sum(axis=2).sum(axis=1)
        sum1_arr = np.sum( new_psi2d.sum(axis=2)*tau_vals, axis=1)
        lag_post[i] = sum1_arr / sum2_arr
        
        #Get weights
        wtau, lags, _, _, _ = get_weights(xc-xc[0], yc, xl-xc[0], yl, k=2)
        
        #Get downsampled dist
        min_bound, _, max_bound, _, _ = get_bounds(lag_post[i], wtau, lags, width=15, rel_height=.99)
        downsampled_posterior.append( lag_post[i][(lag_post[i] > min_bound) & (lag_post[i] < max_bound)]  )
    
    return downsampled_posterior


##############################################################################################################
##############################################################################################################
#Plot

def plot_lag_spectrum(res_arr, wl_bins, line_names=None, labels=None, 
                      xlim=None, ylim=None,
                      show=False, output_fname=None):

    c = const.c.cgs.value
    assert len(res_arr) == len(wl_bins)
    
    
    if labels is None:
        labels = ['']*len(res_arr)
    if line_names is None:
        line_names = ['']*len(res_arr)
    if xlim is None:
        xlim = [None]*len(res_arr)
    if ylim is None:
        ylim = [None]*len(res_arr)
    

    xerr_lo = []
    xerr_hi = []
    yerr_lo = []
    yerr_hi = []
    yvals = []

    for i in range(len(res_arr)):
        nbin = len(wl_bins[i]) - 1
        
        xerr_lo_i = np.zeros(nbin)
        xerr_hi_i = np.zeros(nbin)
        yerr_lo_i = np.zeros(nbin)
        yerr_hi_i = np.zeros(nbin)
        yvals_i = np.zeros(nbin)
        downsampled_posterior = get_lag_dists(res_arr[i], wl_bins[i])
        
        wl_bin_centers = []
        for j in range(len(wl_bins[i])-1):
            wl_bin_centers.append( (wl_bins[i][j] + wl_bins[i][j+1])/2.0 )
        
        for j in range(nbin):
            xerr_lo_i[j] = wl_bin_centers[j] - wl_bins[i][j]
            xerr_hi_i[j] = wl_bins[i][j+1] - wl_bin_centers[j]
            
            yerr_lo_i[j] = np.percentile(downsampled_posterior[j], 16)
            yerr_hi_i[j] = np.percentile(downsampled_posterior[j], 84)
            yvals_i[j] = np.median(downsampled_posterior[j])
            
        xerr_lo.append(xerr_lo_i)
        xerr_hi.append(xerr_hi_i)
        yerr_lo.append(yerr_lo_i)
        yerr_hi.append(yerr_hi_i)
        yvals.append(yvals_i)
        

            
           
    med_vals_tot = np.zeros(len(res_arr))
    lo_vals_tot = np.zeros(len(res_arr))
    hi_vals_tot = np.zeros(len(res_arr))
    for i in range(len(res_arr)):
        downsampled_posterior_tot = get_lag_dists(res_arr[i], [0, np.inf])[0]
        med_vals_tot[i] =  np.median(downsampled_posterior_tot)
        lo_vals_tot[i] =  np.percentile(downsampled_posterior_tot, 16)
        hi_vals_tot[i] =  np.percentile(downsampled_posterior_tot, 84)
            
            
    Nrow = 2
    Ncol = len(res_arr)
    fig, ax = plt.subplots(Nrow, Ncol, figsize=(Ncol*5 + (Ncol-1)*1, Nrow*4.5), sharex='col', sharey='row')
    
    if Ncol == 1:
        ax = np.array([ax]).T
    
    for i in range(len(res_arr)):
        
        wl_bin_centers = []
        for j in range(len(wl_bins[i])-1):
            wl_bin_centers.append( (wl_bins[i][j] + wl_bins[i][j+1])/2.0 )
            
        
        wl = res_arr[i].bp.data['line2d_data']['profile'][0,:,0]/(1+res_arr[i].z)
        vel_vals = (c/1e5)*( wl - res_arr[i].central_wl )/res_arr[i].central_wl
        rms_prof = get_rms_spectrum(res_arr[i])
        
        
        
        ax[0,i].errorbar(wl_bin_centers, yvals[i], xerr=[xerr_lo[i], xerr_hi[i]], yerr=[yerr_lo[i], yerr_hi[i]], fmt='o',
                        ms=4, mec='DodgerBlue', mfc='c', capsize=3)
        ax[0,i].axhline(med_vals_tot[i], color='g', ls='--', alpha=.5)
        ax[0,i].axhspan(lo_vals_tot[i], hi_vals_tot[i], color='lime', alpha=0.1)
        
        if ylim[i] is not None:
            ax[0,i].set_ylim(ylim[i])
        
        
        
        ax[1,i].plot(wl, rms_prof, c='k')

        ax[1,i].set_xlabel(r'Rest Wavelength [$\rm \AA$]', fontsize=14)
        ax[0,i].text(.05, .95, labels[i], transform=ax[0,i].transAxes, fontsize=14, va='top')

        if i == 0:        
            ax[0,i].set_ylabel(r'$\tau$ [d]', fontsize=14)
            ax[1,i].set_ylabel(r'$\rm f_{\lambda, ' + line_names[i] + r', rms}$', fontsize=14)
        
        
        ax2 = ax[0,i].twiny()
        ax2.plot(vel_vals/1e3, rms_prof, c='none')
        ax2.set_xlabel(r'Velocity [$\rm 10^3 \; km \; s^{-1}$]', fontsize=14, labelpad=10)
        
        if xlim[i] is None:
            xlim[i] = ax[0,i].get_xlim()
    
        ax2.set_xlim( (c/1e8)*(xlim[i][0]-res_arr[i].central_wl)/res_arr[i].central_wl, 
                      (c/1e8)*(xlim[i][1]-res_arr[i].central_wl)/res_arr[i].central_wl )
        
    
        for a in [ax[0,i], ax[1,i]]:
            a.axvline(res_arr[i].central_wl, color='purple', linestyle='--', alpha=.5)
            a.tick_params('both', labelsize=10)
            a.tick_params('both', which='major', length=6)
            a.tick_params('both', which='minor', length=3)
            
            if xlim is not None:
                a.set_xlim(xlim[i])
            
        ax[0,i].tick_params('x', labelsize=0)
    
        ax2.tick_params('both', labelsize=10)
        ax2.tick_params('both', which='major', length=6)
        ax2.tick_params('both', which='minor', length=3)
        
        
    plt.subplots_adjust(hspace=0.0, wspace=.05)
    
    if output_fname is not None:
        plt.savefig(output_fname, bbox_inches='tight')
        
    if show:
        plt.show()
        
    plt.close()
    plt.cla()
    plt.clf()
    
    return
