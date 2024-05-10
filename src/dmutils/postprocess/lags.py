import numpy as np
import astropy.constants as const
from scipy.stats import binned_statistic

from pypetal.weighting.utils import get_weights, get_bounds
from dmutils.input import read_input_file
from dmutils.postprocess.result import weighted_percentile

import matplotlib.pyplot as plt


##############################################################################################################
##############################################################################################################
#Base functions


def get_rms_spectrum(res):
    
    flux = res.bp.data['line2d_data']['profile'][:,:,1]
    mean_prof = np.mean(flux, axis=0)

    rms_prof = np.zeros_like(mean_prof)
    for i in range(flux.shape[0]):
        rms_prof += (flux[i] - mean_prof)**2
        
    rms_prof = np.sqrt(rms_prof/flux.shape[0])

    return rms_prof  



def get_mean_spec(time, wl, flux, nwl=None, const=0, tbounds=[0, np.inf]):
    wlmin = []
    wlmax = []
    flux_tot = []


    allsame = []
    for i in range(len(time)):
        allsame.append(np.allclose(wl[i], wl[0]))


    if np.all(allsame) & (nwl is None):
        wlmin = wl[0].min()
        wlmax = wl[0].max()
        wl_centers = wl[0].copy()

        good_ind = np.argwhere( (time > tbounds[0]) & (time < tbounds[1]) ).flatten()        
        binned_spec = flux[good_ind] + const
        
        nepoch = len(binned_spec)



    else:
        for i in range(len(time)):
            if (time[i] < tbounds[0]) or (time[i] > tbounds[1]):
                continue
                    
            wlmin.append(wl[i].min())
            wlmax.append(wl[i].max())
            flux_tot.append(flux[i] + const)
            

        nepoch = len(flux_tot)
        wlmin = np.max(wlmin)
        wlmax = np.min(wlmax)        

        
        binned_wl = np.linspace(wlmin, wlmax, nwl+1)
        wl_centers = []
        for i in range(len(binned_wl)-1):
            wl_centers.append((binned_wl[i]+binned_wl[i+1])/2)

        binned_spec = np.zeros((nepoch, len(binned_wl)-1))
        for i in range(nepoch):
            binned_spec[i] = binned_statistic(wl[i], flux_tot[i], bins=binned_wl, statistic=np.nanmedian)[0]
    
        
    
    mean_spec = np.nanmedian(binned_spec, axis=0)
    rms_spec = np.zeros_like(mean_spec)
    for i in range(nepoch):
        for j in range(len(mean_spec)):
            
            if not np.isnan(binned_spec[i][j]):
                rms_spec[j] += (binned_spec[i][j] - mean_spec[j])**2
            
    rms_spec = np.sqrt(rms_spec/nepoch)
    mean_spec -= const
    
    return np.array(wl_centers), mean_spec, rms_spec





def get_bins(res, fbin=None, nbin=None, vstart=-5e3, vend=5e3 ):

    central_wl = res.central_wl
    wl = res.bp.data['line2d_data']['profile'][0,:,0]/(1+res.z)
    rms_spec = get_rms_spectrum(res)

    # spec_in = rms_spec.copy()
    # rms_spec = rms_spec[ rms_spec > 0]

    if nbin is None:
        fix_fbin = True
    else:
        fix_fbin = False

    #v/c = (w-center)/center
    wlstart = vstart/3e5 * central_wl + central_wl
    wlend = vend/3e5 * central_wl + central_wl

    if fix_fbin:

        bins = [wlstart]
        while bins[-1] < wlend:
            fbin_i = 0
            wlbin_i = bins[-1]
            
            
            while fbin_i < fbin:
                wlbin_i += 1
                mask = (wl > bins[-1]) & (wl < wlbin_i)
                fbin_i = np.trapz( x=wl[mask], y=rms_spec[mask] )


            bins.append(wlbin_i)  
            
    else:
        nbin_out = np.inf
        df = 1

        fbin = df
        dw = 1
        
        while nbin_out > nbin:
            bins_left = [central_wl]
            bins_right = [central_wl]
            

            #Middle -> left
            for _ in range(100):
                fbin_i = 0
                wlbin_i = bins_left[-1] - dw
                
                
                for i in range( int(   (bins_left[-1] - wlstart)/dw   ) ):                
                    wlbin_i = bins_left[-1] - (i+1)*dw
                    
                    mask = (wl < bins_left[-1]) & (wl > wlbin_i)
                    fbin_i = np.trapz( x=wl[mask], y=rms_spec[mask] )
                    if fbin_i > fbin:
                        break

                if fbin_i > fbin:
                    bins_left.append(wlbin_i)                 
                if bins_left[-1] < wlstart:
                    break
            



            #Middle -> right
            for _ in range(100):
                fbin_i = 0
                wlbin_i = bins_right[-1] + dw
                
                for i in range( int(   (wlend - bins_right[-1])/dw   ) ):                
                    wlbin_i = bins_right[-1] + (i+1)*dw
                    
                    mask = (wl > bins_right[-1]) & (wl < wlbin_i)
                    fbin_i = np.trapz( x=wl[mask], y=rms_spec[mask] )
                    if fbin_i > fbin:
                        break

                
                if fbin_i > fbin:
                    bins_right.append(wlbin_i)    
                if bins_right[-1] > wlend:
                    break        
        
        

            nbin_out = len(bins_right) + len(bins_left) - 2
            fbin += df

    bins = np.concatenate([ bins_left[::-1], bins_right[1:] ])
       
    return bins




##############################################################################################################
##############################################################################################################
#Utility functions 


def get_lag_dists(res, wl_bins, weights=None):

    if weights is None:
        weights = np.ones( len(res.bp.results['sample_info']) )
        weights /= np.sum(weights)


    
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
    downsampled_weights = []

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
        
        mask = (lag_post[i] > min_bound) & (lag_post[i] < max_bound)
        downsampled_posterior.append( lag_post[i][mask] )
        downsampled_weights.append( weights[mask] )
    

    return downsampled_posterior, downsampled_weights




def get_binned_lcs(input_fname, wl_bins, z):

    time, wl, flux, err = read_input_file(input_fname)
    wl /= 1+z
    
    nbin = len(wl_bins) - 1

    x = time.copy()
    ytot = np.zeros( (nbin, len(x)) )
    yerrtot = np.zeros( (nbin, len(x)) )

    for i in range(len(x)):        
        
        for j in range(nbin):
            mask = (wl[i] >= wl_bins[j]) & (wl[i] < wl_bins[j+1])
            
            ytot[j,i] = np.sum(flux[i][mask])
            yerrtot[j,i] = np.sqrt( np.sum(err[i][mask]**2) )
    

    return x, ytot, yerrtot


##############################################################################################################
##############################################################################################################
#Plot

def plot_lag_spectrum(res_arr, wl_bins, weight_all=None, line_names=None, labels=None, 
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
    if weight_all is None:
        weight_all = [None]*len(res_arr)
    

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
        downsampled_posterior, downsampled_weights = get_lag_dists(res_arr[i], wl_bins[i], weights=weight_all[i])
        
        wl_bin_centers = []
        for j in range(len(wl_bins[i])-1):
            wl_bin_centers.append( (wl_bins[i][j] + wl_bins[i][j+1])/2.0 )
        
        for j in range(nbin):
            xerr_lo_i[j] = wl_bin_centers[j] - wl_bins[i][j]
            xerr_hi_i[j] = wl_bins[i][j+1] - wl_bin_centers[j]
            
            yerr_lo_i[j] = weighted_percentile(downsampled_posterior[j], downsampled_weights[j], .16)
            yerr_hi_i[j] = weighted_percentile(downsampled_posterior[j], downsampled_weights[j], .84)
            yvals_i[j] = weighted_percentile(downsampled_posterior[j], downsampled_weights[j], .84)
            
        xerr_lo.append(xerr_lo_i)
        xerr_hi.append(xerr_hi_i)
        yerr_lo.append(yerr_lo_i)
        yerr_hi.append(yerr_hi_i)
        yvals.append(yvals_i)
        

            
           
    med_vals_tot = np.zeros(len(res_arr))
    lo_vals_tot = np.zeros(len(res_arr))
    hi_vals_tot = np.zeros(len(res_arr))
    for i in range(len(res_arr)):
        downsampled_posterior_tot, downsampled_weights_tot = get_lag_dists(res_arr[i], [0, np.inf], weight_all[i])
        downsampled_posterior_tot = downsampled_posterior_tot[0]
        downsampled_weights_tot = downsampled_weights_tot[0]

        med_vals_tot[i] = weighted_percentile(downsampled_posterior_tot, downsampled_weights_tot, .5)
        lo_vals_tot[i] = weighted_percentile(downsampled_posterior_tot, downsampled_weights_tot, .16)
        hi_vals_tot[i] = weighted_percentile(downsampled_posterior_tot, downsampled_weights_tot, .84)
            
            
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
