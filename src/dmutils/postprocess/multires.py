import numpy as np
import sys

import matplotlib.pyplot as plt
from matplotlib import gridspec

from astropy.table import Table
from astropy.io import ascii
import astropy.constants as const
from dmutils.postprocess.result import weighted_percentile, val2latex, val2latex_weighted
from dmutils.postprocess.lags import plot_lag_spectrum


def isnone(arr):    
    if isinstance(arr, list) or isinstance(arr, np.ndarray):
        mask = np.zeros(len(arr), dtype=bool)
        for i, a in enumerate(arr):
            mask[i] = (a is None)
        
        return mask
    else:    
        return (arr is None)
    
    
def text_format(vals, weights):
    return '{:.2f}'.format(weighted_percentile(vals, weights))
    


def plot_mult(res_arr, weights_all=None, res_names=None, 
              bounds_arr=None, tf_ymax_arr=None, tf_xbounds_arr=None, 
              cloud_cbar_range=None, tf_cbar_range=None,
              skip_clouds=1, plot_rblr=True, ptype='median',
              output_fname=None, show=False):
    
    assert len(res_arr) > 0
    nres = len(res_arr)

    if cloud_cbar_range == 'share':
        max_vel = -np.inf
        min_vel = np.inf
        skip = 10
        
        for res in res_arr:
            cloud_dat = np.loadtxt(res.cloud_fname)
            vx_vals = cloud_dat[:,3]
            vy_vals = cloud_dat[:,4]
            
            max_v = np.max( [np.max(vy_vals[::skip]), np.max(vx_vals[::skip])] ) 
            min_v = np.min( [np.min(vy_vals[::skip]), np.min(vx_vals[::skip])] )

            if max_v > max_vel:
                max_vel = max_v
                
            if min_v < min_vel:
                min_vel = min_v

        cloud_cbar_range = [min_vel, max_vel]
        
        
        
    if tf_cbar_range == 'share':
        max_tf = -np.inf
        min_tf = np.inf
        
        for res in res_arr:
            idx, par = res.bp.find_max_prob()
            plot_arr = np.median( res.bp.results['tran2d_rec'], axis=0 )
            plot_arr[ plot_arr > 1 ] = 0.
            
            max_t = np.max( plot_arr ) 
            min_t = np.min( plot_arr )

            if max_t > max_tf:
                max_tf = max_t
                
            if min_t < min_tf:
                min_tf = min_t

        tf_cbar_range = [min_tf, max_tf]


    if weights_all is None:
        weights_all = [None]*nres
    if res_names is None:
        res_names = [None]*nres
    if bounds_arr is None:
        bounds_arr = [None]*nres
    if tf_ymax_arr is None:
        tf_ymax_arr = [None]*nres
    if tf_xbounds_arr is None:
        tf_xbounds_arr = [None]*nres
    if cloud_cbar_range is None:
        cloud_cbar_range = [[None,None]]*nres
    if tf_cbar_range is None:
        tf_cbar_range = [[None,None]]*nres
        
    if isinstance(tf_ymax_arr, float) or isinstance(tf_ymax_arr, int):
        tf_ymax_arr = [tf_ymax_arr]*nres
     
    arrs = [bounds_arr, tf_xbounds_arr, cloud_cbar_range, tf_cbar_range]
    for i in range(len(arrs)):
        
        if len(arrs[i]) == 2:
            if np.all(isnone(arrs[i])):
                pass
            else:
                if nres != 2:
                    arrs[i] = [arrs[i]]*nres
                else:
                    if isinstance(arrs[i][0], list):
                        pass
                    else:
                        arrs[i] = [arrs[i]]*nres

    if len(tf_ymax_arr) == 2:
        if np.all(isnone(tf_ymax_arr)):
            pass
        else:
            if nres != 2:
                tf_ymax_arr = [tf_ymax_arr]*nres
            else:
                assert isinstance(tf_ymax_arr[0]*1.0, float)
            

        
    
    assert nres == len(res_names) == len(tf_ymax_arr)
    for arr in arrs:
        assert nres == len(arr)
    

    fig = plt.figure(figsize=(20, 5*nres))
    gs_tot = gridspec.GridSpec(nres, 3, figure=fig, hspace=.25, wspace=.2, width_ratios=[1, 1, 1])
    
    for i, res in enumerate(res_arr):
        gs_i = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_tot[i,:2], wspace=.08)
        
        ax1 = fig.add_subplot(gs_i[0])
        ax2 = fig.add_subplot(gs_i[1], sharey=ax1, sharex=ax1)
        ax_clouds = [ax1, ax2]
        ax_tf = fig.add_subplot(gs_tot[i,2])

        ax_clouds = res.plot_clouds(colorbar=True, bounds=arrs[0][i], ax=ax_clouds, 
                                    vmin=arrs[2][i][0], vmax=arrs[2][i][1], plot_rblr=plot_rblr,
                                    posterior_weights=weights_all[i], ptype=ptype,
                                    skip=skip_clouds, show=False)
        
        ax_tf = res.transfer_function_2dplot(weights=weights_all[i], ax=ax_tf, ymax=tf_ymax_arr[i], xbounds=arrs[1][i], 
                                             vmin=arrs[3][i][0], vmax=arrs[3][i][1], ptype=ptype,
                                             show=False)
        
        
        #Set cloud labels
        if i == 0:
            ax_clouds[0].set_title('Side View', fontsize=22)
            ax_clouds[1].set_title('Observer POV', fontsize=22)
        if i == len(res_arr)-1:
            ax_clouds[0].set_xlabel('x [lt-d]', fontsize=20)
            ax_clouds[1].set_xlabel('y [lt-d]', fontsize=20)


        #Set transfer function labels
        if i == 0:
            # ax_tf.set_title(r'Max Likelihood $\rm \Psi(v, t)$', fontsize=22)
            ax_tf.set_title(r'$\rm \Psi(v, \tau)$', fontsize=22)
        if i == len(res_arr)-1:
            ax_tf.set_xlabel(r'Velocity [$\rm 10^3 \; km \ s^{-1} $]', fontsize=20)


        #Set the names of each result (row)
        l, b, w, h = ax_tf.get_position().bounds
        ytxt = b + h/2
        plt.figtext(.94, ytxt, res_names[i], fontsize=30, rotation=270, va='center', ha='center')


    if output_fname is not None:
        plt.savefig(output_fname, bbox_inches='tight', dpi=200)
        
    if show:
        plt.show()
        
    plt.cla()
    plt.clf()
    plt.close()
    
    return




def plot_mult_fitres(res_arr, weights_all=None, res_names=None, 
                     include_res=False, inflate_err=False, temps=None,
                     xbounds_arr=None, output_fname=None, show=False):
    
    
    assert len(res_arr) > 0
    nres = len(res_arr)
    
    if temps is None:
        temps = np.ones(len(res_arr))
    
    if weights_all is None:
        weights_all = [None]*nres
    if res_names is None:
        res_names = [None]*nres
    if xbounds_arr is None:
        xbounds_arr = [None]*nres

    if len(xbounds_arr) == 2:
        if np.all(isnone(xbounds_arr)):
            pass
        else:
            if nres != 2:
                xbounds_arr = [xbounds_arr]*nres
            else:
                if isinstance(xbounds_arr[0], list):
                    pass
                else:
                    xbounds_arr = [xbounds_arr]*nres
                    
                    
    assert nres == len(res_names) == len(xbounds_arr)



    fig = plt.figure(figsize=(25, 5*nres))
    
    if include_res:
        gs_tot = gridspec.GridSpec(nres, 6, figure=fig, hspace=.2, wspace=.2)
    else:
        gs_tot = gridspec.GridSpec(nres, 4, figure=fig, hspace=.2, wspace=.2)
    
    for i, res in enumerate(res_arr):
        
        if include_res:
            gs_l = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_tot[i,:3], wspace=.05)
            ax1 = fig.add_subplot(gs_l[0])
            ax2 = fig.add_subplot(gs_l[1], sharey=ax1, sharex=ax1)
            ax3 = fig.add_subplot(gs_l[2], sharey=ax1, sharex=ax1)
            ax_2d = [ax1, ax2, ax3]
        else:
            gs_l = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_tot[i,:2], wspace=.09)
            ax1 = fig.add_subplot(gs_l[0])
            ax2 = fig.add_subplot(gs_l[1], sharey=ax1, sharex=ax1)
            ax_2d = [ax1, ax2]

        if include_res:
            gs_r = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_tot[i,3:], wspace=.08, hspace=0)
            ax1 = fig.add_subplot(gs_r[0])
            ax2 = fig.add_subplot(gs_r[1], sharex=ax1)
            ax_lc = [ax1, ax2]
        else:
            gs_r = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_tot[i,2:], wspace=.08, hspace=0)
            ax1 = fig.add_subplot(gs_r[0])
            ax2 = fig.add_subplot(gs_r[1], sharex=ax1)
            ax_lc = [ax1, ax2]
        
        
        ax_2d = res.line2d_plot(weights=weights_all[i], include_res=include_res, xbounds=xbounds_arr[i], 
                                ax=ax_2d, show=False)
        ax_lc = res.lc_fits_plot(weights=weights_all[i], temp=temps[i], inflate_err=inflate_err, 
                                 ax=ax_lc, show=False)
        

        #Set labels on the line2d plots
        prof_titles = ['Data', 'Model', 'Residuals']
        ff = 2
        if i == 0:    
            for n in range(len(ax_2d)):
                ax_2d[n].set_title(prof_titles[n], fontsize=16*ff)

        if i == len(res_arr)-1:
            for n in range(len(ax_2d)):
                ax_2d[n].set_xlabel(r'Velocity [$\rm 10^3 \; km \; s^{-1}$]', fontsize=11*ff, labelpad=10)


        #Set labels on the LC plots
        if i == len(res_arr)-1:
            ax_lc[-1].set_xlabel('MJD', fontsize=15*ff)


        #Set the names of each result (row)
        l, b, w, h = ax_lc[0].get_position().bounds
        ytxt = b
        plt.figtext(.97, ytxt, res_names[i], fontsize=30, rotation=270, va='center', ha='center')

    if output_fname is not None:
        plt.savefig(output_fname, bbox_inches='tight', dpi=200)
        
    if show:
        plt.show()
        
    plt.cla()
    plt.clf()
    plt.close()

    
    return




def plot_mult_lag(res_arr_arr, wl_bins_arr, weight_all_arr, line_names, line_labels, res_labels,
                  xlim, ylim, use_error=True, show=False, output_fname=None):

    #Full, low, high
    Nrow = 2
    Ncol = len(res_arr_arr[0])

    fig, ax = plt.subplots(Nrow, Ncol, figsize=(Ncol*5 + (Ncol-1)*1, Nrow*4.5), sharex='col', sharey='row')

    #Binned lags
    color1a = ['k', 'b', 'r']
    color1b = ['gray', 'DodgerBlue', 'pink']
    #Integrated lags
    color2a = ['gray', 'c', 'darkorange']
    color2b = ['gray', 'lightblue', 'orange']
    #RMS flux
    color3 = ['k', 'b', 'r']



    for i in range(len(res_arr_arr)):
        if i == 0:
            labels = line_labels
        else:
            labels = ['']*len(line_labels)

        ax = plot_lag_spectrum( res_arr_arr[i], wl_bins_arr[i], weight_all_arr[i], line_names, 
                                labels, res_labels[i], xlim=xlim, ylim=ylim, 
                                color1a=color1a[i], color1b=color1b[i], color2a=color2a[i],
                                color2b=color2b[i], color3=color3[i],
                                show=False, ax=ax, use_error=use_error, nplot=i, 
                                output_fname=None )


    plt.subplots_adjust(hspace=0.0, wspace=.05)
    plt.figlegend(bbox_to_anchor=(.12,.97), loc='upper left', fontsize=12, ncols=Ncol)

    if output_fname is not None:
        plt.savefig(output_fname, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()
    plt.cla()
    plt.clf()

    return
    
    



def latex_table_mult(res_arr, weights_all=None, print_err=True, res_names=None, output_fname=sys.stdout):

    if print_err:
        format_func = val2latex_weighted
    else:
        format_func = text_format

    assert len(res_arr) > 0
    nres = len(res_arr)

    if res_names is None:
        res_names = ['Res ' + str(i+1) for i in range(nres)]

    names_tot = np.array(['BLR model ln(Rblr)', 'BLR model beta', 'BLR model F',
       'BLR model Inc', 'BLR model Opn', 'BLR model Kappa',
       'BLR model gamma', 'BLR model xi', 'BLR model ln(Mbh)',
       'BLR model fellip', 'BLR model fflow', 'BLR model ln(sigr_circ)',
       'BLR model ln(sigthe_circ)', 'BLR model ln(sigr_rad)',
       'BLR model ln(sigthe_rad)', 'BLR model theta_rot',
       'BLR model ln(sig_turb)', 'line broaden', 'sys_err_line',
       'sys_err_con', 'sigmad', 'taud', 'trend', 'A', 'Ag'])

    logparam_names = np.array(['BLR model ln(Rblr)', 'BLR model ln(sigr_circ)',
                     'BLR model ln(sigthe_circ)', 'BLR model ln(sigr_rad)',
                     'BLR model ln(sigthe_rad)', 'BLR model ln(sig_turb)', 'sigmad', 'taud'])

    latex_names = [r'$\log_{10}(R_{BLR})$', r'$\beta$', r'$F$', r'$i$', r'$\theta_{opn}$',
                r'$\kappa$', r'$\gamma$', r'$\xi$', r'$\log_{10}(M_{BH})$',
                r'$f_{ellip}$', r'$f_{flow}$', r'$\log_{10}(\sigma_{\rho, circ})$', r'$\log_{10}(\sigma_{\theta, circ})$',
                r'$\log_{10}(\sigma_{\rho, rad})$', r'$\log_{10}(\sigma_{\theta, rad})$', r'$\theta_{e}$', r'$\log_{10}(\sigma_{turb})$',
                r'$\Delta V_{line}$', r'$\sigma_{sys, line}$', r'$\sigma_{sys, con}$',
                r'$\log_{10}( \sigma_d )$', r'$\log_{10}( \tau_d )$', r'B', r'$A$', r'$A_g$']

    units = ['lt-day', '', r'$R_{BLR}$', 'deg', 'deg', '', '', '', r'$M_{\odot}$', '', '', '', '', '', '', 'deg',
            r'$v_{circ}$', r'$\rm km \; s^{-1}$', '', '', '', 'd', r'$\rm d^{-1}$', '', '']


    values = np.zeros(( nres, len(names_tot) ), dtype=object)


    for i, res in enumerate(res_arr):
        if weights_all is None:
            weights = np.ones( len(res_arr[i].bp.results['sample_info']) )
            weights /= np.sum(weights)
        else:
            weights = weights_all[i]
        
        
        prof_err = res.bp.data['line2d_data']['profile'][:,:,2]
        
        c = const.c.cgs.value
        wl_vals = res.bp.data['line2d_data']['profile'][0,:,0].copy()
        vel_vals = (c/1e5) * ( wl_vals/(1+res.z) - res.central_wl )/res.central_wl
        dv = np.abs(vel_vals[1] - vel_vals[0])/res.bp.VelUnit
        
        line_lc_err = np.sqrt( np.sum(prof_err**2, axis=1) )*dv
        _, _ , yerr_cont = res.bp.data['con_data'].T



        for j in range(len(names_tot)):
            values[i, j] = '---'

        for j, name in enumerate(res.bp.para_names['name']):
                        
            if name in names_tot:
                name_ind = np.argwhere( names_tot == name )[0][0]
                
                if name in logparam_names:
                    values[i, name_ind] = format_func(  res.bp.results['sample'][:,j]/np.log(10), weights  )
                elif name == 'BLR model ln(Mbh)':
                    mbh_samps = res.bp.results['sample'][:,j]/np.log(10) + 6
                    values[i, name_ind] = format_func( mbh_samps, weights )
                elif name == 'BLR model Inc':
                    values[i, name_ind] = format_func( np.arccos(res.bp.results['sample'][:,j])*180/np.pi, weights )
                elif name == 'sys_err_line':
                    vals = res.bp.results['sample'][:,j]
                    values[i, name_ind] = format_func( (np.exp(vals) - 1.0) * np.mean(line_lc_err), weights )
                elif name == 'sys_err_con':
                    vals = res.bp.results['sample'][:,j]
                    values[i, name_ind] = format_func( (np.exp(vals) - 1.0) * np.mean(yerr_cont), weights )
                elif name == 'line broaden':
                    vals = res.bp.results['sample'][:,j]*50. + 220.
                    values[i, name_ind] = format_func(vals, weights)
                else:
                    values[i, name_ind] = format_func(res.bp.results['sample'][:,j], weights)

    colnames = np.hstack([ ['Parameter', 'Unit'], res_names ])
    table_input = np.vstack([latex_names, units, values]).T
    dat = Table(table_input, names=colnames)


    col_align_str = '|l|l|'
    for _ in range(len(res_arr)):
        col_align_str += 'c|'

    custom_dict = {'tabletype': 'table*', 'preamble': r'\begin{center}', 'tablefoot': r'\end{center}', 
                    'col_align': col_align_str, 'header_start': r'\hline', 'header_end': r'\hline',
                    'data_end': r'\hline'}


    ascii.write(dat, output=output_fname, Writer=ascii.Latex,
                latexdict=custom_dict, overwrite=True)

    return

