import numpy as np
import sys

import matplotlib.pyplot as plt
from matplotlib import gridspec

from astropy.table import Table
from dmutils.plots import val2latex



def plot_mult(res_arr, res_names=None, bounds_arr=None, tf_ymax_arr=None, tf_xbounds_arr=None, 
              output_fname=None, show=False):
    
    assert len(res_arr) > 0
    nres = len(res_arr)
    
    if res_names is None:
        res_names = [None]*nres
    if bounds_arr is None:
        bounds_arr = [None]*nres
    if tf_ymax_arr is None:
        tf_ymax_arr = [None]*nres
    if tf_xbounds_arr is None:
        tf_xbounds_arr = [None]*nres
        
    if len(bounds_arr) == 2:
        if nres != 2:
            bounds_arr = [bounds_arr]*nres
        else:
            assert isinstance(bounds_arr[0]*1.0, float)
        
    if len(tf_ymax_arr) == 2:
        if nres != 2:
            tf_ymax_arr = [tf_ymax_arr]*nres
        else:
            assert isinstance(tf_ymax_arr[0]*1.0, float)
            
    if len(tf_xbounds_arr) == 2:
        if nres != 2:
            tf_xbounds_arr = [tf_xbounds_arr]*nres
        else:
            assert isinstance(tf_xbounds_arr[0]*1.0, float)
            
    
            
            

        
    
    assert nres == len(res_names) == len(bounds_arr) == len(tf_ymax_arr) == len(tf_xbounds_arr)
    
    fig = fig = plt.figure(figsize=(4*nres, 12))
    gs_tot = gridspec.GridSpec(nres, 3, figure=fig, hspace=.5, width_ratios=[1,1,1.5])
    
    for i, res in enumerate(res_arr):
        gs_i = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_tot[i,:2])
        
        ax1 = fig.add_subplot(gs_i[0])
        ax2 = fig.add_subplot(gs_i[1], sharey=ax1, sharex=ax1)
        ax_clouds = [ax1, ax2]
        ax_tf = fig.add_subplot(gs_tot[i,2])

        ax_clouds = res.plot_clouds(colorbar=True, bounds=bounds_arr[i], ax=ax_clouds, show=False)
        ax_tf = res.transfer_function_2dplot(ax=ax_tf, ymax=tf_ymax_arr[i], xbounds=tf_xbounds_arr[i], show=False)
    
        plt.figtext(.95, .5, res_names[i], fontsize=20, rotation=90, va='center', ha='center')


    if output_fname is not None:
        plt.savefig(output_fname, bbox_inches='tight', dpi=200)
        
    if show:
        plt.show()
        
    plt.cla()
    plt.clf()
    plt.close()
    
    return





def latex_table_mult(res_arr, res_names=None, output_fname=sys.stdout):

    assert len(res_arr) > 0
    nres = len(res_arr)

    if res_names is None:
        res_names = [None]*nres

    names_tot = ['BLR model ln(Rblr)', 'BLR model beta', 'BLR model F',
       'BLR model Inc', 'BLR model Opn', 'BLR model Kappa',
       'BLR model gamma', 'BLR model xi', 'BLR model ln(Mbh)',
       'BLR model fellip', 'BLR model fflow', 'BLR model ln(sigr_circ)',
       'BLR model ln(sigthe_circ', 'BLR model ln(sigr_rad)',
       'BLR model ln(sigthe_rad)', 'BLR model theta_rot',
       'BLR model ln(sig_turb)', 'line broaden', 'sys_err_line',
       'sys_err_con', 'sigmad', 'taud', 'trend', 'A', 'Ag']

    logparam_names = ['BLR model ln(Rblr)', 'BLR model ln(sigr_circ)',
                     'BLR model ln(sigthe_circ', 'BLR model ln(sigr_rad)',
                     'BLR model ln(sigthe_rad)', 'BLR model ln(sig_turb)', 'sigmad', 'taud']

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
        for j in range(len(names_tot)):
            values[i, j] = '---'

        for j, name in enumerate(res.bp.para_names['name']):
                        
            if name in names_tot:
                name_ind = np.argwhere( names_tot == name )[0][0]
                
                if name_ind in logparam_names:
                    values[i, name_ind] = val2latex(  res.bp.results['sample'][:,j]/np.log(10)  )
                elif name_ind == 'BLR model ln(Mbh)':
                    mbh_samps = res.bp.results['sample'][:,j]/np.log(10) + 6
                    values[i, name_ind] = val2latex( mbh_samps )
                elif name_ind == 'BLR model Inc':
                    values[i, name_ind] = val2latex(res.bp.results['sample'][:,j]*180/np.pi )
                else:
                    values[i, name_ind] = val2latex(res.bp.results['sample'][:,j])

    colnames = np.hstack([ ['Parameter', 'Unit'], res_names ])
    table_input = np.vstack([latex_names, units, values]).T
    dat = Table(table_input, names=colnames)


    col_align_str = '|l|l|'
    for _ in range(len(res_arr)):
        col_align_str += 'c|'

    custom_dict = {'tabletype': 'table*', 'preamble': r'\begin{center}', 'tablefoot': r'\end{center}', 
                    'col_align': '|l|l|c|l|l|', 'header_start': r'\hline', 'header_end': r'\hline',
                    'data_end': r'\hline'}


    ascii.write(dat, output=output_fname, Writer=ascii.Latex,
                latexdict=custom_dict)

    return

