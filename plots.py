import os
os.environ['PATH'] = '/home/stone28/software/texlive/install/bin/x86_64-linux/'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from bbackend import postprocess, bplotlib

from palettable.cmocean.sequential import Matter_20_r, Matter_20





def transfer_function_2dplot(param_fname, z, central_wl, ymax=None, xbounds=None, 
                             output_fname=None, show=False):
    bp = bplotlib(param_fname)
    
    
    #Get data
    idx, par = bp.find_max_prob()
    bh_idx = bp.locate_bhmass()

    mbh = 10**( par[bh_idx]/np.log(10) + 6) * 1.99e33

    wl_vals = bp.data['line2d_data']['profile'][0,:,0]/(1+z)
    t_vals = bp.results['tau_rec'][idx]
    vel_vals = 3e5*( wl_vals - central_wl )/central_wl

    c = 3e10
    G = 6.67e-8
    env_vals = mbh * G/c/(vel_vals*1e5)**2
    env_vals /= 60*60*24 # convert to days
    
    
    
    
    plot_arr = bp.results['tran2d_rec'][idx].copy()
    plot_arr[ plot_arr > 1 ] = 0.


    fig, ax = plt.subplots()
    im = ax.imshow( plot_arr, origin='lower', aspect='auto',
                extent=[vel_vals[0], vel_vals[-1],t_vals[0], t_vals[-1]],
                interpolation='gaussian', cmap=Matter_20_r.mpl_colormap)

    ax.plot(vel_vals, env_vals, color='w', ls='--')

    if ymax is not None:
        ax.set_ylim(0, ymax)
        
    if xbounds is not None:
        ax.set_xlim(xbounds[0], xbounds[1])

    ax.set_ylabel('Lag [d]')
    ax.set_xlabel(r'Velocity [$\rm km \ s^{-1} $]')

    plt.colorbar(im, ax=ax)
    ax.set_title(r'Maximum Probability $\rm \Psi(v, t)$')

    ax.axvline(0, color='w', ls='--')

    if output_fname is not None:
        plt.savefig(output_fname, bbox_inches='tight', dpi=200)
        
    if show:
        plt.show()
        
    plt.cla()
    plt.clf()
    plt.close()
    
    return




def list_params(param_fname):
    
    bp = bplotlib(param_fname)
    idx, par = bp.find_max_prob()
    
    mbh = 10**( par[bh_idx]/np.log(10) + 6) * 1.99e33
    
    
    print('Fraction of elliptical orbits: {:.3f}'.format(par[9]))
    print('Radial Direction Param: {:.3f}'.format(par[10]))
    print('Deg Rot on Ellipse of Velocity: {:.3f}'.format(par[15]))
    print('')
    print('Opening Angle: {:.3f} deg'.format(par[4]))
    print('Inclination: {:.3f} deg'.format(par[3]*180/np.pi))
    print('')
    print('log10 Mbh/Msol: {:.3f}'.format( np.log10(mbh/1.99e33)) )
    print('Rblr [ld]: {:.3f}'.format( 10**(par[0]/np.log(10)) ) )
    print('')
    print('log10 DRW Tau: {:.3f}'.format(par[21]))
    print('log10 DRW Sigma: {:.3f}'.format(par[20]))
    print('Trend: {:.3f}'.format(par[22]) )
    print('')
    print('Sys Err Con: {:.3f}'.format(par[19]))
    print('Sys Err Line: {:.3f}'.format(par[18]))
    print('')
    print('Face Concentration: {}'.format(par[6]))
    print('Far(-)/Near(+) Side Concentration: {}'.format(par[5]))
    print('Midplane Transparency: {}'.format(par[7]))
    
    return





def plot_clouds(param_fname, cloud_fname, rotate=False, skip=10, bounds=[-10,10], colorbar=False,
                output_fname=None, show=False):
    
    
    
    """Create a plot of the cloud positions in the BLR, color-coded by their LOS velocity, with sizes corresponding
    to their weights. The LOS velocity will be different for each of the two panels: edge-on and face-on.
    
    Parameters
    ----------
    
    param_fname : str
        The name of the parameter file used to run BRAINS.
        
    cloud_fname : str
        The name of the file containing the cloud positions and velocities.
        
    rotate : bool, optional
        If True, the clouds will be rotated so that the disk is flat along the x-axis. The velocities
        will not be rotated.
        
    skip : int, optional
        The number of clouds to skip when plotting. This is useful for large numbers of clouds. The default
        value is 10, so every 10th cloud will be plotted.
        
    bounds : list, optional
        The bounds of the plot in the x and y directions. The default is [-10,10].
        
    colorbar : bool, optional
        If True, a colorbar for the velocities will be added to the plot. The default is False.
        
    output_fname : str, optional
        The name of the file to save the plot to. If None, the plot will not be saved. The default is None.
        
    show : bool, optional
        If True, the plot will be shown. The default is False.

    
    """

    
    
    bp = bplotlib(param_fname)
    inc = np.median(bp.results['sample'][:,3])
    
    cloud_dat = np.loadtxt(cloud_fname)
    x_vals = cloud_dat[:,0]
    y_vals = cloud_dat[:,1]
    z_vals = cloud_dat[:,2]
    weights = cloud_dat[:,-1]

    vx_vals = cloud_dat[:,3]
    vy_vals = cloud_dat[:,4]
    vz_vals = cloud_dat[:,5]
    
    
    if rotate:
        x_vals = x_vals*np.cos(-inc) + z_vals*np.sin(-inc)
        y_vals = y_vals
        z_vals = -x_vals*np.sin(-inc) + z_vals*np.cos(-inc)

        vx_vals = vx_vals*np.cos(-inc) + vz_vals*np.sin(-inc)
        vy_vals = vy_vals
        vz_vals = -vx_vals*np.sin(-inc) + vz_vals*np.cos(-inc)
        
    
    
    
    
    
    sizes = 40*weights

    fig, ax = plt.subplots(1, 2, figsize=(11,5), sharey=True)
    ax[0].scatter(x_vals[::skip], z_vals[::skip], s=sizes[::skip], c=vy_vals[::skip], 
                marker='o', ec='k', linewidths=.5, alpha=.9, cmap='coolwarm')
    ax[0].set_xlabel('x', fontsize=18)
    ax[0].set_ylabel('z', fontsize=18)
    ax[0].set_title('Edge-On', fontsize=20)
    

    ax[1].scatter(y_vals[::skip], z_vals[::skip], s=sizes[::skip], c=vx_vals[::skip], 
                    marker='o', ec='k', linewidths=.5, alpha=.9, cmap='coolwarm_r')
    ax[1].set_xlabel('y', fontsize=18)
    ax[1].set_title('Face-On', fontsize=20)

    if bounds is not None:
        for a in ax:
            a.set_xlim(bounds)
            a.set_ylim(bounds)

    plt.subplots_adjust(wspace=.08)

    if colorbar:
        max_v = np.max( np.max(vy_vals[::skip]), np.max(vx_vals[::skip]) )
        min_v = np.min( np.min(vy_vals[::skip]), np.min(vx_vals[::skip]) )
        
        norm = Normalize(vmin=min_v, vmax=max_v)
        sm = ScalarMappable(norm=norm, cmap='coolwarm')
        cbar = plt.colorbar(sm, ax=ax)
    
    if output_fname is not None:
        plt.savefig(output_fname, bbox_inches='tight', dpi=200)
    
    if show:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close()
    
    return






def cloud_quiver_plot(param_fname, cloud_fname, ncloud_vel=5, skip=1, max_r=10,
                      output_fname=None, show=False):
    
    """Make a 3D quiver plot of the clouds in the BLR fit.
    

    Parameters
    ----------
    
    param_fname : str
        The name of the parameter file used to run BRAINS.
        
    cloud_fname : str
        The name of the file containing the cloud positions and velocities.
        
    ncloud_vel : int, optional
        The number of velocities chosen per cloud. The default is 5.
        
    skip : int, optional
        The number of clouds to skip when plotting. This is useful for large numbers of clouds. The default
        
    max_r : float, optional
        The maximum radius to plot the clouds out to. The default is 10.
        
    output_fname : str, optional
        The name of the file to save the plot to. If None, the plot will not be saved. The default is None.
        
    show : bool, optional
        If True, the plot will be shown. The default is False.
        
    
    """
    
    bp = bplotlib(param_fname)
    inc = np.median(bp.results['sample'][:,3])
    
    cloud_dat = np.loadtxt(cloud_fname)
    x_vals = cloud_dat[:,0]
    y_vals = cloud_dat[:,1]
    z_vals = cloud_dat[:,2]
    weights = cloud_dat[:,-1]

    vx_vals = cloud_dat[:,3]
    vy_vals = cloud_dat[:,4]
    vz_vals = cloud_dat[:,5]
    
    
    
    
    
    x_sum = x_vals[::ncloud_vel].copy()
    y_sum = y_vals[::ncloud_vel].copy()
    z_sum = z_vals[::ncloud_vel].copy()

    xtot = x_sum[::skip].copy()
    ytot = y_sum[::skip].copy()
    ztot = z_sum[::skip].copy()
    
    
    
    
    
    vx_sum = np.zeros(len(x_vals)//ncloud_vel)
    vy_sum = np.zeros(len(x_vals)//ncloud_vel)
    vz_sum = np.zeros(len(x_vals)//ncloud_vel)
    weights_sum = np.zeros(len(x_vals)//ncloud_vel)
    
    for i in range(len(vx_vals)//ncloud_vel):
        vx_sum[i] = np.sum(vx_vals[ncloud_vel*i:ncloud_vel*(i+1)])
        vy_sum[i] = np.sum(vy_vals[ncloud_vel*i:ncloud_vel*(i+1)])
        vz_sum[i] = np.sum(vz_vals[ncloud_vel*i:ncloud_vel*(i+1)])
        weights_sum[i] = np.sum(weights[ncloud_vel*i:ncloud_vel*(i+1)])
    

    weights_sum /= ncloud_vel
    vxtot = vx_sum[::skip].copy()
    vytot = vy_sum[::skip].copy()
    vztot = vz_sum[::skip].copy()
    weighttot = weights_sum[::skip].copy()
    
    rtot = np.sqrt(xtot**2 + ytot**2 + ztot**2)
    mask = rtot < max_r






    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(15,15))
    
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=Matter_20_r.mpl_colormap)
    
    color_arr = sm.to_rgba(weighttot[mask])
    alpha_arr = weighttot*.7
    
    
    ax.quiver(xtot[mask], ytot[mask], ztot[mask],
            vxtot[mask], vytot[mask], vztot[mask],
            alpha=alpha_arr[mask], normalize=True, length=.7,
            linewidths=1.5)
    
    ax.set_xlabel(r'$\textbf{x}$', fontsize=25, labelpad=20)
    ax.set_ylabel(r'$\textbf{y}$', fontsize=25, labelpad=20)
    ax.set_zlabel(r'$\textbf{z}$', fontsize=25, labelpad=20)

    cbar = plt.colorbar(sm, ax=ax, pad=.1, shrink=.7, aspect=25)
    cbar.ax.set_title(r'$w_i$', fontsize=25, pad=20)
    cbar.ax.tick_params('both', length=10)
    cbar.ax.tick_params('both', length=5, which='minor')    
    cbar.ax.minorticks_on()
    
    if output_fname is not None:
        plt.savefig(output_fname, bbox_inches='tight', dpi=200)
    
    if show:
        plt.show()
        
    plt.cla()
    plt.clf()
    plt.close()
    
    return
