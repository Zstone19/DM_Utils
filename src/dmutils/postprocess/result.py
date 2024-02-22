import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import (BboxConnector,
                                                   TransformedBbox, inset_axes)

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from palettable.cmocean.sequential import Matter_20_r, Matter_20

import numpy as np
import astropy.constants as const
from astropy.io import ascii
from astropy.table import Table

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

from bbackend import postprocess, bplotlib
from pypetal.weighting.utils import get_weights, get_bounds
from pypetal.utils.petalio import err2str


###############################################################################
############################### CLASS DEFINITION ##############################
###############################################################################

class Result:
    
    def __init__(self, fname, line_name=None, latex_dir=None):
        self.bp = bplotlib(fname)
        
        #Load parameters
        self.z = float(self.bp.param._parser._sections['dump']['redshift'])
        self.central_wl = float(self.bp.param._parser._sections['dump']['linecenter'])
        self.ncloud = int(self.bp.param._parser._sections['dump']['ncloudpercore'])
        self.vel_per_cloud = int(self.bp.param._parser._sections['dump']['nvpercloud'])
        
        #Load filenames
        self.param_fname = fname
        
        fname_dir = os.path.dirname( os.path.dirname(fname) )
        self.cloud_fname = fname_dir + '/' + self.bp.param._parser._sections['dump']['cloudsfileout']
        
        
        #Line name
        if line_name is None:
            self.line_name = 'Line'
        else:
            self.line_name = line_name
            
        #Latex directory
        self.latex_dir = latex_dir
            
            

        #Plotting settings
        mpl.rcParams['xtick.minor.visible'] = True
        mpl.rcParams['xtick.top'] = True
        mpl.rcParams['xtick.direction'] = 'in'

        mpl.rcParams['ytick.minor.visible'] = True
        mpl.rcParams['ytick.right'] = True
        mpl.rcParams['ytick.direction'] = 'in'

        mpl.rcParams["figure.autolayout"] = False

        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['savefig.format'] = 'pdf'

###############################################################################
################################ INDIVIDUAL PLOTS #############################
###############################################################################

    def dns_postprocess(self, analysis_dir, temp=1):
        
        cwd = os.getcwd()
        os.chdir(analysis_dir)
        postprocess(2, temp=temp)
        os.chdir(cwd)
        
        return

    
    def dns_limits(self, analysis_dir):

        cwd = os.getcwd()
        os.chdir(analysis_dir)
        self.bp.plot_limits()
        os.chdir(cwd)
        
        return



    def line2d_plot(self, gaussian_smooth=False, gaussian_sig=[0,0], xbounds=None,
                    include_res=True,
                    ax=None, output_fname=None, show=False):


        if ax is None:
            ax_in = False
        else:
            ax_in = True

        c = const.c.cgs.value
        ff = 1.5
        
        idx, _ = self.bp.find_max_prob()
        model_flux = self.bp.results['line2d_rec'][idx]
        data_flux = self.bp.data['line2d_data']['profile'][:,:,1]
        
        vmin = np.min([np.min(model_flux), np.min(data_flux)])
        vmax = np.max([np.max(model_flux), np.max(data_flux)])
        
        vmin = np.max([0, vmin])


        if ax is None:
            
            if include_res:
                Ncol = 3
            else:
                Ncol = 2
                
            fig, ax = plt.subplots(1, Ncol, figsize=(10, Ncol), sharey=True, sharex=True)


        #############################################################
        #Real data

        t = self.bp.data['line2d_data']['time']
        wl = self.bp.data['line2d_data']['profile'][:,:,0]/(1+self.z)

        time = np.zeros( wl.shape )
        for i in range(len(t)):
            time[i,:] = t[i]
            
        vel = (c/1e5)*( wl - self.central_wl )/(self.central_wl)
        
        if gaussian_smooth:
            
            #Get the profile at regularly sampled times and velocities
            spline_func = RegularGridInterpolator( (t, vel[0]), data_flux, method='linear' )
            
            t_interp, dt = np.linspace(t.min(), t.max(), 1000, retstep=True)
            vel_interp, dv = np.linspace(vel[0].min(), vel[0].max(), 1000, retstep=True)        
            gaussian_sig = [ gaussian_sig[0]/dt, gaussian_sig[1]/dv ]
            
            positions_input = []
            for i in range(len(t_interp)):
                for j in range(len(vel_interp)):
                    positions_input.append( [t_interp[i], vel_interp[j]] )
                    
            positions_input = np.array(positions_input)
            
            even_flux = spline_func(positions_input)
            even_flux = even_flux.reshape( (len(t_interp), len(vel_interp)) )
            
            #Convolve with a gaussian kernel
            smooth_flux = gaussian_filter(even_flux, gaussian_sig)
            
            #Sample at the original times and velocities
            spline_func2 = RegularGridInterpolator( (t_interp, vel_interp), smooth_flux, method='linear' )
            
            positions_input2 = []
            for i in range(len(t)):
                for j in range(len(vel[0])):
                    positions_input2.append( [t[i], vel[0][j]] )        
            
            plot_flux = spline_func2(positions_input2)
            plot_flux = plot_flux.reshape( (len(t), len(vel[0])) )
            
            
            
            t_interp_plot = np.zeros((1000,1000))
            vel_interp_plot = np.zeros((1000,1000))
            for i in range(1000):
                t_interp_plot[i,:] = t_interp
                vel_interp_plot[:,i] = vel_interp
            
            del t_interp, vel_interp, spline_func, even_flux
            del smooth_flux, spline_func2
            
        else:
            plot_flux = data_flux.copy()
        
        
        ax[0].pcolormesh(vel/1e3, time, plot_flux, cmap=Matter_20_r.mpl_colormap, vmin=vmin, vmax=vmax)
        
        #############################################################
        #Model data
        
        nt = int(self.bp.param._parser._sections['dump']['nlinerecon'])
        nv = int(self.bp.param._parser._sections['dump']['nvelrecon'])
        
        ax[1].pcolormesh(vel/1e3, time, model_flux, cmap=Matter_20_r.mpl_colormap, vmin=vmin, vmax=vmax)
        
        #############################################################
        #Residuals
        
        if include_res:
            prof_err = self.bp.data['line2d_data']['profile'][:, :, 2].copy()
            line_mean_err = np.mean(prof_err)
            
            sample = self.bp.results['sample']
            idx_line =  np.nonzero(self.bp.para_names['name'] == 'sys_err_line')[0][0]
            syserr_line = (np.exp(np.median(sample[:, idx_line])) - 1.0) * line_mean_err
            
            
            
            im = ax[2].pcolormesh( vel/1e3, time, (plot_flux - model_flux)/np.sqrt(prof_err**2 + syserr_line**2), 
                            cmap='RdBu_r', vmin=-7, vmax=7)

        #############################################################
        #Aesthetics
        
        ax[0].set_ylabel('MJD', fontsize=13*ff, labelpad=10)
        
        if not ax_in:
            ax[0].set_title('Data', fontsize=16*ff)
            ax[1].set_title('Model', fontsize=16*ff)
        
        if (include_res) & (not ax_in):
            ax[2].set_title('Residuals', fontsize=16*ff)
        
        for a in ax:
            
            if not ax_in:
                a.set_xlabel(r'Velocity [$\rm 10^3 \; km \; s^{-1}$]', fontsize=11*ff, labelpad=5)

            a.tick_params('both', which='major', length=6)
            a.tick_params('both', which='minor', length=3)
            a.tick_params('both', labelsize=10*ff)
            
        for a in ax[1:]:
            a.tick_params('y', which='both', labelsize=0)
            
        for a in ax[:2]:
            a.tick_params('both', which='both', color='w', width=1.5)
            
        if xbounds is not None:
            for a in ax:
                a.set_xlim(xbounds)
        
        
        #plt.subplots_adjust(wspace=0.05)
        
        if include_res:
            cbar = plt.colorbar(im, ax=ax[2], pad=0.01, aspect=15 )
            cbar.ax.set_ylabel(r'Units of $\sigma$', fontsize=12, rotation=270, labelpad=10)
    
        if output_fname is not None:
            plt.savefig(output_fname, bbox_inches='tight', dpi=200)
        
        if show:
            plt.show()
        
        if ax is not None:
            return ax
        else:
            return fig, ax


    def transfer_function_2dplot(self, ymax=None, xbounds=None, 
                                 vmin=None, vmax=None,
                                 ax=None, output_fname=None, show=False):
        
        if ax is None:
            ax_in = False
        else:
            ax_in = True

        
        c = const.c.cgs.value
        G = const.G.cgs.value
        Msol = const.M_sun.cgs.value
        
        #Get data
        idx, par = self.bp.find_max_prob()
        bh_idx = self.bp.locate_bhmass()

        mbh = 10**( par[bh_idx]/np.log(10) + 6) * Msol

        wl_vals = self.bp.data['line2d_data']['profile'][0,:,0]/(1+self.z)
        t_vals = self.bp.results['tau_rec'][idx]
        vel_vals = (c/1e5)*( wl_vals - self.central_wl )/self.central_wl
        
        env_vals = mbh * G/c/(vel_vals*1e5)**2
        env_vals /= 60*60*24 # convert to days
        
        
        
        
        plot_arr = self.bp.results['tran2d_rec'][idx].copy()
        plot_arr[ plot_arr > 1 ] = 0.


        if ax is None:
            fig, ax = plt.subplots()


        im = ax.imshow( plot_arr, origin='lower', aspect='auto',
                    extent=[vel_vals[0]/1000, vel_vals[-1]/1000, t_vals[0], t_vals[-1]],
                    interpolation='gaussian', cmap=Matter_20_r.mpl_colormap, 
                    vmin=vmin, vmax=vmax)

        ax.plot(vel_vals/1000, env_vals, color='c', ls='--', lw=2)

        if ymax is not None:
            ax.set_ylim(0, ymax)
            
        if xbounds is not None:
            ax.set_xlim(xbounds[0], xbounds[1])
            
            if not ax_in:
                ax.set_xlabel(r'Velocity [$\rm 10^3 \; km \ s^{-1} $]', fontsize=16)

        ax.set_ylabel('Lag [d]', fontsize=16)
        
        cbar = plt.colorbar(im, ax=ax, pad=.01, aspect=15)
        cbar.ax.tick_params('both', labelsize=14)

        if not ax_in:
            ax.set_title(r'Max Likelihood $\rm \Psi(v, t)$', fontsize=18)

        ax.axvline(0, color='c', ls='--', lw=2)


        ax.tick_params('both', which='both', color='w', width=1.5)
        ax.tick_params('both', which='major', length=6)
        ax.tick_params('both', which='minor', length=3)
        ax.tick_params('both', labelsize=14)
                

        if output_fname is not None:
            plt.savefig(output_fname, bbox_inches='tight', dpi=200)
            
        if show:
            plt.show()
            
            
        if ax is not None:
            return ax
        else:
            return fig, ax


    def plot_clouds(self, rotate=False, skip=10, bounds=[-10,10],
                    plot_rblr=True, 
                    colorbar=False, vmin=None, vmax=None,
                    ax=None, output_fname=None, show=False):
        
        
        
        """Create a plot of the cloud positions in the BLR, color-coded by their LOS velocity, with sizes corresponding
        to their weights. The LOS velocity will be different for each of the two panels: edge-on and face-on.
        
        Parameters
        ----------
            
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


        if ax is None:
            ax_in = False
        else:
            ax_in = True


        inc = np.median(self.bp.results['sample'][:,3])
        rblr = 10**np.median(self.bp.results['sample'][:,0])
        
        cloud_dat = np.loadtxt(self.cloud_fname)
        x_vals = cloud_dat[:,0]
        y_vals = cloud_dat[:,1]
        z_vals = cloud_dat[:,2]
        weights = cloud_dat[:,-1]

        vx_vals = cloud_dat[:,3]
        vy_vals = cloud_dat[:,4]
        vz_vals = cloud_dat[:,5]
        
        x_rblr = np.linspace(bounds[0], bounds[1], 5000)
        y_rblr1 = np.sqrt(rblr**2 - x_rblr**2)
        y_rblr2 = -y_rblr1.copy()
        z_rblr = np.zeros_like(x_rblr)
        
        
        zmin = (bounds[0] + bounds[1])/2 - (bounds[1]-bounds[0])/10
        zmax = (bounds[0] + bounds[1])/2 + (bounds[1]-bounds[0])/10
        xline1 = np.full( 1000, -rblr )
        zline1 = np.linspace(zmin, zmax, 1000)
        xline2 = np.full( 1000, rblr )
        zline2 = zline1.copy()
        
        
        if rotate:
            x_vals0 = x_vals.copy()
            y_vals0 = y_vals.copy()
            z_vals0 = z_vals.copy()
            vx_vals0 = vx_vals.copy()
            vy_vals0 = vy_vals.copy()
            vz_vals0 = vz_vals.copy()
            
            #Rotate so that the disk is flat along the x-axis
            x_vals = x_vals0*np.cos(-inc) + z_vals0*np.sin(-inc)
            y_vals = y_vals0
            z_vals = -x_vals0*np.sin(-inc) + z_vals0*np.cos(-inc)

            vx_vals = vx_vals0*np.cos(-inc) + vz_vals0*np.sin(-inc)
            vy_vals = vy_vals0
            vz_vals = -vx_vals0*np.sin(-inc) + vz_vals0*np.cos(-inc)
                        
        else:
            x_rblr0 = x_rblr.copy()
            z_rblr0 = z_rblr.copy()
            
            xline10 = xline1.copy()
            zline10 = zline1.copy()
            xline20 = xline2.copy()
            zline20 = zline2.copy()
            
            #Rotate so that the radius drawing matches the disk inclination
            x_rblr = x_rblr0*np.cos(inc) + z_rblr0*np.sin(inc)
            z_rblr = -x_rblr0*np.sin(inc) + z_rblr0*np.cos(inc)
            
            xline1 = xline10*np.cos(inc) + zline10*np.sin(inc)
            zline1 = -xline10*np.sin(inc) + zline10*np.cos(inc)
            xline2 = xline20*np.cos(inc) + zline20*np.sin(inc)
            zline2 = -xline20*np.sin(inc) + zline20*np.cos(inc)
        
        
        
        
        sizes = 40*weights

        
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(11,5), sharey=True)    
        
        
        ax[0].scatter(x_vals[::skip], z_vals[::skip], s=sizes[::skip], c=vy_vals[::skip]/1000, 
                    marker='o', ec='k', linewidths=.5, alpha=.9, cmap='coolwarm')
        ax[0].set_ylabel('z [lt-d]', fontsize=20)
        

        ax[1].scatter(y_vals[::skip], z_vals[::skip], s=sizes[::skip], c=vx_vals[::skip]/1000, 
                        marker='o', ec='k', linewidths=.5, alpha=.9, cmap='coolwarm_r')
        

        if plot_rblr:
            ax[1].plot(y_rblr1, z_rblr, color='k', ls='--', lw=1.5, zorder=1000000)
            ax[1].plot(y_rblr2, z_rblr, color='k', ls='--', lw=1.5, zorder=1000000)
            
            ax[0].plot(xline1, zline1, color='k', ls='--', lw=1.5, zorder=1000000)
            ax[0].plot(xline2, zline2, color='k', ls='--', lw=1.5, zorder=1000000)

        
        if not ax_in:
            ax[0].set_xlabel('x [lt-d]', fontsize=20)
            ax[1].set_xlabel('y [lt-d]', fontsize=20)
            ax[0].set_title('Side View', fontsize=22)
            ax[1].set_title('Observer POV', fontsize=22)

        if bounds is not None:
            for a in ax:
                a.set_xlim(bounds)
                a.set_ylim(bounds)


        for a in ax:
            a.tick_params('both', which='major', length=8)
            a.tick_params('both', which='minor', length=3)
            a.tick_params('x', which='both', labelsize=12)

        ax[0].tick_params('y', which='both', labelsize=14)
        ax[1].tick_params('y', which='both', labelsize=0)
        
        if ax is None:
            plt.subplots_adjust(wspace=.08)

        if colorbar:
            
            if vmax is None:
                max_v = np.max( [np.max(vy_vals[::skip]), np.max(vx_vals[::skip])] )  
            else:
                max_v = vmax
      
            if vmin is None:
                min_v = np.min( [np.min(vy_vals[::skip]), np.min(vx_vals[::skip])] )
            else:
                min_v = vmin
            
            norm = Normalize(vmin=min_v/1000, vmax=max_v/1000)
            sm = ScalarMappable(norm=norm, cmap='coolwarm')
            cbar = plt.colorbar(sm, ax=ax, pad=.01, aspect=15)
            
            cbar.ax.set_ylabel(r'Velocity [$\rm 10^{3} \; km \; s^{-1} $]', fontsize=17, rotation=270, labelpad=25)
            cbar.ax.tick_params('both', labelsize=14)
        
        if output_fname is not None:
            plt.savefig(output_fname, bbox_inches='tight', dpi=200)
        
        if show:
            plt.show()

        if ax is not None:
            return ax
        else:
            return fig, ax





    def cloud_quiver_plot(self, skip=1, max_r=10,
                          ax=None, color=False, length=.7, output_fname=None, show=False):
        
        """Make a 3D quiver plot of the clouds in the BLR fit.
        

        Parameters
        ----------
            
        skip : int, optional
            The number of clouds to skip when plotting. This is useful for large numbers of clouds. The default
            
        max_r : float, optional
            The maximum radius to plot the clouds out to. The default is 10.
            
        output_fname : str, optional
            The name of the file to save the plot to. If None, the plot will not be saved. The default is None.
            
        show : bool, optional
            If True, the plot will be shown. The default is False.
            
        
        """
        
        cloud_dat = np.loadtxt(self.cloud_fname)
        x_vals = cloud_dat[:,0]
        y_vals = cloud_dat[:,1]
        z_vals = cloud_dat[:,2]
        weights = cloud_dat[:,-1]

        vx_vals = cloud_dat[:,3]
        vy_vals = cloud_dat[:,4]
        vz_vals = cloud_dat[:,5]
        
        
        
        
        
        x_sum = x_vals[::self.vel_per_cloud].copy()
        y_sum = y_vals[::self.vel_per_cloud].copy()
        z_sum = z_vals[::self.vel_per_cloud].copy()

        xtot = x_sum[::skip].copy()
        ytot = y_sum[::skip].copy()
        ztot = z_sum[::skip].copy()
        
        
        
        
        
        vx_sum = np.zeros(len(x_vals)//self.vel_per_cloud)
        vy_sum = np.zeros(len(x_vals)//self.vel_per_cloud)
        vz_sum = np.zeros(len(x_vals)//self.vel_per_cloud)
        weights_sum = np.zeros(len(x_vals)//self.vel_per_cloud)
        
        for i in range(len(vx_vals)//self.vel_per_cloud):
            vx_sum[i] = np.sum(vx_vals[self.vel_per_cloud*i:self.vel_per_cloud*(i+1)])
            vy_sum[i] = np.sum(vy_vals[self.vel_per_cloud*i:self.vel_per_cloud*(i+1)])
            vz_sum[i] = np.sum(vz_vals[self.vel_per_cloud*i:self.vel_per_cloud*(i+1)])
            weights_sum[i] = np.sum(weights[self.vel_per_cloud*i:self.vel_per_cloud*(i+1)])
        

        weights_sum /= self.vel_per_cloud
        vxtot = vx_sum[::skip].copy()
        vytot = vy_sum[::skip].copy()
        vztot = vz_sum[::skip].copy()
        weighttot = weights_sum[::skip].copy()
        
        rtot = np.sqrt(xtot**2 + ytot**2 + ztot**2)
        mask = rtot < max_r





        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(15,15))
    
    
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(norm=norm, cmap=Matter_20_r.mpl_colormap)
        
        color_arr = sm.to_rgba(weighttot[mask])
        alpha_arr = weighttot*.7
        
        if color:
            ax.quiver(xtot[mask], ytot[mask], ztot[mask],
                    vxtot[mask], vytot[mask], vztot[mask],
                    alpha=alpha_arr[mask], normalize=True, length=length,
                    color=color_arr, linewidths=1.5)

        else:
            ax.quiver(xtot[mask], ytot[mask], ztot[mask],
                    vxtot[mask], vytot[mask], vztot[mask],
                    alpha=alpha_arr[mask], normalize=True, length=length,
                    linewidths=1.5)
        
        ax.set_xlabel(r'$x$ [lt-d]', fontsize=25, labelpad=20)
        ax.set_ylabel(r'$y$ [lt-d]', fontsize=25, labelpad=20)
        ax.set_zlabel(r'$z$ [lt-d]', fontsize=25, labelpad=20)

        if color:
            cbar = plt.colorbar(sm, ax=ax, pad=.1, shrink=.7, aspect=25)
            cbar.ax.set_title(r'$w_i$', fontsize=25, pad=20)
            cbar.ax.tick_params('both', length=10)
            cbar.ax.tick_params('both', length=5, which='minor')    
            cbar.ax.minorticks_on()
        
        if output_fname is not None:
            plt.savefig(output_fname, bbox_inches='tight', dpi=200)
        
        if show:
            plt.show()
            
        if ax is not None:
            return ax
        else:
            return fig, ax
        
        

    def lc_fits_plot(self, inflate_err=False, ax=None, output_fname=None, show=False):
        
        if ax is None:
            ax_in = False
        else:
            ax_in = True
        
        c = const.c.cgs.value
        ff = 2
        
        if ax is None:
            _, ax = plt.subplots(2, figsize=(10, 4), sharex=True) 
        
        con_err_idx = np.nonzero(self.bp.para_names['name'] == 'sys_err_con')[0][0]
        line_err_idx = np.nonzero(self.bp.para_names['name'] == 'sys_err_line')[0][0]       
        
                
        ######################################################
        #Continuum
        
        xin, yin, yerrin = self.bp.data['con_data'].T
        xout = self.bp.results['con_rec'][0,:,0]
        yout_lo, yout_med, yout_hi = np.percentile(self.bp.results['con_rec'][:,:,1], [16, 50, 84], axis=0)
        
        con_mean_err = np.mean(yerrin)
        syserr_con = (np.exp(np.median(self.bp.results['sample'][:, con_err_idx])) - 1.0) * con_mean_err
        
        lines, caps, bars = ax[0].errorbar(xin, yin, np.sqrt(yerrin**2 + syserr_con**2), fmt='.k', ms=1.5)
        [bar.set_alpha(.3) for bar in bars]
        
        ax[0].plot(xout, yout_med, color='orange')
        ax[0].fill_between(xout, yout_lo, yout_hi, color='orange', alpha=.3)        
        
        
        xmin = np.min(xin) - .1*( np.max(xin) - np.min(xin) )
        
        
        ######################################################
        #Line
        
        prof = self.bp.data['line2d_data']['profile'][:,:,1]
        prof_err = self.bp.data['line2d_data']['profile'][:,:,2]
        
        wl_vals = self.bp.data['line2d_data']['profile'][0,:,0].copy()
        vel_vals = (c/1e5)*( (wl_vals/(1+self.z)) - self.central_wl )/self.central_wl #km/s
        dV = (vel_vals[1] - vel_vals[0])/self.bp.VelUnit #km/s
        
        
        line_lc = np.sum(prof, axis=1)*dV
        line_lc_err = np.sqrt( np.sum(prof_err**2, axis=1) )*dV
        
        line_mean_err = np.mean(line_lc_err)
        syserr_line = (np.exp(np.median(self.bp.results['sample'][:, line_err_idx])) - 1.0) * line_mean_err
        
        
        
        xin = self.bp.data['line2d_data']['time']
        yin = line_lc*self.central_wl*self.bp.VelUnit/(c/1e5)
        yerrin = np.sqrt(line_lc_err**2 + (syserr_line**2)*(dV**2))
        
        rec_line_lc = np.sum(self.bp.results['line2d_rec'], axis=2)*dV
        yout_lo, yout_med, yout_hi = np.percentile(rec_line_lc, [16, 50, 84], axis=0) * self.central_wl*self.bp.VelUnit/(c/1e5)
        
        if inflate_err:
            yerrin_mask = (yerrin < .05*yout_med)
            yerrin[yerrin_mask] = .05*yout_med

        lines, caps, bars = ax[1].errorbar(xin, yin, yerrin, fmt='.k', ms=1.5)
        [bar.set_alpha(.3) for bar in bars]
        
        ax[1].plot(xin, yout_med, color='orange')
        ax[1].fill_between(xin, yout_lo, yout_hi, color='orange', alpha=.3)
        
        
        ######################################################
        
        ax[0].set_ylabel(r'$F_{\rm cont}$', fontsize=14*ff, rotation=270, labelpad=35)
        ax[1].set_ylabel(r'$F_{ \rm ' + self.line_name + r'}$', fontsize=14*ff, rotation=270, labelpad=35)
        
        if not ax_in:
            ax[-1].set_xlabel('MJD', fontsize=15*ff)
        
        for a in ax:
            a.tick_params('both', which='major', length=8)
            a.tick_params('both', which='minor', length=3)
            a.tick_params('both', which='major', labelsize=10*ff)
            
            a.set_xlim(left=xmin)
            a.yaxis.set_label_position("right")
            a.tick_params('y', which='both', labelleft=False, labelright=True)
            
        ax[0].tick_params('x', which='both', labelbottom=False)
        

        #plt.subplots_adjust(hspace=0)
        
        fig = plt.gcf()
        fig.align_ylabels(ax)

        if output_fname is not None:
            plt.savefig(output_fname, bbox_inches='tight', dpi=200)
            
        if show:
            plt.show()
            
        if ax is not None:
            return ax
        else:
            return fig, ax
        
        
        
    def prof_fit_quality_plot(self, wl_range=200, ymax_l=35, ymax_r=2,
                              ax=None, output_fname=None, show=False):
        
        time_in = self.bp.data['line2d_data']['time']
        wl_in = self.bp.data['line2d_data']['profile'][0,:,0]/(1+self.z)
        flux_in = self.bp.data['line2d_data']['profile'][:,:,1]
        err_in = self.bp.data['line2d_data']['profile'][:,:,2]


        nt = self.bp.results['line2d_rec'].shape[1]
        nv = self.bp.results['line2d_rec'].shape[2]
        prof_lo, prof_med, prof_hi = np.percentile(self.bp.results['line2d_rec'], [16,50,84], axis=0)

        time_out = np.linspace(time_in.min(), time_in.max(), nt)
        wl_out = np.linspace(wl_in.min(), wl_in.max(), nv)



        
        
        mask = (wl_in > self.central_wl - wl_range) & (wl_in < self.central_wl + wl_range)

        if ax is None:
            fig, ax = plt.subplots( 1,2, figsize=(11,4) )

        norm = Normalize(vmin=time_in[0], vmax=time_in[-1])
        sm = ScalarMappable(norm=norm, cmap='Reds')

        chi2 = []
        for i in range(nt):
            ax[1].plot( wl_in[mask], (prof_med[i][mask] - flux_in[i][mask])/flux_in[i][mask], color=sm.to_rgba(time_in[i]) )
            chi2.append( np.sum( (prof_med[i][mask] - flux_in[i][mask])**2 / err_in[i][mask]**2 )/self.bp.results['sample'].shape[1] )
            
        ax[1].set_ylim(-1.5, ymax_r)

        ax[1].set_ylabel( r'$ (F_{\rm model} - F_{\rm data})/F_{\rm data} $' )
        ax[1].set_xlabel(r'Rest Wavelength [$\rm \AA$]')





        ax[0].plot( time_in, chi2, lw=.5 )
        ax[0].scatter( time_in, chi2, color='k', s=5 )
        ax[0].axhline(1, ls='--', color='gray')

        ax[0].set_ylim(0, ymax_l)

        ax[0].set_ylabel(r'$\chi^2_\nu$')
        ax[0].set_xlabel(r'MJD')



        plt.subplots_adjust(wspace=0.3)
        cbar = plt.colorbar(sm, ax=ax, pad=0.01)
        cbar.ax.set_title('MJD')

        if output_fname is not None:
            plt.savefig(output_fname, bbox_inches='tight', dpi=200)

        if show:
            plt.show()
    
        if ax is not None:
            return ax
        else:
            return fig, ax   
        
        
############################################################################################################## 
        
    def plot_lag_posterior(self, weight=True, k=2, width=15,
                           ax=None, output_fname=None, show=False):
        
        c = const.c.cgs.value
        
        if ax is None:
            if weight:
                gs = gridspec.GridSpec(1, 1)                
                sub_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0],
                                                        height_ratios=[1, 3], hspace=0)
                ax_bot = plt.subplot(sub_gs[1])
                ax_top = plt.subplot(sub_gs[0], sharex=ax_bot)

                ax = [ax_top, ax_bot]
            else:
                fig, ax = plt.subplots(1, 1, figsize=(5,5), sharey=True)
            
            
        psi2d = self.bp.results['tran2d_rec']
        tau_vals = self.bp.results['tau_rec']
        sum2_arr = psi2d.sum(axis=2).sum(axis=1)
        sum1_arr = np.sum( psi2d.sum(axis=2)*tau_vals, axis=1)
        lag_posterior = sum1_arr / sum2_arr
        
        
        if weight:
            xc = self.bp.data['con_data'].T[0]
            yc = self.bp.data['con_data'].T[1]
            
            xl = self.bp.data['line2d_data']['time']
            
            prof = self.bp.data['line2d_data']['profile'][:,:,1]            
            wl_vals = self.bp.data['line2d_data']['profile'][0,:,0].copy()
            vel_vals = (c/1e5)*( (wl_vals/(1+self.z)) - self.central_wl )/self.central_wl #km/s
            dV = (vel_vals[1] - vel_vals[0])/self.bp.VelUnit #km/s
            line_lc = np.sum(prof, axis=1)*dV
            yl = line_lc*self.central_wl*self.bp.VelUnit/(c/1e5)
            
            assert np.all( np.isfinite(xc) )
            assert np.all( np.isfinite(yc) )
            assert np.all( np.isfinite(xl) )
            assert np.all( np.isfinite(yl) )
            wtau, lags, ntau, acf, n0 = get_weights(xc-xc[0], yc, xl-xc[0], yl, k=k)

        
        
            min_bound, peak, max_bound, smooth_dist, smooth_weight_dist = get_bounds(lag_posterior, wtau, lags, width=width, rel_height=.99)
            downsampled_posterior = lag_posterior[(lag_posterior > min_bound) & (lag_posterior < max_bound)]

            med_lag = np.median(downsampled_posterior)
            lag_err_lo = med_lag - np.percentile( downsampled_posterior, 16 )
            lag_err_hi = np.percentile( downsampled_posterior, 84 ) - med_lag
            
            _, ax = plot_weight_output(lags, lag_posterior, 
                       lag_err_lo, lag_err_hi, med_lag, 
                       min_bound, max_bound, peak, 
                       wtau, smooth_dist, acf, 
                       ax_tot=ax, show=False)
            
        else:            
            ax.hist(lag_posterior, bins=25)
            ax.axvline(np.median(lag_posterior), color='r', ls='--')
            
            lag_text = r'$\tau = ' + '{:.3f}'.format(np.median(lag_posterior)) + r' $'
            ax.text( .05, .95, lag_text, transform=ax.transAxes, fontsize=15, va='top', ha='left')
            ax.set_xlabel(r'$\tau$ [d]', fontsize=15)
            
        
        
        if output_fname is not None:
            plt.savefig(output_fname, bbox_inches='tight', dpi=200)

        if show:
            plt.show()
    
        if ax is not None:
            return ax
        else:
            return fig, ax  


###############################################################################
################################# SUMMARY PLOTS ###############################
###############################################################################

    def add_latex_dir(self):
        assert self.latex_dir is not None
        os.environ['PATH'] = self.latex_dir
        return

    def li_summary_style2022(self):
        assert self.latex_dir is not None
        self.add_latex_dir()
        self.bp.plot_results_2d_style2022()
        return

    def li_summary_style2018(self):
        assert self.latex_dir is not None
        self.add_latex_dir()
        self.bp.plot_results_2d_style2018()
        return


    def summary1(self, tf_ymax=500, tf_xbounds=[-5000,5000], line_xbounds=None,
                 include_res=True,
                 output_fname=None, show=False):
        
        """Similar to the Li+2022 plot, but instead of profile fits, put the transfer function in.
        """
        
        fig = plt.figure(figsize=(12, 7))
        gs_tot = gridspec.GridSpec(2, 12, figure=fig, hspace=.5, height_ratios=[1,1.1])
        
        
        #TOP: Profiles
        
        if include_res:
            gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_tot[0,:])
            ax1 = fig.add_subplot(gs_top[0])
            ax2 = fig.add_subplot(gs_top[1], sharey=ax1, sharex=ax1)
            ax3 = fig.add_subplot(gs_top[2], sharey=ax1, sharex=ax1)
            ax_top = [ax1, ax2, ax3]
            
        else:
            gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_tot[0,:])
            ax1 = fig.add_subplot(gs_top[0])
            ax2 = fig.add_subplot(gs_top[1], sharey=ax1, sharex=ax1)
            ax_top = [ax1, ax2]
        
        
        ax_top = self.line2d_plot(xbounds=line_xbounds, ax=ax_top, show=False)
        
            #Set line2d labels
        prof_titles = ['Data', 'Model', 'Residuals']
        ff = 1.5  
        for n in range(len(ax_top)):
            ax_top[n].set_title(prof_titles[n], fontsize=16*ff)
            ax_top[n].set_xlabel(r'Velocity [$\rm 10^3 \; km \; s^{-1}$]', fontsize=11*ff, labelpad=10)


        
        
        
        gs_bot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_tot[1,:], 
                                                  width_ratios=[1.5,1,1], wspace=.1)
        
        #BOTTOM LEFT: Transfer Function
        ax_bl = fig.add_subplot(gs_bot[0])
        ax_bl = self.transfer_function_2dplot(ax=ax_bl, ymax=tf_ymax, xbounds=tf_xbounds, show=False)
        
            #Set tf labels
        ax_bl.set_title(r'Max Likelihood $\rm \Psi(v, t)$', fontsize=18)
        ax_bl.set_xlabel(r'Velocity [$\rm km \ s^{-1} $]', fontsize=16)
        
        
        
        
        #BOTTOM RIGHT: LC Fits
        gs_br = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_bot[1:])
        ax1 = fig.add_subplot(gs_br[0])
        ax2 = fig.add_subplot(gs_br[1], sharex=ax1)
        ax_br = [ax1, ax2]
        
        ax_br = self.lc_fits_plot(ax=ax_br, show=False)
        
        
            #Set lc labels
        ax_br[-1].set_xlabel('MJD', fontsize=15*ff)
        
        

        
        
        if output_fname is not None:
            plt.savefig(output_fname, bbox_inches='tight', dpi=200)
            
        if show:
            plt.show()
            
        plt.cla()
        plt.clf()
        plt.close()
        
        return
    
    
    
    
    def summary2(self, bounds=[-50, 50], plot_rblr=True, weight=True, output_fname=None, show=False):
        
        fig = plt.figure(figsize=(20,10))
        gs_tot = gridspec.GridSpec(2, 4, figure=fig, wspace=.1)
        
        
        #LEFT: Quiver Plot
        ax_l = fig.add_subplot(gs_tot[:2,:2], projection='3d')
        max_r = np.mean(np.abs(bounds))
        ax_l = self.cloud_quiver_plot(max_r=max_r, length=5, color=False, ax=ax_l, show=False)
        
        
        #TOP RIGHT: Cloud Positions
        gs_tr = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_tot[0,2:], wspace=.05)
        ax1 = fig.add_subplot(gs_tr[0])
        ax2 = fig.add_subplot(gs_tr[1], sharey=ax1, sharex=ax1)
        ax_tr = [ax1, ax2]
        
        ax_tr = self.plot_clouds(plot_rblr=plot_rblr, colorbar=True, bounds=bounds, ax=ax_tr, show=False)
        
            #Set cloud labels
        ax_tr[0].set_title('Side View', fontsize=22)
        ax_tr[1].set_title('Observer POV', fontsize=22)
        ax_tr[0].set_xlabel('x [lt-d]', fontsize=20)
        ax_tr[1].set_xlabel('y [lt-d]', fontsize=20)
        
        
        #BOTTOM RIGHT: Posteriors
        gs_br = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_tot[1,2:])
        ax1 = fig.add_subplot(gs_br[:, 0])
        
            #MBH
        mbh_samples = self.bp.results['sample'][:,self.bp.locate_bhmass()]/np.log(10) + 6        
        ax1.hist(mbh_samples, bins=25)
        ax1.axvline(np.median(mbh_samples), color='r', ls='--')
        
        mbh_text = val2latex(mbh_samples)
        ax1.text( .05, .95, mbh_text, transform=ax1.transAxes, fontsize=15, va='top', ha='left')
        ax1.set_xlabel(r'$\log_{10}(M_{BH}/M_{\odot})$', fontsize=15)        
        
            #Lag
        if weight:
            sub_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_br[:, 1],
                                                    height_ratios=[1, 3], hspace=0)
            ax_bot = plt.subplot(sub_gs[1])
            ax_top = plt.subplot(sub_gs[0], sharex=ax_bot)
            ax2 = [ax_top, ax_bot]
        
        else:
            ax2 = fig.add_subplot(gs_br[:, 1])

        
        self.plot_lag_posterior(weight=weight, k=2, width=15,
                                ax=ax2, show=False)


        
        if output_fname is not None:
            plt.savefig(output_fname, bbox_inches='tight', dpi=200)
        
        if show:
            plt.show()
            
        plt.cla()
        plt.clf()
        plt.close()
    
        return


###############################################################################
################################ LISTS & TABLES ###############################
###############################################################################

    def list_params(self):
        
        Msol = const.M_sun.cgs.value
        
        _, par = self.bp.find_max_prob()
        bh_idx = self.bp.locate_bhmass()
        
        mbh = 10**( par[bh_idx]/np.log(10) + 6) * 1.99e33
        
        
        print('Fraction of elliptical orbits: {:.3f}'.format(par[9]))
        print('Radial Direction Param: {:.3f}'.format(par[10]))
        print('Deg Rot on Ellipse of Velocity: {:.3f}'.format(par[15]))
        print('')
        print('Opening Angle: {:.3f} deg'.format(par[4]))
        print('Inclination: {:.3f} deg'.format(par[3]*180/np.pi))
        print('')
        print('log10 Mbh/Msol: {:.3f}'.format( np.log10(mbh/Msol)) )
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
    


    def make_latex_param_table(self, output_fname=sys.stdout):

        cutoff_ind = len(self.bp.para_names['name'])
        while self.bp.para_names['name'][cutoff_ind-1] == 'time series':
            cutoff_ind -= 1
            
        names = self.bp.para_names['name'][:cutoff_ind]
        
        latex_names = [r'$\log_{10}(R_{BLR})$', r'$\beta$', r'$F$', r'$i$', r'$\theta_{opn}$',
                    r'$\kappa$', r'$\gamma$', r'$\xi$', r'$\log_{10}(M_{BH})$',
                    r'$f_{ellip}$', r'$f_{flow}$', r'$\log_{10}(\sigma_{\rho, circ})$', r'$\log_{10}(\sigma_{\theta, circ})$',
                    r'$\log_{10}(\sigma_{\rho, rad})$', r'$\log_{10}(\sigma_{\theta, rad})$', r'$\theta_{e}$', r'$\log_{10}(\sigma_{turb})$',
                    r'$\Delta V_{line}$', r'$\sigma_{sys, line}$', r'$\sigma_{sys, con}$',
                    r'$\log_{10}( \sigma_d )$', r'$\log_{10}( \tau_d )$', r'B', r'$A$', r'$A_g$']
        latex_names = latex_names[:cutoff_ind]
        
        units = ['lt-day', '', r'$R_{BLR}$', 'deg', 'deg', '', '', '', r'$M_{\odot}$', '', '', '', '', '', '', 'deg',
                r'$v_{circ}$', r'$\rm km \; s^{-1}$', '', '', '', 'd', r'$\rm d^{-1}$', '', '']
        
        param_names_better = ['Mean BLR Radius', 'BLR Radial Dist Shape Parameter', 'Inner Edge of the BLR',
                              'Inclination Angle', 'Opening Angle', 'Nearside/Farside Preference', 
                              'Disk Outer Face Preference', 'Midplane Transparency', 'SMBH Mass',
                              'Fraction of Elliptical Orbits', 'Radial Orbit Direction', 
                              'Radial Velocity StdDev for Elliptical Orbits',
                              'Tangential Velocity StdDev for Elliptical Orbits',
                              'Radial Velocity StdDev for Radial Orbits',
                              'Tangential Velocity StdDev for Radial Orbits',
                              'Location of the Center of the Gaussian Velocity Dist',
                              'StdDev of the Turbulent Velocity Dist',
                              'Instrumental Line Broadening', 'Systematic Line Error', 'Systematic Continuum Error',
                              'DRW Long-Term Variability Amplitude', 'DRW Timescale',
                              'Linear Trend Slope', 'Linear Trend Amplitude', 'Linear Trend Timescale']
        
        param_limits = [ '', '', '', r'[0, 90]', r'[0, 90]', '[-0.5, 0.5]', '', '[0, 1]', '', '[0, 1]', '[0.5, 0.5]',
                        '', '', '', '', r'[0, 90]', '', '', '', '', '', '', '', '', '']

        
        values = []
        for i in range(len(names)):
            if i in [0, 11, 12, 13, 14, 16, 20, 21]:
                values.append(val2latex(  self.bp.results['sample'][:,i]/np.log(10)  ))
            elif i == 8:
                mbh_samps = self.bp.results['sample'][:,i]/np.log(10) + 6
                values.append(val2latex( mbh_samps ))
            elif i == 3:
                values.append(val2latex(self.bp.results['sample'][:,i]*180/np.pi ))
            else:
                values.append(val2latex(self.bp.results['sample'][:,i]))

        
        dat = Table([latex_names, units, values, param_names_better, param_limits], names=['Parameter', 'Unit', 'Value', 'Description', 'Bounds'])
        
        
        custom_dict = {'tabletype': 'table*', 'preamble': r'\begin{center}', 'tablefoot': r'\end{center}', 
                       'col_align': '|l|l|c|l|l|', 'header_start': r'\hline', 'header_end': r'\hline',
                       'data_end': r'\hline'}
        
        ascii.write(dat, output=output_fname, Writer=ascii.Latex,
                    latexdict=custom_dict)

        
        return dat
    
    
###############################################################################
###############################################################################
############################################################################### 
    
    
def val2latex(vals, n=2):
    val = np.median(vals)
    val_hi = np.percentile(vals, 84)
    val_lo = np.percentile(vals, 16)
    
    err_hi = val_hi - val
    err_lo = val - val_lo
    
    if n == 0:
        return r'${0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$'.format(val, err_hi, err_lo)
    if n == 1:
        return r'${0:.1f}^{{+{1:.1f}}}_{{-{2:.1f}}}$'.format(val, err_hi, err_lo)        
    if n == 2:
        return r'${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'.format(val, err_hi, err_lo)




def plot_weight_output(lags, lag_dist, 
                       lag_err_lo, lag_err_hi, lag_value, 
                       llim, rlim, peak, 
                       weight_dist, smooth_dist, acf, 
                    #    zoom=False, 
                       ax_tot=None, show=False, output_fname=None):

    #Read general kwargs
    time_unit = 'd'

    #--------------------------------------------------------------------------------

    if ax_tot is None:
        gs = gridspec.GridSpec(1, 1)
            
        sub_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0],
                                                height_ratios=[1, 3], hspace=0)

        ax_bot = plt.subplot(sub_gs[1])
        ax_top = plt.subplot(sub_gs[0], sharex=ax_bot)

        ax_tot = [ax_top, ax_bot]
        
        
    xlabel = r'\tau'


    #Set ACF=0 when ACF<0
        #Left side
    left_inds = np.argwhere( (acf < 0) & (lags < 0) ).T[0]
    if len(left_inds) == 0:
        ind1 = 0
    else:
        ind1 = np.max( left_inds )

        #Right side
    right_inds = np.argwhere( (acf < 0) & (lags > 0) ).T[0]
    if len(right_inds) == 0:
        ind2 = len(acf)-1
    else:
        ind2 = np.min( right_inds )

    acf[:ind1] = 0
    acf[ind2:] = 0


    #Plot original lag distribution
    dbin = lags[1] - lags[0]
    bin0 = np.min(lags)

    bin_edges = []
    bin0 = np.min(lags) - dbin/2

    for j in range(len(lags)+1):
        bin_edges.append( bin0 + j*dbin )

    hist, _ = np.histogram(lag_dist, bins=bin_edges)
    hist = np.array(hist, dtype=float)


    #Set top of plot
    good_ind = np.argwhere( ( lags >= llim ) & ( lags <= rlim ) ).T[0]

    if np.argmax(hist) in good_ind:
        ytop = 1.5*np.max(hist)

    elif np.percentile(hist, 99) > np.max(hist):
        ytop = np.percentile(hist, 99)

    else:
        ytop = 1.5*np.max(hist[good_ind])


    bin1 = ax_tot[1].fill_between(lags, np.zeros_like(hist), hist, step='mid')
    bin2 = ax_tot[1].fill_between(lags, np.zeros_like(hist), weight_dist*hist, step='mid', color='r', alpha=.7)
    ax_tot[1].axvspan( llim, rlim, color='k', alpha=.1 )

    ax_tot[1].set_ylim(0, ytop)


    #Plot ACF
    im3, = ax_tot[0].plot(lags, acf, c='r')


    #Plot weighting function
    im1, = ax_tot[0].plot(lags, weight_dist, c='k')
    ax_tot[0].set_yticks([.5, 1])

    #Plot smoothed distribution
    im2, = ax_tot[0].plot(lags, smooth_dist/np.max(smooth_dist), c='DodgerBlue')
    ax_tot[0].axvspan( llim, rlim, color='k', alpha=.1 )

    #Write lag and error
    peak_str = err2str( lag_value, lag_err_hi, lag_err_lo, dec=2 )
    peak_str = r'$' + r'{}'.format(peak_str) + r'$' + ' ' + time_unit

    xtxt = .05
    ha='left'
    ax_tot[1].text( xtxt, .85, peak_str,
                                ha=ha, transform=ax_tot[1].transAxes,
                                fontsize=15 )

    ax_tot[1].set_xlabel( r'$' + xlabel + r'$ [' + time_unit + ']', fontsize=15 )

    for a in ax_tot:
        a.set_xlim( llim-.1*(rlim-llim), rlim+.1*(rlim-llim) )
        a.tick_params('both', which='major', length=7)
        a.tick_params('both', which='minor', length=3)

    ax_tot[1].tick_params('both', labelsize=11)
    ax_tot[0].tick_params('y', labelsize=11)
    ax_tot[0].tick_params('x', labelsize=0)


    #Put legends for the top and bottom plots
    ax_tot[0].legend( [im1, im2, im3], [r'w($\tau$)', 'Smoothed Dist', 'ACF'],
                            bbox_to_anchor=(1,1.1), fontsize=11, loc='upper left' )

    ax_tot[1].legend( [bin1, bin2], ['Original', 'Weighted'],
                            bbox_to_anchor=(1, 1), fontsize=11, loc='upper left')


    if output_fname is not None:
        plt.savefig( output_fname, dpi=200, bbox_inches='tight' )


    if show:
        plt.show()
    
    if ax_tot is not None:
        return ax_tot
    else:
        return fig, ax_tot
