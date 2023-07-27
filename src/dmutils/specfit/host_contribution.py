import numpy as np

from astropy.table import Table
from astropy.io import fits

from scipy.interpolate import interp1d

from pyqsofit.PyQSOFit import QSOFit

import multiprocessing as mp
import os
import utils
import sys




####################################################################################
######################### GET HOST FLUX FOR ALL EPOCHS #############################
####################################################################################

if __name__ == '__main__':
    rmid = int(sys.argv[1])

    dat_dir = '/data3/stone28/2drm/sdssrm/spec/'
    p0_dir = '/data2/yshen/sdssrm/public/prepspec/'
    summary_dir = '/data2/yshen/sdssrm/public/'
    spec_prop, table_arr, ra, dec = utils.get_spec_dat(rmid, dat_dir, p0_dir, summary_dir)


    output_dir_res = '/data3/stone28/2drm/sdssrm/fit_res_mg2/'
    res_dir = output_dir_res + 'rm{:03d}/'.format(rmid)



def host_job(ind, ra, dec, qsopar_dir, rej_abs_line, nburn, nsamp, nthin, linefit):

    print('Fitting host contribution for epoch {:03d}'.format(ind+1))
    
    lam = np.array(table_arr[ind]['Wave[vaccum]'])
    flux = np.array(table_arr[ind]['corrected_flux'])
    err = np.array(table_arr[ind]['corrected_err'])
    
    and_mask = np.array(table_arr[ind]['ANDMASK'])
    or_mask = np.array(table_arr[ind]['ORMASK'])
    
    z = spec_prop['z'][ind]
    
    
    mjd = spec_prop['mjd'][ind]
    plateid = spec_prop['plateid'][ind]
    fiberid = spec_prop['fiberid'][ind]

    
    qi = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                and_mask_in=and_mask, or_mask_in=or_mask)
    
    qi.Fit(name='Object', nsmooth=1, deredden=True, 
            and_mask=True, or_mask=True,
        reject_badpix=False, wave_range=None, wave_mask=None, 
        decompose_host=True, npca_gal=5, npca_qso=20, 
        Fe_uv_op=True, poly=True,
        rej_abs_conti=False, rej_abs_line=rej_abs_line,
        MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=linefit, 
        save_result=False, plot_fig=False, save_fig=False, plot_corner=False, 
        save_fits_name=None, save_fits_path=None, verbose=False)

    return qi.wave, qi.host
    

def get_host_flux(njob, qsopar_dir, nburn, nsamp, nthin,
                 ra, dec,
                 rej_abs_line=False, linefit=False, ncpu=None):

    arg1 = np.array( range(njob) )
    arg2 = np.full(njob, ra)
    arg3 = np.full(njob, dec)
    
    arg4 = []
    arg5 = np.full(njob, rej_abs_line, dtype=bool)
    arg6 = np.full(njob, nburn)
    arg7 = np.full(njob, nsamp)
    arg8 = np.full(njob, nthin)
    arg9 = np.full(njob, linefit, dtype=bool)
    
    for i in arg1:
        arg4.append(qsopar_dir)
        
        
    argtot = zip(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)

    if ncpu is None:
        ncpu = njob

    pool = mp.Pool(ncpu)
    res = pool.starmap( host_job, argtot )
    pool.close()
    pool.join()    
    
    
    host_fluxes = []
    wl_arrs = []
    
    for i in range(len(res)):
        wl_arrs.append(res[i][0])
        host_fluxes.append(res[i][1])
    
    return wl_arrs, host_fluxes


def save_host_fluxes(wl_arrs, host_fluxes, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    
    output_fnames = []
    for i in range(len(wl_arrs)):
        output_fnames.append( output_dir + 'host_flux_epoch{:03d}.dat'.format(i+1) )
    
    
    for i in range(len(wl_arrs)):        
        dat = Table( [wl_arrs[i], host_fluxes[i]], names=['RestWavelength', 'HostFlux'] )
        dat.write(output_fnames[i], format='ascii', overwrite=True)
    
    return



####################################################################################
########################### SUBTRACT SAVED HOST FLUX ###############################
####################################################################################

def interpolate_host_flux(rest_wl, host_flux_fname, kind='linear'):
    dat = Table.read(host_flux_fname, format='ascii')
    host_wl = np.array(dat['RestWavelength'])
    host_flux = np.array(dat['HostFlux'])
    
    
    min_wl = np.min(host_wl)
    max_wl = np.max(host_wl)
    mask = (rest_wl >= min_wl) & (rest_wl <= max_wl)
    
    func = interp1d(host_wl, host_flux, kind=kind)
    interp_host_flux = func(rest_wl[mask])
    
    return interp_host_flux, mask


def remove_host_flux(wl, flux, err, and_mask, or_mask, host_flux_fname, z=None):
    if z is None:
        z = 0.0
    
    rest_wl = wl / (1+z)
    interp_host_flux, mask = interpolate_host_flux(rest_wl, host_flux_fname)
    
    return flux[mask] - interp_host_flux, wl[mask], err[mask], and_mask[mask], or_mask[mask]

    
####################################################################################
####################################### RUN ########################################
####################################################################################

if __name__ == '__main__':
    #Get host flux
    wl_arrs, host_fluxes = get_host_flux( len(spec_prop), res_dir,
                                    100, 200, 10,
                                    ra, dec)
    save_host_fluxes(wl_arrs, host_fluxes, '/data3/stone28/2drm/sdssrm/host_fluxes/rm{:03d}/'.format(rmid))