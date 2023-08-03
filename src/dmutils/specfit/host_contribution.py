import multiprocessing as mp
import os
from functools import partial

import numpy as np

from astropy.table import Table
from scipy.interpolate import splev, splrep

from pyqsofit.PyQSOFit import QSOFit


#####################################################################################
#####################################################################################
#####################################################################################


def check_rerun(qi, line_name):

    if line_name == 'mg2':
        c = np.argwhere( qi.uniq_linecomp_sort == 'MgII' ).T[0][0]
        chi2_nu1 = float(qi.comp_result[c*7+4])
        
        rerun = chi2_nu1 > 3
        
    elif line_name == 'hb':
        c = np.argwhere( qi.uniq_linecomp_sort == 'H$\\beta$' ).T[0][0]
        chi2_nu1 = float(qi.comp_result[c*7+4])
        
        names = qi.line_result_name
        oiii_mask = (names == 'OIII4959c_1_scale')
        oiii_scale = float(qi.line_result[oiii_mask])

        rerun = (chi2_nu1 > 3) | (oiii_scale < 1)        
        
    elif line_name == 'ha':
        c = np.argwhere( qi.uniq_linecomp_sort == 'H$\\alpha$' ).T[0][0]
        chi2_nu1 = float(qi.comp_result[c*7+4])
        
        rerun = chi2_nu1 > 3
        
    elif line_name is None:
        
        c = np.argwhere( qi.uniq_linecomp_sort == 'MgII' ).T[0][0]
        chi2_nu1 = float(qi.comp_result[c*7+4])
        
        rerun1 = chi2_nu1 > 3
        
        
        
        c = np.argwhere( qi.uniq_linecomp_sort == 'H$\\beta$' ).T[0][0]
        chi2_nu1 = float(qi.comp_result[c*7+4])
        
        names = qi.line_result_name
        oiii_mask = (names == 'OIII4959c_1_scale')
        oiii_scale = float(qi.line_result[oiii_mask])

        rerun2 = (chi2_nu1 > 3) | (oiii_scale < 1) 
        
        
        
        c = np.argwhere( qi.uniq_linecomp_sort == 'H$\\alpha$' ).T[0][0]
        chi2_nu1 = float(qi.comp_result[c*7+4])
        
        rerun3 = chi2_nu1 > 3
        
        
        rerun = rerun1 | rerun2 | rerun3
        
        
    return rerun


#####################################################################################
#####################################################################################
#####################################################################################


def host_job(ind, obj, qsopar_dir, line_name, rej_abs_line, nburn, nsamp, nthin, 
             npca_gal=5, npca_qso=20,
             linefit=False, Fe_uv_params=None, Fe_op_params=None):
    
    print('Fitting host contribution for epoch {:03d}'.format(ind+1))
    
    assert line_name in ['mg2', 'hb', 'ha', None]
    
    lam = np.array(obj.table_arr[ind]['Wave[vaccum]'])
    flux = np.array(obj.table_arr[ind]['corrected_flux'])
    err = np.array(obj.table_arr[ind]['corrected_err'])
    
    and_mask = np.array(obj.table_arr[ind]['ANDMASK'])
    or_mask = np.array(obj.table_arr[ind]['ORMASK'])
    
    mjd = obj.mjd[ind]
    plateid = obj.plateid[ind]
    fiberid = obj.fiberid[ind]

    use_and_mask = True
    use_or_mask = True
    poly = True

    if line_name == 'mg2':
        poly = False
        wave_range = np.array([2200, 3090])

    elif line_name == 'hb':
        poly = False
        wave_range = np.array([4435, 5535])
            
    elif line_name == 'ha':
        poly = False
        wave_range = np.array([6100, 7000])
            
        use_and_mask = False
        use_or_mask = False
        
    elif line_name is None:
        wave_range = None

    
    qi = QSOFit(lam, flux, err, obj.z, ra=obj.ra, dec=obj.dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                and_mask_in=and_mask, or_mask_in=or_mask)
    
    qi.Fit(name='Object', nsmooth=1, deredden=True, 
            and_mask=use_and_mask, or_mask=use_or_mask,
            reject_badpix=False, wave_range=wave_range, wave_mask=None, 
            decompose_host=True, npca_gal=npca_gal, npca_qso=npca_qso, 
            Fe_uv_op=True, poly=poly,
            rej_abs_conti=False, rej_abs_line=rej_abs_line,
            MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=linefit, 
            Fe_uv_fix=Fe_uv_params, Fe_op_fix=Fe_op_params,
            save_result=False, plot_fig=False, save_fig=False, plot_corner=False, 
            save_fits_name=None, save_fits_path=None, verbose=False)
    
    
    if linefit: 
        rerun = check_rerun(qi, line_name)
        
        n = 0
        while rerun:
            
            qi = QSOFit(lam, flux, err, obj.z, ra=obj.ra, dec=obj.dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                        and_mask_in=and_mask, or_mask_in=or_mask)
            
            qi.Fit(name='Object', nsmooth=1, deredden=True, 
                    and_mask=use_and_mask, or_mask=use_or_mask,
                    reject_badpix=False, wave_range=wave_range, wave_mask=None, 
                    decompose_host=True, npca_gal=npca_gal, npca_qso=npca_qso, 
                    Fe_uv_op=True, poly=poly,
                    rej_abs_conti=False, rej_abs_line=rej_abs_line,
                    MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=linefit, 
                    Fe_uv_fix=Fe_uv_params, Fe_op_fix=Fe_op_params,
                    save_result=False, plot_fig=False, save_fig=False, plot_corner=False, 
                    save_fits_name=None, save_fits_path=None, verbose=False)
            
            
            rerun = check_rerun(qi, line_name)

            if n > 5:
                break

            n += 1

    return qi.wave, qi.host





def get_host_flux(obj, indices, qsopar_dir, line_name, nburn, nsamp, nthin,
                 rej_abs_line=False, linefit=False, 
                 Fe_uv_params=None, Fe_op_params=None, 
                 ncpu=None):

    njob = len(indices)
    new_host_job = partial(host_job,
                           obj=obj, qsopar_dir=qsopar_dir, line_name=line_name,
                           rej_abs_line=rej_abs_line, 
                           nburn=nburn, nsamp=nsamp, nthin=nthin,
                           linefit=linefit, Fe_uv_params=Fe_uv_params, Fe_op_params=Fe_op_params)


    if ncpu is None:
        ncpu = njob

    pool = mp.Pool(ncpu)
    res = pool.map( new_host_job, indices )
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


def get_best_host_flux(obj, output_dir, line_name=None):
    
    if line_name == 'ha':
        bounds = [6100, 7000]
    elif line_name == 'hb':
        bounds = [4435, 5535]
    elif line_name == 'mg2':
        bounds = [2200, 3090]
    elif line_name is None:
        bounds = [0, np.inf]
    
    snr = []
    for i in range(obj.nepoch):
        dat = obj.table_arr[i].copy()
    
        mask = (dat['Wave[vaccum]']/(1+obj.z) > bounds[0]) & (dat['Wave[vaccum]']/(1+obj.z) < bounds[1])
        snr_i = dat['corrected_flux'][mask]/dat['corrected_err'][mask]
        
        snr.append(np.median(snr_i))


    best_epoch = np.argmax(snr)+1
    
    
    #Resave the best SNR host flux
    best_dat = Table.read(output_dir + 'host_flux_epoch{:03d}.dat'.format(best_epoch), format='ascii')
    best_dat.write(output_dir + 'best_host_flux.dat', format='ascii', overwrite=True)
    
    return best_epoch, snr


#####################################################################################
#####################################################################################
#####################################################################################


def interpolate_host_flux(rest_wl, host_flux_fname):
    dat = Table.read(host_flux_fname, format='ascii')
    host_wl = np.array(dat['RestWavelength'])
    host_flux = np.array(dat['HostFlux'])
    
    
    min_wl = np.min(host_wl)
    max_wl = np.max(host_wl)
    mask = (rest_wl >= min_wl) & (rest_wl <= max_wl)
        
    spl = splrep(host_wl, host_flux, s=0)
    interp_host_flux = splev(rest_wl, spl, der=0)
    
    return interp_host_flux, mask


def remove_host_flux(wl, flux, err, and_mask, or_mask, host_flux_fname, z=None):
    if z is None:
        z = 0.0
    
    rest_wl = wl / (1+z)
    interp_host_flux, mask = interpolate_host_flux(rest_wl, host_flux_fname)
    
    return flux[mask] - interp_host_flux, wl[mask], err[mask], and_mask[mask], or_mask[mask]

