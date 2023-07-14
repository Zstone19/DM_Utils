import numpy as np
import multiprocessing as mp

from astropy.table import Table

from scipy.interpolate import splrep, splev

from pyqsofit.PyQSOFit import QSOFit
from host_contribution import remove_host_flux

import os
import utils
import sys



####################################################################################
########################### GET FeII-Hb FLUX FOR ALL EPOCHS ########################
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

    print('Fitting FeII-Hbeta contribution for epoch {:03d}'.format(ind+1))
    
    lam = np.array(table_arr[ind]['Wave[vaccum]'])
    flux = np.array(table_arr[ind]['corrected_flux'])
    err = np.array(table_arr[ind]['corrected_err'])
    
    and_mask = np.array(table_arr[ind]['ANDMASK'])
    or_mask = np.array(table_arr[ind]['ORMASK'])
    
    z = spec_prop['z'][ind]
    
    
    mjd = spec_prop['mjd'][ind]
    plateid = spec_prop['plateid'][ind]
    fiberid = spec_prop['fiberid'][ind]

    wave_range = np.array([4500, 5500])
    flux, lam, err, and_mask, or_mask = remove_host_flux(lam, flux, err, and_mask, or_mask,
                                                         '/data3/stone28/2drm/sdssrm/constants/host_fluxes/rm{:03d}/best_host_flux.dat'.format(rmid), 
                                                         z=z)

    qi = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                and_mask_in=and_mask, or_mask_in=or_mask)
    
    qi.Fit(name='Object', nsmooth=1, deredden=True, 
            and_mask=True, or_mask=True,
        reject_badpix=False, wave_range=wave_range, wave_mask=None, 
        decompose_host=False, npca_gal=5, npca_qso=20, 
        Fe_uv_op=True, poly=True,
        rej_abs_conti=False, rej_abs_line=rej_abs_line,
        MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=linefit, 
        save_result=False, plot_fig=False, save_fig=False, plot_corner=False, 
        save_fits_name=None, save_fits_path=None, verbose=False)


    c = np.argwhere( qi.uniq_linecomp_sort == 'H$\\beta$' ).T[0][0]
    chi2_nu1 = float(qi.comp_result[c*7+4])
    
    names = qi.line_result_name
    oiii_mask = (names == 'OIII4959c_1_scale')
    oiii_scale = float(qi.line_result[oiii_mask])


    n = 0
    while (chi2_nu1 > 3) or (oiii_scale < 1):
        
        qi = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                    and_mask_in=and_mask, or_mask_in=or_mask)
        
        qi.Fit(name='Object', nsmooth=1, deredden=True, 
                and_mask=True, or_mask=True,
            reject_badpix=False, wave_range=wave_range, wave_mask=None, 
            decompose_host=False, npca_gal=5, npca_qso=20, 
            Fe_uv_op=True, poly=True,
            rej_abs_conti=False, rej_abs_line=rej_abs_line,
            MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=linefit, 
            save_result=False, plot_fig=False, save_fig=False, plot_corner=False, 
            save_fits_name=None, save_fits_path=None, verbose=False)


        c = np.argwhere( qi.uniq_linecomp_sort == 'H$\\beta$' ).T[0][0]
        chi2_nu1 = float(qi.comp_result[c*7+4])
        
        names = qi.line_result_name
        oiii_mask = (names == 'OIII4959c_1_scale')
        oiii_scale = float(qi.line_result[oiii_mask])

        if n > 5:
            break

        n += 1

    return qi






def get_feii_flux(njob, qsopar_dir, nburn, nsamp, nthin,
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
    qi_arr = pool.starmap( host_job, argtot )
    pool.close()
    pool.join()    
    
    
    wl_arrs = []
    feii_arrs = []
    for i in range(njob):
        wl_arrs.append(qi_arr[i].wave)
        feii_arrs.append( qi_arr[i].f_fe_mgii_model + qi_arr[i].f_fe_balmer_model )
    
    
    return wl_arrs, feii_arrs





def save_feii_fluxes(wl_arrs, host_fluxes, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    
    output_fnames = []
    for i in range(len(wl_arrs)):
        output_fnames.append( output_dir + 'FeII_fit_epoch{:03d}.dat'.format(i+1) )
    
    
    for i in range(len(wl_arrs)):        
        dat = Table( [wl_arrs[i], host_fluxes[i]], names=['RestWavelength', 'FeII_Hbeta'] )
        dat.write(output_fnames[i], format='ascii', overwrite=True)
    
    return





####################################################################################
###################### SUBTRACT SAVED FeII Hbeta FLUX ##############################
####################################################################################

def interpolate_fe2_flux(rest_wl, host_flux_fname, kind='linear'):
    dat = Table.read(host_flux_fname, format='ascii')
    ref_wl = np.array(dat['RestWavelength'])
    ref_flux = np.array(dat['FeII_Hbeta'])
    
    spl = splrep(ref_wl, ref_flux, s=0)
    interp_host_flux = splev(rest_wl, spl, der=0)
    
    return interp_host_flux


def remove_fe2_hb_flux(wl, flux, ref_feii_fname, z=None):
    if z is None:
        z = 0.0
    
    rest_wl = wl / (1+z)
    interp_host_flux = interpolate_fe2_flux(rest_wl, ref_feii_fname)
    
    return flux - interp_host_flux


####################################################################################
####################################### RUN ########################################
####################################################################################

if __name__ == '__main__':
    wl_arrs, feii_fluxes = get_feii_flux( len(spec_prop), res_dir,
                                    100, 200, 10,
                                    ra, dec, linefit=True)
    save_feii_fluxes(wl_arrs, feii_fluxes, '/data3/stone28/2drm/sdssrm/constants/fe2_hb/rm{:03d}/'.format(rmid))
