import numpy as np
import multiprocessing as mp

from astropy.table import Table

from scipy.interpolate import splrep, splev

from pyqsofit.PyQSOFit import QSOFit

import os
import utils
import sys
import glob




####################################################################################
######################### GET FeII-MgII FLUX FOR ALL EPOCHS ########################
####################################################################################


if __name__ == '__main__':
    rmid = int(sys.argv[1])


    dat_dir = '/data3/stone28/2drm/sdssrm/spec/'
    p0_dir = '/data2/yshen/sdssrm/public/prepspec/'
    summary_dir = '/data2/yshen/sdssrm/public/'
    spec_prop, table_arr, ra, dec = utils.get_spec_dat(rmid, dat_dir, p0_dir, summary_dir)


    output_dir_res = '/data3/stone28/2drm/sdssrm/fit_res_mg2/'
    res_dir = output_dir_res + 'rm{:03d}/'.format(rmid)
    
    

def host_job(ind, ra, dec, qsopar_dir, rej_abs_line, nburn, nsamp, nthin, 
             linefit, mask_line, Fe_uv_params=None):

    print('Fitting FeII-MgII contribution for epoch {:03d}'.format(ind+1))
    
    lam = np.array(table_arr[ind]['Wave[vaccum]'])
    flux = np.array(table_arr[ind]['corrected_flux'])
    err = np.array(table_arr[ind]['corrected_err'])
    
    and_mask = np.array(table_arr[ind]['ANDMASK'])
    or_mask = np.array(table_arr[ind]['ORMASK'])
    
    z = spec_prop['z'][ind]
    
    
    mjd = spec_prop['mjd'][ind]
    plateid = spec_prop['plateid'][ind]
    fiberid = spec_prop['fiberid'][ind]

    wave_range = np.array([2200, 3090])
    if mask_line:
        wave_mask = np.array([[2675, 2925]])
    else:
        wave_mask = None

    qi = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                and_mask_in=and_mask, or_mask_in=or_mask)
    
    qi.Fit(name='Object', nsmooth=1, deredden=True, 
            and_mask=True, or_mask=True,
        reject_badpix=False, wave_range=wave_range, wave_mask=wave_mask, 
        decompose_host=False,
        Fe_uv_op=True, poly=False,
        rej_abs_conti=False, rej_abs_line=rej_abs_line,
        MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=linefit, 
        Fe_uv_fix=Fe_uv_params,
        save_result=False, plot_fig=False, save_fig=False, plot_corner=False, 
        save_fits_name=None, save_fits_path=None, verbose=False)

    return qi




def save_feii_params(qi_arr, output_dir):

    epochs = np.array( range(len(qi_arr)) ) + 1
    
    norm = []
    norm_err = []
    fwhm = []
    fwhm_err = []
    shift = []
    shift_err = []

    for i in range(len(epochs)):
        #Fe_uv_model
        pp_tot = qi_arr[i].conti_result[7:].astype(float)
            
        norm.append(pp_tot[0])
        norm_err.append(pp_tot[1])
        
        fwhm.append(pp_tot[2])
        fwhm_err.append(pp_tot[3])
        
        shift.append(pp_tot[4])
        shift_err.append(pp_tot[5])
    
    
    dat = Table( [epochs, shift, shift_err, norm, norm_err, fwhm, fwhm_err],
                names=['Epoch', 'Norm', 'Norm_Err', 'FWHM', 'FWHM_Err', 'Shift', 'Shift_Err'] )
    dat.write( output_dir + 'best_fit_params.dat', format='ascii', overwrite=True )
    
    return



def get_feii_flux(indices, qsopar_dir, nburn, nsamp, nthin,
                 ra, dec, output_dir,
                 rej_abs_line=False, linefit=False, mask_line=False, 
                 Fe_uv_params=None,
                 ncpu=None):

    njob = len(indices)

    arg1 = indices
    arg2 = np.full(njob, ra)
    arg3 = np.full(njob, dec)
    
    arg4 = []
    arg5 = np.full(njob, rej_abs_line, dtype=bool)
    arg6 = np.full(njob, nburn)
    arg7 = np.full(njob, nsamp)
    arg8 = np.full(njob, nthin)
    arg9 = np.full(njob, linefit, dtype=bool)
    arg10 = np.full(njob, mask_line, dtype=bool)
    arg11 = []
    
    for i in arg1:
        arg4.append(qsopar_dir)
        arg11.append(Fe_uv_params)
        
        
    argtot = zip(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11)

    if ncpu is None:
        ncpu = njob

    pool = mp.Pool(ncpu)
    qi_arr = pool.starmap( host_job, argtot )
    pool.close()
    pool.join()    
    
    
    
    
    wl_fe = np.linspace(2200, 3090, 3000)
    
    feii_arrs = []
    cont_arrs = []
    for i in range(njob):
        pp_tot = qi_arr[i].conti_result[7::2].astype(float)
        
        feii_arrs.append( qi_arr[i].Fe_flux_mgii(wl_fe, pp_tot[:3] ) + qi_arr[i].Fe_flux_mgii(wl_fe, pp_tot[3:6] ) )
        cont_arrs.append( qi_arr[i].PL(wl_fe, pp_tot) )
        
    if Fe_uv_params is None:
        save_feii_params(qi_arr, output_dir)
    else:
        resave_feii_params(indices, Fe_uv_params, qi_arr, output_dir)
    
    return wl_fe, feii_arrs, cont_arrs



def save_feii_fluxes(wl_fe, fe2_fluxes, cont_fluxes, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    
    output_fnames = []
    for i in range(len(fe2_fluxes)):
        output_fnames.append( output_dir + 'FeII_fit_epoch{:03d}.dat'.format(i+1) )
    
    
    for i in range(len(fe2_fluxes)):        
        dat = Table( [wl_fe, fe2_fluxes[i], cont_fluxes[i]], names=['RestWavelength', 'FeII_MgII', 'PL_Cont'] )
        dat.write(output_fnames[i], format='ascii', overwrite=True)
    
    return

####################################################################################
############################ REFIT FLUX IF NECESSARY ###############################
####################################################################################

def find_bad_fits(fit_fnames):

    nepoch = len(fit_fnames)

    #Get data
    wl_arrs = []
    fe2_fluxes = []
    cont_fluxes = []

    for i in range(nepoch):
        dat = Table.read(fit_fnames[i], format='ascii')
        wl_arrs.append(dat['RestWavelength'].tolist())
        fe2_fluxes.append(dat['FeII_MgII'].tolist())
        cont_fluxes.append(dat['PL_Cont'].tolist())


    wl_arrs = np.vstack(wl_arrs)
    fe2_fluxes = np.vstack(fe2_fluxes)
    cont_fluxes = np.vstack(cont_fluxes)
    
    mean_flux = np.mean(fe2_fluxes, axis=0)
    rms_flux = np.std(fe2_fluxes, axis=0)
    
    
    
    #Find bad epochs
    bad_mask = np.zeros(nepoch, dtype=bool)
    for i in range(nepoch):
        nbad = len( np.argwhere( np.abs( fe2_fluxes[i] - mean_flux ) > 2*rms_flux ).T[0] )
        bad_mask[i] = nbad > len(fe2_fluxes[i])//2
    
    return bad_mask


def resave_feii_params(indices, Fe_uv_params, qi_arr, output_dir):


    epochs = indices + 1
    
    norm = []
    norm_err = []
    fwhm = []
    fwhm_err = []
    shift = []
    shift_err = []

    for i in range(len(epochs)):
        #Fe_uv_model
        pp_tot = qi_arr[i].conti_result[7:].astype(float)
            
        if Fe_uv_params[0] is None:
            norm.append(pp_tot[0])
            norm_err.append(pp_tot[1])
        else:
            norm.append(Fe_uv_params[0])
            norm_err.append(0.)

        if Fe_uv_params[1] is None:       
            fwhm.append(pp_tot[2])
            fwhm_err.append(pp_tot[3])
        else:
            fwhm.append(Fe_uv_params[1])
            fwhm_err.append(0.)

        if Fe_uv_params[2] is None:
            shift.append(pp_tot[4])
            shift_err.append(pp_tot[5])
        else:
            shift.append(Fe_uv_params[2])
            shift_err.append(0.)
    
    
    
    dat_og = Table.read(output_dir + 'best_fit_params.dat', format='ascii')
    
    for i, ind in enumerate(indices):
        dat_og['Norm'][ind] = norm[i]
        dat_og['Norm_Err'][ind] = norm_err[i]
        
        dat_og['FWHM'][ind] = fwhm[i]
        dat_og['FWHM_Err'][ind] = fwhm_err[i]
        
        dat_og['Shift'][ind] = shift[i]
        dat_og['Shift_Err'][ind] = shift_err[i]       
    

    dat_og.write( output_dir + 'best_fit_params.dat', format='ascii', overwrite=True )

    return



def resave_feii_fluxes(indices, wl_fe, fe2_fluxes, cont_fluxes, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_fnames = []
    for ind in indices:
        output_fnames.append( output_dir + 'FeII_fit_epoch{:03d}.dat'.format(ind+1) )
    
    for i in range(len(fe2_fluxes)):        
        dat = Table( [wl_fe, fe2_fluxes[i], cont_fluxes[i]], names=['RestWavelength', 'FeII_MgII', 'PL_Cont'] )
        dat.write(output_fnames[i], format='ascii', overwrite=True)
    
    return





def refit_bad_epochs(fit_dir, qsopar_dir, nburn, nsamp, nthin, ra, dec,
                     rej_abs_line=False, linefit=False, mask_line=False,
                     ncpu=None):
    
    #Get filenames
    fit_fnames = glob.glob(fit_dir + 'FeII_fit_epoch*.dat')
    nepoch = len(fit_fnames)
    fit_fnames = [ fit_dir + 'FeII_fit_epoch{:03d}.dat'.format(i+1) for i in range(nepoch) ]
        
    bad_mask = find_bad_fits(fit_fnames)
    indices = np.argwhere(bad_mask).T[0]
    
    print('Epochs to refit: ', indices+1)
    
    
    #Get fixed parameters (seems like only FWHM matters)
    param_dat = Table.read(fit_dir + 'best_fit_params.dat', format='ascii')
    
    norm_fix = np.median( param_dat['Norm'][~bad_mask].tolist() )
    fwhm_fix = np.median( param_dat['FWHM'][~bad_mask].tolist() )
    shoft_fix = np.median( param_dat['Shift'][~bad_mask].tolist() )
    fixed_params = [None, fwhm_fix, None]
    
    
    
    #Run fitting again
    wl_fe, fe2_fluxes, cont_fluxes = get_feii_flux(indices, qsopar_dir, nburn, nsamp, nthin,
                                                ra, dec, fit_dir,
                                                rej_abs_line=rej_abs_line, linefit=linefit, mask_line=mask_line, 
                                                Fe_uv_params=fixed_params,
                                                ncpu=ncpu)
    
    resave_feii_fluxes(indices, wl_fe, fe2_fluxes, cont_fluxes, fit_dir)

    return

####################################################################################
###################### SUBTRACT SAVED FeII MgII FLUX ###############################
####################################################################################

def interpolate_fe2_flux(rest_wl, ref_flux_fname, cont=False):
    dat = Table.read(ref_flux_fname, format='ascii')
    ref_wl = np.array(dat['RestWavelength'])
    ref_flux = np.array(dat['FeII_MgII'])
    
    spl = splrep(ref_wl, ref_flux, s=0)
    interp_fe2_flux = splev(rest_wl, spl, der=0)
    
    if cont:
        ref_cont = np.array(dat['PL_Cont'])
        
        spl = splrep(ref_wl, ref_cont, s=0)
        interp_cont = splev(rest_wl, spl, der=0)
        
        return interp_fe2_flux, interp_cont
    
    else:
        return interp_fe2_flux


def remove_fe2_mg2_flux(wl, flux, ref_feii_fname, z=None, cont=False):
    if z is None:
        z = 0.0
    
    rest_wl = wl / (1+z)
    
    if cont:
        interp_fe2_flux, interp_cont_flux = interpolate_fe2_flux(rest_wl, ref_feii_fname, cont=cont)
        return flux - interp_fe2_flux - interp_cont_flux
    
    else:
        interp_fe2_flux = interpolate_fe2_flux(rest_wl, ref_feii_fname)
        return flux - interp_fe2_flux


####################################################################################
####################################### RUN ########################################
####################################################################################

if __name__ == '__main__':
    output_dir = '/data3/stone28/2drm/sdssrm/constants/fe2_mg2/rm{:03d}/'.format(rmid)
    
    wl_fe, feii_fluxes, cont_fluxes = get_feii_flux( range(len(spec_prop)), res_dir,
                                    100, 200, 10,
                                    ra, dec, output_dir, linefit=False,
                                    mask_line=True)
    save_feii_fluxes(wl_fe, feii_fluxes, cont_fluxes, output_dir)
    
    refit_bad_epochs(output_dir, res_dir, 100, 200, 10,
                     ra, dec, linefit=False, mask_line=True)
