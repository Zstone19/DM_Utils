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
             linefit, mask_line, Fe_uv_params=None, Fe_uv_range=None):

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
        Fe_uv_fix=Fe_uv_params, Fe_uv_range=Fe_uv_range,
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
                 Fe_uv_params=None, Fe_uv_range=None,
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
    arg12 = []
    
    for i in arg1:
        arg4.append(qsopar_dir)
        arg11.append(Fe_uv_params)
        arg12.append(Fe_uv_range)
        
        
    argtot = zip(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12)

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
        
    if (Fe_uv_params is None) and (Fe_uv_range is None):
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

def find_bad_fits(fit_fnames, param_fname, method='prof', nsig=3):


    ##########################################################
    #METHOD 1

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
    bad_mask1 = np.zeros(nepoch, dtype=bool)
    for i in range(nepoch):
        nbad = len( np.argwhere( np.abs( fe2_fluxes[i] - mean_flux ) > 2*rms_flux ).T[0] )
        bad_mask1[i] = nbad > len(fe2_fluxes[i])//2
        

    ##########################################################
    #METHOD 2
    
    if nsig == 1:
        plo = 16
        phi = 84
    elif nsig == 2:
        plo = 2.5
        phi = 97.5
    elif nsig == 3:
        plo = 0.15
        phi = 99.85
    elif nsig == 4:
        plo = 0.02
        phi = 99.98
    elif nsig == 5:
        plo = 0.003
        phi = 99.997
        
        
    
    param_dat = Table.read(param_fname, format='ascii')
    assert len(param_dat) == nepoch
    
    cols = ['Norm', 'FWHM', 'Shift']
    bad_mask2 = np.zeros(nepoch, dtype=bool)
    
    for col in cols:
        cond1 = param_dat[col] < np.percentile(param_dat[col], plo)
        cond2 = param_dat[col] > np.percentile(param_dat[col], phi)
        
        bad_mask2 |= (cond1 | cond2)
        # bad_mask2 |= np.abs(param_dat[col] - np.median(param_dat[col])) > nsig*np.std(param_dat[col])
    
    ##########################################################
    #OUTPUT
    
    if method == 'prof':
        bad_mask = bad_mask1
    elif method == 'param':
        bad_mask = bad_mask2
    elif method == 'both':
        bad_mask = bad_mask1 | bad_mask2
        
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
                     fix=None, ranges=None, all=False, method='both', nsig=3,
                     rej_abs_line=False, linefit=False, mask_line=False,
                     ncpu=None):
    
    
    #Get filenames
    nepoch = len( glob.glob(fit_dir + 'FeII_fit_epoch*.dat') )
    fit_fnames = [ fit_dir + 'FeII_fit_epoch{:03d}.dat'.format(i+1) for i in range(nepoch) ]
    param_fname = fit_dir + 'best_fit_params.dat'
    
    bad_mask = find_bad_fits(fit_fnames, param_fname, method=method, nsig=nsig)

    if all is True:
        indices = np.array( range(nepoch) )
        print('Refitting all epochs')
    elif all is False:
        indices = np.argwhere(bad_mask).T[0]
        print('Epochs to refit: ', indices+1)
    else:
        indices = np.array(all)
        print('Epochs to refit: ', indices+1)
    
    
    #Get fixed parameters (seems like only FWHM matters)
    param_dat = Table.read(fit_dir + 'best_fit_params.dat', format='ascii')
    norm_lo, norm_fix, norm_hi = np.percentile( param_dat['Norm'][~bad_mask].tolist(), [16,50,84] )
    fwhm_lo, fwhm_fix, fwhm_hi = np.percentile( param_dat['FWHM'][~bad_mask].tolist(), [16,50,84] )
    shift_lo, shift_fix, shift_hi = np.percentile( param_dat['Shift'][~bad_mask].tolist(), [16,50,84] )


    if fix is not None:
        fixed_params = [None, None, None]  
        if 'norm' in fix:
            fixed_params[0] = norm_fix
        if 'fwhm' in fix:
            fixed_params[1] = fwhm_fix
        if 'shift' in fix:
            fixed_params[2] = shift_fix
    else:
        fixed_params = None
        
    if ranges is not None:
        range_params = [None, None, None]
        if 'norm' in ranges:
            range_params[0] = [norm_lo, norm_hi]
        if 'fwhm' in ranges:
            range_params[1] = [fwhm_lo, fwhm_hi]
        if 'shift' in ranges:
            range_params[2] = [shift_lo, shift_hi]   
    else:
        range_params = None
    
    #Run fitting again
    wl_fe, fe2_fluxes, cont_fluxes = get_feii_flux(indices, qsopar_dir, nburn, nsamp, nthin,
                                                ra, dec, fit_dir,
                                                rej_abs_line=rej_abs_line, linefit=linefit, mask_line=mask_line, 
                                                Fe_uv_params=fixed_params, Fe_uv_range=range_params,
                                                ncpu=ncpu)

    resave_feii_fluxes(indices, wl_fe, fe2_fluxes, cont_fluxes, fit_dir)

    return bad_mask




def iterate_refitting(fit_dir, qsopar_dir, nburn, nsamp, nthin, ra, dec,
                     fix=None, ranges=None, all=False, method='both',
                     rej_abs_line=False, linefit=False, mask_line=False,
                     ncpu=None, niter=2):


    nepoch = len( glob.glob(fit_dir + 'FeII_fit_epoch*.dat') )
    fit_fnames = [ fit_dir + 'FeII_fit_epoch{:03d}.dat'.format(i+1) for i in range(nepoch) ]
    param_fname = fit_dir + 'best_fit_params.dat'




    nsig_arr = np.full(niter, 3, dtype=int)

    #First iteration
    masks_tot = refit_bad_epochs(fit_dir, qsopar_dir, nburn, nsamp, nthin, ra, dec,
                                fix=fix, ranges=ranges, all=all, method=method, nsig=nsig_arr[0],
                                rej_abs_line=rej_abs_line, linefit=linefit, mask_line=mask_line,
                                ncpu=ncpu)
    
    refit_epochs = np.argwhere(masks_tot).T[0] + 1
    refit_iter = np.ones(len(refit_epochs), dtype=int)
    nsig_tot = np.zeros(len(refit_epochs), dtype=int)
    


    
    if niter == 1:
        return

#    fix_new = fix
        
    if all is True:
        niter = 2
        
        if fix == ['fwhm']:
            fix_arr = [['shift', 'fwhm']] 
            fix_arr = [fix]    
                
        if fix == ['shift']:
            fix_arr = [['shift', 'fwhm']]
            fix_arr = [fix]
    
    for i in range(niter - 1):
        mask_i = refit_bad_epochs(fit_dir, qsopar_dir, nburn, nsamp, nthin, ra, dec,
                                fix=fix_arr[i], ranges=ranges, all=False, method=method, nsig=nsig_arr[i+1],
                                rej_abs_line=rej_abs_line, linefit=linefit, mask_line=mask_line,
                                ncpu=ncpu)
        
        refit_epochs_i = np.argwhere(mask_i).T[0] + 1
        refit_iter_i = np.full(len(refit_epochs_i), i+2)
        nsig_tot_i = np.full(len(refit_epochs_i), nsig_arr[i+1])
    
        refit_epochs = np.concatenate( [refit_epochs, refit_epochs_i] )
        refit_iter = np.concatenate( [refit_iter, refit_iter_i] )
        nsig_tot = np.concatenate( [nsig_tot, nsig_tot_i] )
        
        
        
        
    #Find out what epochs were refit more than once
    unique_epochs = np.unique(refit_epochs)
    bad_indices = []
    for epoch in unique_epochs:
        nrefit = len( np.argwhere(refit_epochs == epoch).T[0] )
        
        if nrefit > 1:
            bad_indices.append(epoch-1)
            
    
    #Find out what epochs are still bad
    bad_mask = find_bad_fits(fit_fnames, param_fname, method=method, nsig=nsig_arr[-1])
    bad_ind_still = np.argwhere(bad_mask).T[0]
    for ind in bad_ind_still:
        if ind not in bad_indices:
            bad_indices.append(ind)
    
            
            
    #Force fit these epochs
    if len(bad_indices) > 0:
        print('Fixing epochs:', np.array(bad_indices)+1)
        refit_bad_epochs(fit_dir, qsopar_dir, nburn, nsamp, nthin, ra, dec,
                        fix=['norm', 'fwhm', 'shift'], ranges=ranges, all=bad_indices, method=method, nsig=1,
                        rej_abs_line=rej_abs_line, linefit=linefit, mask_line=mask_line,
                        ncpu=ncpu)
        
    
    #Save refit data
    with open(fit_dir + 'refit_data.dat', '+a') as f:
        f.write('Iteration Nsig Epoch Fixed\n')
        
        for i in range(len(refit_epochs)):
            f.write('{} {} {} {}\n'.format(refit_iter[i], nsig_tot[i], refit_epochs[i], refit_epochs[i]-1 in bad_indices))

        
    return masks_tot, refit_epochs, refit_iter
    
    

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

    iterate_refitting(output_dir, res_dir, 100, 200, 10,
                        ra, dec,
                        all=True, fix=['fwhm'], method='both',
                        linefit=False, mask_line=True, niter=3)


#To get the results I have now for MgII, I did the following:
#    - Ran throigh all of the objects with an initial FeII fit
#    - Ran through all of the objects again, fixing the FWHM to the median of the initial fits
#    - Fixed fits on the problematic objects with the following procedure:
#        - An epoch is considered "bad" if:
#            - More than half of the pixels on the profile deviate from the median profile by more than 2sigma
#              OR
#            - The fitted FWHM, shift, or norm is more than 3sigma from the median FWHM, shift, or norm
#        - For the bad objects, fix its FWHM, shift, and norm to the median values
    