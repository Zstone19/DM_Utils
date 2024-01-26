import numpy as np
from scipy.interpolate import splrep, splev
from scipy.special import erf
from astropy.table import Table

import multiprocessing as mp
import os
import glob
from functools import partial

from pyqsofit.PyQSOFit import QSOFit
from dmutils.specfit.host_contribution import remove_host_flux
from dmutils.specfit.qsopar import make_qsopar


from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)



##########################################################################################
##################################### Utility Functions ##################################
##########################################################################################

def save_feii_params(qi_arr, output_dir, line_name):

    epochs = np.array( range(len(qi_arr)) ) + 1
    
    
    ##########################################
    # FeII
    
    if line_name in ['mg2', 'c4']:
        tot_params = np.zeros( (len(epochs), 3*2) )

        for i in range(len(epochs)):
            #Fe_uv_model
            pp_tot = qi_arr[i].conti_result[7:].astype(float)
                
            for j in range(3):
                tot_params[i, 2*j] = pp_tot[2*j]
                tot_params[i, 2*j + 1] = pp_tot[2*j + 1]        
        
        colnames = ['Norm', 'Norm_Err', 'FWHM', 'FWHM_Err', 'Shift', 'Shift_Err']
        
        
    elif line_name in ['hb', 'ha']:        
        tot_params = np.zeros( (len(epochs), 6*2) )

        for i in range(len(epochs)):
            #Fe_uv_model and Fe_op_model
            pp_tot = qi_arr[i].conti_result[7:].astype(float)
                
            for j in range(6):
                tot_params[i, 2*j] = pp_tot[2*j]
                tot_params[i, 2*j + 1] = pp_tot[2*j + 1]                        
        
        
        colnames = ['Norm_uv', 'Norm_uv_Err', 'FWHM_uv', 'FWHM_uv_Err', 'Shift_uv', 'Shift_uv_Err',
                    'Norm_op', 'Norm_op_Err', 'FWHM_op', 'FWHM_op_Err', 'Shift_op', 'Shift_op_Err']
        


    dat = Table( [epochs], names=['Epoch'] )
    for i in range(len(colnames)):
        dat[colnames[i]] = tot_params[:, i]

    dat.write( output_dir + 'best_fit_params.dat', format='ascii', overwrite=True )
    
    
    
    ##########################################
    # Power Law Continuum
        
    colnames = ['PL_norm', 'PL_norm_err', 'PL_slope', 'PL_slope_err']
    tot_params = np.zeros( (len(epochs), len(colnames)) )
    for i in range(len(epochs)):
        pp_tot = qi_arr[i].conti_result.astype(float)

        for j in range(len(colnames)):
            tot_params[i, j] = pp_tot[ qi_arr[i].conti_result_name == colnames[j]]
    
    
    
    dat = Table( [epochs], names=['Epoch'] )
    for i in range(len(colnames)):
        dat[colnames[i]] = tot_params[:, i]
        
    dat.write( output_dir + 'best_fit_params_cont.dat', format='ascii', overwrite=True )
    
    
    return



def resave_feii_params(indices, Fe_uv_params, Fe_op_params, qi_arr, output_dir, line_name):
    
    epochs = indices + 1
    
    ##########################################
    # FeII
    
    if line_name in ['mg2', 'c4']:
        tot_params = np.zeros( (len(epochs), 3*2) )

        for i in range(len(epochs)):
            #Fe_uv_model
            pp_tot = qi_arr[i].conti_result[7:].astype(float)
                
            for j in range(3):
                if Fe_uv_params[j] is None:
                    tot_params[i, 2*j] = pp_tot[2*j]
                    tot_params[i, 2*j + 1] = pp_tot[2*j + 1]   
                else:
                    tot_params[i, 2*j] = Fe_uv_params[j]
                    tot_params[i, 2*j + 1] = 0.     
        
        
        colnames = ['Norm', 'Norm_Err', 'FWHM', 'FWHM_Err', 'Shift', 'Shift_Err']

        
    elif line_name in ['hb', 'ha']:
        tot_params = np.zeros( (len(epochs), 3*2*2) )

        for i in range(len(epochs)):
            #Fe_uv_model and Fe_op_model
            pp_tot = qi_arr[i].conti_result[7:].astype(float)
                
            for j in range(3):
                if Fe_uv_params[j] is None:
                    tot_params[i, 2*j] = pp_tot[2*j]
                    tot_params[i, 2*j + 1] = pp_tot[2*j + 1]
                else:
                    tot_params[i, 2*j] = Fe_uv_params[j]
                    tot_params[i, 2*j + 1] = 0.
                    
                if Fe_op_params[j] is None:
                    tot_params[i, 2*j + 6] = pp_tot[2*j + 6]
                    tot_params[i, 2*j + 7] = pp_tot[2*j + 7]
                else:
                    tot_params[i, 2*j + 6] = Fe_op_params[j]
                    tot_params[i, 2*j + 7] = 0.
                        
        
        colnames = ['Norm_uv', 'Norm_uv_Err', 'FWHM_uv', 'FWHM_uv_Err', 'Shift_uv', 'Shift_uv_Err',
                    'Norm_op', 'Norm_op_Err', 'FWHM_op', 'FWHM_op_Err', 'Shift_op', 'Shift_op_Err']
        
        
    dat_og = Table.read(output_dir + 'best_fit_params.dat', format='ascii')
    
    for i, ind in enumerate(indices):
        for j, col in enumerate(colnames):
            dat_og[col][ind] = tot_params[i, j]

    dat_og.write( output_dir + 'best_fit_params.dat', format='ascii', overwrite=True )
    
    
    
    ##########################################
    # Power Law Continuum
    
    colnames = ['PL_norm', 'PL_norm_err', 'PL_slope', 'PL_slope_err']        
    tot_params = np.zeros( (len(epochs), len(colnames)) )
    for i in range(len(epochs)):
        pp_tot = qi_arr[i].conti_result.astype(float)

        for j in range(len(colnames)):
            tot_params[i, j] = pp_tot[ qi_arr[i].conti_result_name == colnames[j]]

            
    
    dat_og = Table.read(output_dir + 'best_fit_params_cont.dat', format='ascii')
    
    for i, ind in enumerate(indices):
        for j, col in enumerate(colnames):
            dat_og[col][ind] = tot_params[i, j]
            
    dat_og.write( output_dir + 'best_fit_params_cont.dat', format='ascii', overwrite=True )

    return


def save_feii_fluxes(wl_fe, fe2_fluxes, cont_fluxes, output_dir, line_name):

    os.makedirs(output_dir, exist_ok=True)
    
    output_fnames = []
    for i in range(len(fe2_fluxes)):
        output_fnames.append( output_dir + 'FeII_fit_epoch{:03d}.dat'.format(i+1) )
    
    if line_name == 'mg2':
        colname = 'FeII_MgII'
    elif line_name == 'c4':
        colname = 'FeII_CIV'
    elif line_name == 'hb':
        colname = 'FeII_Hbeta'
    elif line_name == 'ha':
        colname = 'FeII_Halpha'
    
    for i in range(len(fe2_fluxes)):        
        dat = Table( [wl_fe, fe2_fluxes[i], cont_fluxes[i]], names=['RestWavelength', colname, 'PL_Cont'] )
        dat.write(output_fnames[i], format='ascii', overwrite=True)
    
    return


##########################################################################################
################################### The First Run-Through ################################
##########################################################################################


def check_rerun(qi, line_name):

    if line_name == 'mg2':
        c = np.argwhere( qi.uniq_linecomp_sort == 'MgII' ).T[0][0]
        chi2_nu1 = float(qi.comp_result[c*7+4])
        
        rerun = chi2_nu1 > 3
    elif line_name == 'c4':
        c = np.argwhere( qi.uniq_linecomp_sort == 'CIV' ).T[0][0]
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
        
        
    return rerun
        
        


def host_job(ind, obj, qsopar_dir, line_name,
             rej_abs_line, 
             nburn, nsamp, nthin,
             host_dir=None,
             linefit=False, mask_line=True, 
             Fe_uv_params=None, Fe_uv_range=None,
             Fe_op_params=None, Fe_op_range=None):

    print('Fitting FeII-{} contribution for epoch {:03d}'.format(line_name, ind+1))
    assert line_name in ['mg2', 'c4', 'hb', 'ha']
    
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

    if line_name == 'mg2':
        wave_range = np.array([2200, 3090])
        if mask_line:
            wave_mask = np.array([[2675, 2925]])
        else:
            wave_mask = None
            
    elif line_name == 'c4':
        wave_range = np.array([1445, 1705])
        if mask_line:
            wave_mask = np.array([[1465, 1700]])
        else:
            wave_mask = None
            
        use_and_mask = False
        use_or_mask = False


    elif line_name == 'hb':
        wave_range = np.array([4435, 5535])
        if mask_line:
            wave_mask = np.array([[4700, 5100]])
        else:
            wave_mask = None
            
        assert host_dir is not None, 'Must provide host_dir for H-beta'
            
    elif line_name == 'ha':
        wave_range = np.array([6100, 7000])
        if mask_line:
            wave_mask = np.array([[6400, 6800]])
        else:
            wave_mask = None
            
        assert host_dir is not None, 'Must provide host_dir for H-alpha'
        use_and_mask = False
        use_or_mask = False


    if line_name in ['ha', 'hb']:
        flux, lam, err, and_mask, or_mask = remove_host_flux(lam, flux, err, and_mask, or_mask,
                                                            host_dir + 'best_host_flux.dat', 
                                                            z=obj.z)

    
    name = 'RM{:03d}e{:03d}'.format(obj.rmid, ind+1)
    res_dir = obj.main_dir + line_name + '/fe2/'
    try:
        qi = QSOFit(lam, flux, err, obj.z, ra=obj.ra, dec=obj.dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                    and_mask_in=and_mask, or_mask_in=or_mask)    
    
        with suppress_stdout_stderr():
            qi.Fit(name=name, nsmooth=1, deredden=True, 
                    and_mask=use_and_mask, or_mask=use_or_mask,
                    reject_badpix=False, wave_range=wave_range, wave_mask=wave_mask, 
                    decompose_host=False,
                    Fe_uv_op=True, poly=False,
                    rej_abs_conti=False, rej_abs_line=rej_abs_line,
                    MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=linefit, 
                    Fe_uv_fix=Fe_uv_params, Fe_uv_range=Fe_uv_range,
                    Fe_op_fix=Fe_op_params, Fe_op_range=Fe_op_range,
                    save_result=False, plot_fig=False, save_fig=False, plot_corner=False, 
                    save_fits_name=None, save_fits_path=None, verbose=False,
                    kwargs_conti_emcee={'progress':False}, kwargs_line_emcee={'progress':False})
    except Exception:
        use_and_mask = False
        use_or_mask = False
        
        qi = QSOFit(lam, flux, err, obj.z, ra=obj.ra, dec=obj.dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                and_mask_in=and_mask, or_mask_in=or_mask)

        with suppress_stdout_stderr():
            qi.Fit(name=name, nsmooth=1, deredden=True, 
                    and_mask=use_and_mask, or_mask=use_or_mask,
                    reject_badpix=False, wave_range=wave_range, wave_mask=wave_mask, 
                    decompose_host=False,
                    Fe_uv_op=True, poly=False,
                    rej_abs_conti=False, rej_abs_line=rej_abs_line,
                    MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=linefit, 
                    Fe_uv_fix=Fe_uv_params, Fe_uv_range=Fe_uv_range,
                    Fe_op_fix=Fe_op_params, Fe_op_range=Fe_op_range,
                    save_result=False, plot_fig=False, save_fig=False, plot_corner=False, 
                    save_fits_name=None, save_fits_path=None, verbose=False,
                    kwargs_conti_emcee={'progress':False}, kwargs_line_emcee={'progress':False})        
        
    
    
    #Rerun until line fit is good
    if linefit: 
        rerun = check_rerun(qi, line_name)
        
        n = 0
        while rerun:
            
            try:
                qi = QSOFit(lam, flux, err, obj.z, ra=obj.ra, dec=obj.dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                            and_mask_in=and_mask, or_mask_in=or_mask)
                
                with suppress_stdout_stderr():
                    qi.Fit(name=name, nsmooth=1, deredden=True, 
                            and_mask=use_and_mask, or_mask=use_or_mask,
                            reject_badpix=False, wave_range=wave_range, wave_mask=wave_mask, 
                            decompose_host=False, 
                            Fe_uv_op=True, poly=False,
                            rej_abs_conti=False, rej_abs_line=rej_abs_line,
                            MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=linefit, 
                            Fe_uv_fix=Fe_uv_params, Fe_uv_range=Fe_uv_range,
                            Fe_op_fix=Fe_op_params, Fe_op_range=Fe_op_range,
                            save_result=False, plot_fig=False, save_fig=False, plot_corner=False,
                            save_fits_name=None, save_fits_path=None, verbose=False,
                            kwargs_conti_emcee={'progress':False}, kwargs_line_emcee={'progress':False})
            except Exception:
                use_and_mask = False
                use_or_mask = False
                
                qi = QSOFit(lam, flux, err, obj.z, ra=obj.ra, dec=obj.dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                            and_mask_in=and_mask, or_mask_in=or_mask)
                
                with suppress_stdout_stderr():
                    qi.Fit(name=name, nsmooth=1, deredden=True, 
                            and_mask=use_and_mask, or_mask=use_or_mask,
                            reject_badpix=False, wave_range=wave_range, wave_mask=wave_mask, 
                            decompose_host=False,
                            Fe_uv_op=True, poly=False,
                            rej_abs_conti=False, rej_abs_line=rej_abs_line,
                            MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=linefit, 
                            Fe_uv_fix=Fe_uv_params, Fe_uv_range=Fe_uv_range,
                            Fe_op_fix=Fe_op_params, Fe_op_range=Fe_op_range,
                            save_result=False, plot_fig=False, save_fig=False, plot_corner=False,
                            save_fits_name=None, save_fits_path=None, verbose=False,
                            kwargs_conti_emcee={'progress':False}, kwargs_line_emcee={'progress':False})  


            rerun = check_rerun(qi, line_name)

            if n > 5:
                break

            n += 1

    return qi


def get_feii_flux(obj, indices, qsopar_dir, nburn, nsamp, nthin,
                 output_dir, line_name, 
                 host_dir=None,
                 rej_abs_line=False, linefit=False, mask_line=False, 
                 Fe_uv_params=None, Fe_uv_range=None,
                 Fe_op_params=None, Fe_op_range=None,
                 ncpu=None):


    njob = len(indices)
    new_host_job = partial(host_job, 
                           obj=obj, qsopar_dir=qsopar_dir, line_name=line_name,
                           host_dir=host_dir,
                           rej_abs_line=rej_abs_line,
                           nburn=nburn, nsamp=nsamp, nthin=nthin,
                           linefit=linefit, mask_line=mask_line,
                           Fe_uv_params=Fe_uv_params, Fe_uv_range=Fe_uv_range,
                           Fe_op_params=Fe_op_params, Fe_op_range=Fe_op_range)

    if ncpu is None:
        ncpu = njob

    pool = mp.Pool(ncpu)
    qi_arr = pool.map( new_host_job, indices )
    pool.close()
    pool.join()    
    
    
    
    if line_name == 'mg2':
        wl_fe = np.linspace(2200, 3090, 3000)
    elif line_name == 'c4':
        wl_fe = np.linspace(1445, 1705, 3000)
    elif line_name == 'hb':
        wl_fe = np.linspace(4435, 5535, 3000)
    elif line_name == 'ha':
        wl_fe = np.linspace(6100, 7000, 3000)


    feii_arrs = []
    cont_arrs = []
    for i in range(njob):
        pp_tot = qi_arr[i].conti_result[7::2].astype(float)
        
        
        if line_name in ['mg2', 'c4']:
            feii_arrs.append( qi_arr[i].Fe_flux_mgii(wl_fe, pp_tot[:3] ) )
        elif line_name in ['hb', 'ha']:
            feii_mgii = qi_arr[i].Fe_flux_mgii(wl_fe, pp_tot[:3])
            feii_balmer = qi_arr[i].Fe_flux_balmer(wl_fe, pp_tot[3:6])
            feii_arrs.append( feii_mgii + feii_balmer )
            
    
        cont_arrs.append( qi_arr[i].PL(wl_fe, pp_tot) )
        
    if (Fe_uv_params is None) and (Fe_uv_range is None) and (Fe_op_params is None) and (Fe_op_range is None):
        save_feii_params(qi_arr, output_dir, line_name)
    else:
        resave_feii_params(indices, Fe_uv_params, Fe_op_params, qi_arr, output_dir, line_name)
    
    return wl_fe, feii_arrs, cont_arrs


##########################################################################################
################################### Refitting Bad Epochs #################################
##########################################################################################

def resave_feii_fluxes(indices, wl_fe, fe2_fluxes, cont_fluxes, output_dir, line_name):
    
    if line_name == 'mg2':
        colname = 'FeII_MgII'
    elif line_name == 'c4':
        colname = 'FeII_CIV'
    elif line_name == 'hb':
        colname = 'FeII_Hbeta'
    elif line_name == 'ha':
        colname = 'FeII_Halpha'
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_fnames = []
    for ind in indices:
        output_fnames.append( output_dir + 'FeII_fit_epoch{:03d}.dat'.format(ind+1) )
    
    for i in range(len(fe2_fluxes)):        
        dat = Table( [wl_fe, fe2_fluxes[i], cont_fluxes[i]], names=['RestWavelength', colname, 'PL_Cont'] )
        dat.write(output_fnames[i], format='ascii', overwrite=True)
    
    return



def find_bad_fits(fit_fnames, param_fname, line_name, method='prof', nsig=3):


    ##########################################################
    #METHOD 1 - Deviation from mean profile

    nepoch = len(fit_fnames)

    #Get data
    wl_arrs = []
    fe2_fluxes = []
    cont_fluxes = []
    
    if line_name == 'mg2':
        colname = 'FeII_MgII'
    elif line_name == 'c4':
        colname = 'FeII_CIV'
    elif line_name == 'hb':
        colname = 'FeII_Hbeta'
    elif line_name == 'ha':
        colname = 'FeII_Halpha'

    for i in range(nepoch):
        dat = Table.read(fit_fnames[i], format='ascii')
        wl_arrs.append(dat['RestWavelength'].tolist())
        fe2_fluxes.append(dat[colname].tolist())
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
    #METHOD 2 - Deviation from median FeII parameters
    
    width = erf(nsig/np.sqrt(2))
    plo = 100*(.5 - width/2)
    phi = 100*(.5 + width/2)
    
    param_dat = Table.read(param_fname, format='ascii')
    assert len(param_dat) == nepoch
    
    if line_name in ['mg2', 'c4']:
        cols = ['Norm', 'FWHM', 'Shift']
    elif line_name in ['ha', 'hb']:
        cols = ['Norm_uv', 'FWHM_uv', 'Shift_uv', 'Norm_op', 'FWHM_op', 'Shift_op']
    
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



def refit_bad_epochs(obj, fit_dir, qsopar_dir, nburn, nsamp, nthin, line_name,
                     host_dir=None,
                     fix=None, ranges=None, all=False, method='both', nsig=3,
                     rej_abs_line=False, linefit=False, mask_line=False,
                     ncpu=None):
    
    
    #Get filenames
    nepoch = len( glob.glob(fit_dir + 'FeII_fit_epoch*.dat') )
    fit_fnames = [ fit_dir + 'FeII_fit_epoch{:03d}.dat'.format(i+1) for i in range(nepoch) ]
    param_fname = fit_dir + 'best_fit_params.dat'
    
    bad_mask = find_bad_fits(fit_fnames, param_fname, line_name, method=method, nsig=nsig)

    if np.all(~bad_mask):
        print('No epochs to refit')
        return bad_mask
    
    if np.all(bad_mask):
        print('All epochs bad, increasing nsig')
        
        while np.all(bad_mask):
            nsig += 1
            bad_mask = find_bad_fits(fit_fnames, param_fname, line_name, method=method, nsig=nsig)
    
    
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
    
    if line_name in ['mg2', 'c4']:
        colnames = ['Norm', 'FWHM', 'Shift']
        vals = np.zeros( (3, len(colnames)) )

        for i, col in enumerate(colnames):
            lo, med, hi = np.percentile(param_dat[col][~bad_mask], [16,50,84])
            vals[:,i] = [lo, med, hi]

        possible_fix = ['norm', 'fwhm', 'shift']
        if fix is not None:
            fixed_params_uv = [None]*3
            
            for i, name in enumerate(possible_fix):
                if name in fix:
                    fixed_params_uv[i] = vals[1,i]
                
        else:
            fixed_params_uv = None


        if ranges is not None:
            range_params_uv = [None]*3
            
            for i, name in enumerate(possible_fix):
                if name in ranges:
                    range_params_uv[i] = [vals[0,i], vals[2,i]]
                    
        else:
            range_params_uv = None

            
            
        fixed_params_op = None
        range_params_op = None
        
    elif line_name in ['ha', 'hb']:
        colnames = ['Norm_uv', 'FWHM_uv', 'Shift_uv', 'Norm_op', 'FWHM_op', 'Shift_op']
        vals = np.zeros( (3, len(colnames)) )
        
        for i, col in enumerate(colnames):
            lo, med, hi = np.percentile( param_dat[col][~bad_mask].tolist(), [16,50,84] )
            vals[:, i] = [lo, med, hi]
        
        
        possible_fix_uv= ['norm_uv', 'fwhm_uv', 'shift_uv'] 
        possible_fix_op = ['norm_op', 'fwhm_op', 'shift_op']
        if fix is not None:
            fixed_params_uv = [None]*3
            fixed_params_op = [None]*3
            
            for i, name in enumerate(possible_fix_uv):
                if name in fix:
                    fixed_params_uv[i] = vals[1, i]
                    
            for i, name in enumerate(possible_fix_op):
                if name in fix:
                    fixed_params_op[i] = vals[1, i+3]
                    
        else:
            fixed_params_uv = None
            fixed_params_op = None

        if ranges is not None:
            range_params_uv = [None]*3
            range_params_op = [None]*3
            
            for i, name in enumerate(possible_fix_uv):
                if name in ranges:
                    range_params_uv[i] = [vals[0, i], vals[2, i]]
                    
            for i, name in enumerate(possible_fix_op):
                if name in ranges:
                    range_params_op[i] = [vals[0, i+3], vals[2, i+3]]
       
        else:
            range_params_uv = None
            range_params_op = None
            
        
        
    
    #Run fitting again
    wl_fe, fe2_fluxes, cont_fluxes = get_feii_flux(obj, indices, qsopar_dir, nburn, nsamp, nthin,
                                                fit_dir, line_name, host_dir=host_dir,
                                                rej_abs_line=rej_abs_line, linefit=linefit, mask_line=mask_line, 
                                                Fe_uv_params=fixed_params_uv, Fe_uv_range=range_params_uv,
                                                Fe_op_params=fixed_params_op, Fe_op_range=range_params_op,
                                                ncpu=ncpu)

    resave_feii_fluxes(indices, wl_fe, fe2_fluxes, cont_fluxes, fit_dir, line_name)

    return bad_mask




def iterate_refitting(obj, fit_dir, qsopar_dir, nburn, nsamp, nthin, line_name,
                      host_dir=None,
                     fix=None, ranges=None, all=False, method='both',
                     rej_abs_line=False, linefit=False, mask_line=False,
                     ncpu=None, niter=2):


    nepoch = len( glob.glob(fit_dir + 'FeII_fit_epoch*.dat') )
    fit_fnames = [ fit_dir + 'FeII_fit_epoch{:03d}.dat'.format(i+1) for i in range(nepoch) ]
    param_fname = fit_dir + 'best_fit_params.dat'


    nsig_arr = np.full(niter, 3, dtype=int)

    #First iteration
    masks_tot = refit_bad_epochs(obj, fit_dir, qsopar_dir, nburn, nsamp, nthin, line_name, host_dir=host_dir,
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
        
        if line_name in ['mg2', 'c4']:
            if fix == ['fwhm']:
                fix_arr = [['shift', 'fwhm']] 
                fix_arr = [fix]    
                    
            if fix == ['shift']:
                fix_arr = [['shift', 'fwhm']]
                fix_arr = [fix]
                
            if ('fwhm' in fix) and ('shift' in fix):
                fix_arr = [['shift', 'fwhm']]
                fix_arr = [fix]
                
        else:
            cond1 = ('fwhm_uv' in fix) and ('fwhm_op' in fix)
            cond2 = ('shift_uv' in fix) and ('shift_op' in fix)
            
            if cond1 and (len(fix)==2):
                fix_arr = [['shift_uv', 'fwhm_uv', 'shift_op', 'fwhm_op']]
                
            if cond2 and (len(fix)==2):
                fix_arr = [['shift_uv', 'fwhm_uv', 'shift_op', 'fwhm_op']]
                
            if cond1 and cond2 and (len(fix)==4):
                fix_arr = [['shift_uv', 'fwhm_uv', 'shift_op', 'fwhm_op']]    
    

    for i in range(niter - 1):
        mask_i = refit_bad_epochs(obj, fit_dir, qsopar_dir, nburn, nsamp, nthin, line_name, host_dir=host_dir,
                                fix=fix_arr[i], ranges=ranges, all=False, method=method, nsig=nsig_arr[i+1],
                                rej_abs_line=rej_abs_line, linefit=linefit, mask_line=mask_line,
                                ncpu=ncpu)
        
        
        if np.all(~mask_i):
            break
        
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
    bad_mask = find_bad_fits(fit_fnames, param_fname, line_name, method=method, nsig=nsig_arr[-1])
    bad_ind_still = np.argwhere(bad_mask).T[0]
    for ind in bad_ind_still:
        if ind not in bad_indices:
            bad_indices.append(ind)
    
            
            
    #Force fit these epochs
    if len(bad_indices) > 0:
        print('Fixing epochs:', np.array(bad_indices)+1)
        
        if line_name in ['mg2', 'c4']:
            refit_bad_epochs(obj, fit_dir, qsopar_dir, nburn, nsamp, nthin, line_name, host_dir=host_dir,
                            fix=['norm', 'fwhm', 'shift'], 
                            ranges=ranges, all=bad_indices, method=method, nsig=1,
                            rej_abs_line=rej_abs_line, linefit=linefit, mask_line=mask_line,
                            ncpu=ncpu)

        else:
            refit_bad_epochs(obj, fit_dir, qsopar_dir, nburn, nsamp, nthin, line_name, host_dir=host_dir,
                            fix=['norm_uv', 'fwhm_uv', 'shift_uv', 'norm_op', 'fwhm_op', 'shift_op'], 
                            ranges=ranges, all=bad_indices, method=method, nsig=1,
                            rej_abs_line=rej_abs_line, linefit=linefit, mask_line=mask_line,
                            ncpu=ncpu)

    
    #Save refit data
    with open(fit_dir + 'refit_data.dat', '+a') as f:
        f.write('Iteration Nsig Epoch Fixed\n')
        
        for i in range(len(refit_epochs)):
            f.write('{} {} {} {}\n'.format(refit_iter[i], nsig_tot[i], refit_epochs[i], refit_epochs[i]-1 in bad_indices))

        
    return masks_tot, refit_epochs, refit_iter

##########################################################################################
##################################### GET CONTINUUM ######################################
##########################################################################################

def cont_job(ind, obj, line_name, qsopar_dir, host_dir=None, rej_abs_line=False):

    fe_uv_params = np.array( list(obj.fe2_params[ind]) )[[1,3,5]]
    
    if line_name in ['mg2', 'c4']:
        fe_op_params = None
    else:
        fe_op_params = np.array( list(obj.fe2_params[ind]) )[[7,9,11]]
    
    
    qi = host_job(ind, obj, qsopar_dir, line_name,
                  rej_abs_line=rej_abs_line, 
                  nburn=100, nsamp=200, nthin=10,
                  host_dir=host_dir,
                  linefit=True, mask_line=False, 
                  Fe_uv_params=fe_uv_params, Fe_uv_range=None,
                  Fe_op_params=fe_op_params, Fe_op_range=None)
        
    return qi


def get_continuum(obj, line_name, qsopar_dir, host_dir=None, rej_abs_line=False):
    
    _ = make_qsopar(qsopar_dir)
    
    obj.get_fe2_params(line_name)
    newjob = partial(cont_job, obj=obj, line_name=line_name, 
                     qsopar_dir=qsopar_dir, host_dir=host_dir, 
                     rej_abs_line=rej_abs_line)
        
    pool = mp.Pool()
    qi_arr = pool.map( newjob, range(obj.nepoch) )
    pool.close()
    pool.join()
    
    
    colnames = ['PL_norm', 'PL_norm_err', 'PL_slope', 'PL_slope_err']        
    tot_params = np.zeros( (obj.nepoch, len(colnames)) )
    for i in range(obj.nepoch):
        pp_tot = qi_arr[i].conti_result.astype(float)

        for j in range(len(colnames)):
            tot_params[i, j] = pp_tot[ qi_arr[i].conti_result_name == colnames[j]]
        
        
    dat = Table( [obj.epochs], names=['Epoch'] )
    for i in range(len(colnames)):
        dat[colnames[i]] = tot_params[:, i]
        
    dat.write( obj.main_dir + line_name + '/fe2/best_fit_params_cont.dat', format='ascii', overwrite=True )

    return

##########################################################################################
################################### FOR EXTERNAL USE #####################################
##########################################################################################

def interpolate_fe2_flux(rest_wl, ref_flux_fname, line_name, cont=False):
    
    if line_name == 'mg2':
        colname = 'FeII_MgII'
    elif line_name == 'c4':
        colname = 'FeII_CIV'
    elif line_name == 'hb':
        colname = 'FeII_Hbeta'
    elif line_name == 'ha':
        colname = 'FeII_Halpha'
    
    dat = Table.read(ref_flux_fname, format='ascii')
    ref_wl = np.array(dat['RestWavelength'])
    ref_flux = np.array(dat[colname])
    
    spl = splrep(ref_wl, ref_flux, s=0)
    interp_fe2_flux = splev(rest_wl, spl, der=0)
    
    if cont:
        ref_cont = np.array(dat['PL_Cont'])
        
        spl = splrep(ref_wl, ref_cont, s=0)
        interp_cont = splev(rest_wl, spl, der=0)
        
        return interp_fe2_flux, interp_cont
    
    else:
        return interp_fe2_flux
    


def remove_fe2_mg2_flux(wl, flux, ref_feii_fname, line_name, z=None, cont=False):
    if z is None:
        z = 0.0
    
    rest_wl = wl / (1+z)
    
    if cont:
        interp_fe2_flux, interp_cont_flux = interpolate_fe2_flux(rest_wl, ref_feii_fname, line_name, cont=cont)
        return flux - interp_fe2_flux - interp_cont_flux
    
    else:
        interp_fe2_flux = interpolate_fe2_flux(rest_wl, ref_feii_fname, line_name)
        return flux - interp_fe2_flux

##########################################################################################
################################### Save Processed Spectra ###############################
##########################################################################################

def save_processed_spec_indiv(ind, obj, qsopar_dir, line_name,
                              rej_abs_line, 
                              nburn, nsamp, nthin,
                              host_dir=None):


    assert line_name in ['mg2', 'c4', 'hb', 'ha']
    
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

    wave_mask = None
    if line_name == 'mg2':
        wave_range = np.array([2200, 3090])      
              
    elif line_name == 'c4':
        wave_range = np.array([1445, 1705])            

    elif line_name == 'hb':
        wave_range = np.array([4435, 5535])            
        assert host_dir is not None, 'Must provide host_dir for H-beta'
            
    elif line_name == 'ha':
        wave_range = np.array([6100, 7000])            
        assert host_dir is not None, 'Must provide host_dir for H-alpha'

    
    use_and_mask = False
    use_or_mask = False
    
    pl_params = np.array( list(obj.pl_params[ind]) )[[1,3]]
    if line_name in ['mg2', 'c4']:
        fe_uv_params = np.array( list(obj.fe2_params[ind]) )[[1,3,5]]
        fe_op_params = None
    else:
        fe_uv_params = np.array( list(obj.fe2_params[ind]) )[[1,3,5]]
        fe_op_params = np.array( list(obj.fe2_params[ind]) )[[7,9,11]]


    if line_name in ['ha', 'hb']:
        flux, lam, err, and_mask, or_mask = remove_host_flux(lam, flux, err, and_mask, or_mask,
                                                            host_dir + 'best_host_flux.dat', 
                                                            z=obj.z)
        

    name = 'RM{:03d}e{:03d}'.format(obj.rmid, ind+1)
    try:
        qi = QSOFit(lam, flux, err, obj.z, ra=obj.ra, dec=obj.dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                    and_mask_in=and_mask, or_mask_in=or_mask)  
          
        qi.Fit(name=name, nsmooth=1, deredden=True, 
                and_mask=use_and_mask, or_mask=use_or_mask,
                reject_badpix=False, wave_range=wave_range, wave_mask=wave_mask, 
                decompose_host=False,
                Fe_uv_op=True, poly=False,
                rej_abs_conti=False, rej_abs_line=rej_abs_line,
                MCMC=False, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=False, 
                Fe_uv_fix=fe_uv_params, Fe_uv_range=None,
                Fe_op_fix=fe_op_params, Fe_op_range=None,
                PL_fix=pl_params, PL_range=None,
                save_result=False, plot_fig=False, save_fig=False, plot_corner=False, 
                save_fits_name=None, save_fits_path=None, verbose=False,
                kwargs_conti_emcee={'progress':False}, kwargs_line_emcee={'progress':False})
    except Exception:
        use_and_mask = False
        use_or_mask = False
        
        qi = QSOFit(lam, flux, err, obj.z, ra=obj.ra, dec=obj.dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                and_mask_in=and_mask, or_mask_in=or_mask)
        
        qi.Fit(name=name, nsmooth=1, deredden=True, 
                and_mask=use_and_mask, or_mask=use_or_mask,
                reject_badpix=False, wave_range=wave_range, wave_mask=wave_mask, 
                decompose_host=False,
                Fe_uv_op=True, poly=False,
                rej_abs_conti=False, rej_abs_line=rej_abs_line,
                MCMC=False, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=False, 
                Fe_uv_fix=fe_uv_params, Fe_uv_range=None,
                Fe_op_fix=fe_op_params, Fe_op_range=None,
                PL_fix=pl_params, PL_range=None,
                save_result=False, plot_fig=False, save_fig=False, plot_corner=False,
                save_fits_name=None, save_fits_path=None, verbose=False,
                kwargs_conti_emcee={'progress':False}, kwargs_line_emcee={'progress':False})        
    
    return qi


def save_processed_spec(obj, line_name, qsopar_dir, rej_abs_line=False,
                        nburn=100, nsamp=200, nthin=10):
    
    print('Saving processed spectra')
    
    #Run jobs
    obj.get_fe2_params(line_name)
    newjob = partial(save_processed_spec_indiv, obj=obj, qsopar_dir=qsopar_dir, line_name=line_name,
                     rej_abs_line=rej_abs_line, 
                     nburn=nburn, nsamp=nsamp, nthin=nthin,
                     host_dir=obj.main_dir + 'host_flux/')

    pool = mp.Pool()
    qi_arr = pool.map(newjob, range(obj.nepoch))
    pool.close()
    pool.join()
    
    for i in range(len(qi_arr)):
        wl = qi_arr[i].wave
        flux = qi_arr[i].flux - qi_arr[i].f_pl_model - qi_arr[i].f_fe_mgii_model - qi_arr[i].f_fe_balmer_model
        err = qi_arr[i].err
        and_mask = np.zeros_like(wl)
        or_mask = np.zeros_like(wl)
        
        dat = Table([wl*(1+obj.z), flux, err, and_mask, or_mask], names=['Wave[vaccum]', 'corrected_flux', 'corrected_err', 'ANDMASK', 'ORMASK'])
        dat.write(obj.main_dir + 'processed_spec/' + line_name + '/RM{:03d}_E{:03d}_{}_processed.csv'.format(obj.rmid, i+1, line_name), 
                  format='ascii.csv', overwrite=True)
    
    
    return

##########################################################################################
##########################################################################################
##########################################################################################

#HOW TO:
# 1. Run "get_feii_flux" to get preliminary results for the FeII flux for a given line
# 2. Save all of the fluxes with "save_feii_flux"
# 3. Run "iterate_refitting" to refit the badly fit epochs (sometimes multiple times depending on the method you choose)
# 4. Run "get_continuum" to refit (continuum+line) of all epochs with the host+FeII subtracted, and save the continuum params
# 5. Run "save_processed_spec" to refit all epochs with no mask, subtract the FeII+cont, and save the results
