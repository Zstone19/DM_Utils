import numpy as np
import multiprocessing as mp
import os
from functools import partial

from astropy.table import Table
from astropy.io import fits

from dmutils.specfit.host_contribution import remove_host_flux
from dmutils.specfit.object import Object

from pyqsofit.PyQSOFit import QSOFit


from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


##############################################################################
############################### USEFUL FUNCTIONS #############################
##############################################################################


#NEED TO REWORK THIS !!!!!!!!!!!!!!!!
def find_optimal_ngauss(lam, flux, err, z, ra, dec, mjd, fitpath, 
                        qsopar_header, qsopar_dat, line_name='hb', 
                        ngauss_max=7, bic_tol=10):
    
    if line_name == 'ha':
        line_str = 'Ha_br'
    elif line_name == 'hb':
        line_str = 'Hb_br'
    elif line_name == 'mg2':
        line_str = 'MgII_br'
    elif line_name == 'c4':
        line_str = 'CIV_br'
    
    
    bic_last = np.inf
    for ngauss in range(1, ngauss_max):
        
        print(fr'Fitting {line_name} with {ngauss} components.')
        
        newdata_n = qsopar_dat.copy()
        for i, row in enumerate(newdata_n):
            if line_str in str(row['linename']):
                newdata_n[i]['ngauss'] = ngauss

        hdu = fits.BinTableHDU(data=newdata_n, header=qsopar_header, name='data')
        hdu.writeto(os.path.join(fitpath, 'qsopar.fits'), overwrite=True)
        
        
        q = QSOFit(lam, flux, err, z, ra=ra, dec=dec, mjd=int(mjd), path=fitpath)
    
        q.Fit(name=None, nsmooth=1, deredden=True, reject_badpix=False, wave_range=None, wave_mask=None, 
                decompose_host=True, npca_gal=5, npca_qso=20, 
                Fe_uv_op=True, poly=True, rej_abs_conti=False,
                MCMC=True, epsilon_jitter=1e-4, nburn=100, nsamp=200, nthin=10, linefit=True, 
                save_result=False, plot_fig=False, save_fig=False, plot_corner=False,
                verbose=False)
    
    
        if line_name == 'Ha':
            mask_bic = (q.line_result_name == '2_line_min_chi2')
        elif line_name == 'Hb':
            mask_bic = (q.line_result_name == '1_line_min_chi2')
        
        bic = float(q.line_result[mask_bic][0])
        print('Delta BIC: {:.1f}'.format(bic_last - bic) )
        
        if np.abs(bic_last - bic) < bic_tol:
            ngauss_best = ngauss - 1
            break
    
        bic_last = bic
    
    return ngauss_best




def check_bad_run(qi, line_name):
    
    rerun = False

    if line_name is None:
        #Get chi2 of Hbeta
        c = np.argwhere( qi.uniq_linecomp_sort == 'H$\\beta$' ).T[0][0]
        chi2_nu1 = float(qi.comp_result[c*7+4])
        
        #Get scale of OIII4959
        names = qi.line_result_name
        oiii_mask = (names == 'OIII4959c_1_scale')
        oiii_scale = float(qi.line_result[oiii_mask])
            
        #Get chi2 of MgII
        c = np.argwhere( qi.uniq_linecomp_sort == 'MgII' ).T[0][0]
        chi2_nu2 = float(qi.comp_result[c*7+4])
        
        #Get chi2 of CIV
        c = np.argwhere( qi.uniq_linecomp_sort == 'CIV' ).T[0][0]
        chi2_nu3 = float(qi.comp_result[c*7+4])
        
        #Get chi2 of Halpha
        c = np.argwhere( qi.uniq_linecomp_sort == 'H$\\alpha$' ).T[0][0]
        chi2_nu4 = float(qi.comp_result[c*7+4])        
        
        
        if (chi2_nu1 > 3) or (chi2_nu2 > 3) or (oiii_scale < 1) or (chi2_nu3 > 3) or (chi2_nu4 > 3):
            rerun = True


    elif line_name == 'hb':
        #Get chi2 of Hbeta
        c = np.argwhere( qi.uniq_linecomp_sort == 'H$\\beta$' ).T[0][0]
        chi2_nu1 = float(qi.comp_result[c*7+4])
        
        #Get scale of OIII4959
        names = qi.line_result_name
        oiii_mask = (names == 'OIII4959c_1_scale')
        oiii_scale = float(qi.line_result[oiii_mask])
        
        if (chi2_nu1 > 3) or (oiii_scale < 1):
            rerun = True

        
    elif line_name == 'mg2':
        #Get chi2 of MgII
        c = np.argwhere( qi.uniq_linecomp_sort == 'MgII' ).T[0][0]
        chi2_nu2 = float(qi.comp_result[c*7+4])
    
        if chi2_nu2 > 3:
            rerun = True

            
    elif line_name == 'c4':
        #Get chi2 of CIV
        c = np.argwhere( qi.uniq_linecomp_sort == 'CIV' ).T[0][0]
        chi2_nu3 = float(qi.comp_result[c*7+4])
        
        if chi2_nu3 > 3:
            rerun = True
        

    elif line_name == 'ha':
        c = np.argwhere( qi.uniq_linecomp_sort == 'H$\\alpha$' ).T[0][0]
        chi2_nu4 = float(qi.comp_result[c*7+4])
        
        if chi2_nu4 > 3:
            rerun = True

    return rerun




###############################################################################################
###############################################################################################
###############################################################################################


def run_pyqsofit(obj, ind, output_dir, qsopar_dir, line_name=None, prefix='', host_dir=None, rej_abs_line=False):
    print('Fitting epoch {}'.format(ind+1))

    if line_name not in ['mg2', 'c4']:
        assert host_dir is not None, 'host_dir must be specified for non-MgII lines.'        

    if obj.processed:
        lam = np.array(obj.table_arr[line_name][ind]['Wave[vaccum]'])
        flux = np.array(obj.table_arr[line_name][ind]['corrected_flux'])
        err = np.array(obj.table_arr[line_name][ind]['corrected_err'])
        and_mask = np.array(obj.table_arr[line_name][ind]['ANDMASK'])
        or_mask = np.array(obj.table_arr[line_name][ind]['ORMASK'])
        
    else:
        lam = np.array(obj.table_arr[ind]['Wave[vaccum]'])
        flux = np.array(obj.table_arr[ind]['corrected_flux'])
        err = np.array(obj.table_arr[ind]['corrected_err'])
        
        and_mask = np.array(obj.table_arr[ind]['ANDMASK'])
        or_mask = np.array(obj.table_arr[ind]['ORMASK'])

    epoch = obj.epochs[ind]
    mjd = obj.mjd[ind]
    plateid = obj.plateid[ind]
    fiberid = obj.fiberid[ind]
    
    
    decompose_host = True
    if line_name is None:
        wave_range = None
    elif line_name == 'mg2':
        wave_range = np.array([2200, 3090])
        decompose_host = False
        center = 2798
    elif line_name == 'c4':
        wave_range = np.array([1445, 1705])
        decompose_host = False
        center = 1549
    elif line_name == 'hb':
        wave_range = np.array([4435, 5535])
        center = 4861
    elif line_name == 'ha':
        wave_range = np.array([6100, 7000])
        center = 6563
    

    ##############################
    # FeII
    
    if line_name in ['mg2', 'c4']:
        fe_uv_params = np.array( list(obj.fe2_params[ind]) )[[1,3,5]]
        fe_op_params = None
    else:
        fe_uv_params = np.array( list(obj.fe2_params[ind]) )[[1,3,5]]
        fe_op_params = np.array( list(obj.fe2_params[ind]) )[[7,9,11]]
        
        
    if obj.o3_corr is not None:
        fe_uv_params[0] = fe_uv_params[0] * obj.o3_corr['FluxCorr'][ind]
        fe_uv_params[1] = fe_uv_params[1] * obj.o3_corr['CenterCorr'][ind]
        
        #Let's say that we focus on a given pixel from the spectrum x0
        #QSOFit applies the FeII value from pixel x0*(1+shift) to x0
        
        #We want to apply the FeII value from pixel x0*(1+shift) to x1=x0*o3cc
        #QSOFit will try to apply the FeII value from pixel x1*(1+shift) to x1
        #In order to apply the FeII value from pixel x0*(1+shift) to x1, we need to change the shift to a new value, shift2
        #   x1*(1+shift2) = x0*(1+shift)
        #   shift2 = (x0/x1)*(1+shift) - 1
        #          = (1/o3cc)*(1+shift) - 1

        fe_uv_params[2] = (1/obj.o3_corr['CenterCorr'][ind])*(1+fe_uv_params[2]) - 1
        
        
        if fe_op_params is not None:
            fe_op_params[0] = fe_op_params[0] * obj.o3_corr['FluxCorr'][ind]
            fe_op_params[1] = fe_op_params[1] * obj.o3_corr['CenterCorr'][ind]
            fe_op_params[2] = (1/obj.o3_corr['CenterCorr'][ind])*(1+fe_op_params[2]) - 1
        
        
    fit_fe2 = True
    if obj.processed:
        decompose_host = False
        wave_range = None
        fe_uv_params = None
        fe_op_params = None
        
        fit_fe2 = False
        
    
        
    ##############################
    # Host    

    if decompose_host:
        if obj.o3_corr is not None:
            flux, lam, err, and_mask, or_mask = remove_host_flux(lam/obj.o3_corr['CenterCorr'][ind], 
                                                               flux/obj.o3_corr['FluxCorr'][ind], 
                                                               err/obj.o3_corr['FluxCorr'][ind], 
                                                               and_mask, or_mask,
                                                               host_dir + 'best_host_flux.dat', 
                                                               z=obj.z)
            
            lam *= obj.o3_corr['CenterCorr'][ind]
            flux *= obj.o3_corr['FluxCorr'][ind]
            err *= obj.o3_corr['FluxCorr'][ind]
            
        else:
            flux, lam, err, and_mask, or_mask = remove_host_flux(lam, flux, err, and_mask, or_mask,
                                                            host_dir + 'best_host_flux.dat', 
                                                            z=obj.z)
        
    ##############################
    # Continuum

    poly = True
    if line_name is not None:
        poly = False
        
    pl_params = None
    if obj.processed:
        pl_params = [0, None]
        poly = False
        
    ##############################
    # Run fits
        
    masks = True        
    nburn = 100
    nsamp = 200
    nthin = 10
    
    #Don't use masks if they remove the line
    if line_name is not None:
        new_lam = lam.copy()
        mask_ind = np.where( (and_mask == 0) & (and_mask == 0) , True, False)
        new_lam = new_lam[mask_ind]

        if ( np.min(new_lam) > center*(1+obj.z) - 300) or ( np.max(new_lam) < center*(1+obj.z) + 300):
            masks = False

    if obj.processed:
        masks = False
    

    name = 'RM{:03d}e{:03d}'.format(obj.rmid, epoch) + prefix    
        
    try:
        qi = QSOFit(lam, flux, err, obj.z, ra=obj.ra, dec=obj.dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                    and_mask_in=and_mask, or_mask_in=or_mask)
        
        with suppress_stdout_stderr():
            qi.Fit(name=name, nsmooth=1, deredden=True, 
                    and_mask=masks, or_mask=masks,
                reject_badpix=False, wave_range=wave_range, wave_mask=None, 
                decompose_host=False, npca_gal=5, npca_qso=20, 
                Fe_uv_op=fit_fe2, poly=poly,
                rej_abs_conti=False, rej_abs_line=rej_abs_line,
                MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=True, 
                Fe_uv_fix=fe_uv_params, Fe_op_fix=fe_op_params, PL_fix=pl_params,
                save_result=True, plot_fig=True, save_fig=True, plot_corner=False, kwargs_plot={'save_fig_path':output_dir}, 
                save_fits_name=name+'_pyqsofit', save_fits_path=output_dir, verbose=False)
        
        if qi.wave.min() > center - 300:
            raise Exception
        if qi.wave.max() < center + 300:
            raise Exception
        
    except:
        masks = False
        
        qi = QSOFit(lam, flux, err, obj.z, ra=obj.ra, dec=obj.dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                    and_mask_in=and_mask, or_mask_in=or_mask)
        
        with suppress_stdout_stderr():
            qi.Fit(name=name, nsmooth=1, deredden=True, 
                    and_mask=masks, or_mask=masks,
                reject_badpix=False, wave_range=wave_range, wave_mask=None, 
                decompose_host=False, npca_gal=5, npca_qso=20, 
                Fe_uv_op=fit_fe2, poly=poly,
                rej_abs_conti=False, rej_abs_line=rej_abs_line,
                MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=True, 
                Fe_uv_fix=fe_uv_params, Fe_op_fix=fe_op_params, PL_fix=pl_params,
                save_result=True, plot_fig=True, save_fig=True, plot_corner=False, kwargs_plot={'save_fig_path':output_dir}, 
                save_fits_name=name+'_pyqsofit', save_fits_path=output_dir, verbose=False)


    rerun1 = check_bad_run(qi, line_name)
    rerun = rerun1

    #If chi2nu is too high or it doesn't fit [OIII]4959, rerun a couple of times
    n = 0
    while (rerun is True):

        n += 1
        qi = QSOFit(lam, flux, err, obj.z, ra=obj.ra, dec=obj.dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir)

        with suppress_stdout_stderr():
            qi.Fit(name=name, nsmooth=1, deredden=True, 
                and_mask=masks, or_mask=masks,
                reject_badpix=False, wave_range=wave_range, wave_mask=None, 
                decompose_host=False, npca_gal=5, npca_qso=20, 
                Fe_uv_op=fit_fe2, poly=poly, 
                rej_abs_conti=False, rej_abs_line=rej_abs_line,
                MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=True, 
                Fe_uv_fix=fe_uv_params, Fe_op_fix=fe_op_params, PL_fix=pl_params,
                save_result=True, plot_fig=True, save_fig=True, plot_corner=False, kwargs_plot={'save_fig_path':output_dir}, 
                save_fits_name=name+'_pyqsofit', save_fits_path=output_dir, verbose=False)


        rerun = check_bad_run(qi, line_name)

        if n > 5:
            break



    if rerun1:
        with open(qsopar_dir + 'rerun.txt', 'a') as f:
            f.write('{},{}\n'.format(n, epoch))

    if rerun:
        with open(qsopar_dir + 'bad_run.txt', 'a') as f:
            f.write('{}\n'.format(epoch))


    return qi


#Job function (per epoch)
def job(ind, obj, res_dir, qsopar_dir, line_name=None, prefix='', host_dir=None, rej_abs_line=False):

    epoch = obj.epochs[ind]    
    epoch_dir = res_dir + 'epoch{:03d}/'.format(epoch)
    qi = run_pyqsofit(obj, ind, epoch_dir, qsopar_dir, line_name=line_name, prefix=prefix, 
                      host_dir=host_dir, rej_abs_line=rej_abs_line)
    
    #Get line fitting results
    gauss_result_tot = qi.gauss_result_all
    gauss_result = qi.gauss_result[::2]
    gauss_names = qi.gauss_result_name[::2]


    ####################################################################
    ####################################################################
    # Hbeta

    if (line_name is None) or (line_name == 'hb'):
        #Get Hbeta broad profiles (need to load all MCMC samples)
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:2] != 'Hb') or (gauss_names[3*p + 2][3:5] != 'br'):
                continue
            
            pvals.append(p)


        profiles = []
        for i in range(gauss_result_tot.shape[0]):

            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
        
        
        profiles = np.vstack(profiles)

        prof_med = np.median(profiles, axis=0)
        prof_err_lo = prof_med - np.percentile(profiles, 16, axis=0)
        prof_err_hi = np.percentile(profiles, 84, axis=0) - prof_med

        colnames = ['wavelength', 'profile', 'err_lo', 'err_hi']            
        profile_info = Table( [qi.wave, prof_med, prof_err_lo, prof_err_hi], names=colnames)
        profile_info.write( epoch_dir + 'Hb_br_profile.csv', overwrite=True )


    ####################################################################
    ####################################################################
    # OIII

    if (line_name is None) or (line_name == 'hb'):
        #Get OIII core and wing profiles
        profiles = []
        names = []
        for p in range( len(gauss_result)//3 ):
            if gauss_names[3*p + 2][:4] != 'OIII':
                continue
            
            
            profile = qi.Onegauss( np.log(qi.wave), gauss_result[3*p:3*(p+1)] )
            profiles.append( profile )
            names.append( gauss_names[3*p + 2][:9] )
            
        
        profile_info = np.vstack([ [qi.wave], profiles])
        colnames = ['wavelength']
        for i in range(len(profiles)):
            colnames.append( names[i] )
            
        profile_info = Table( profile_info.T, names=colnames)
        profile_info.write( epoch_dir + 'OIII_profile.csv', overwrite=True )
    

    ####################################################################   
    ####################################################################
    # Halpha
    
    if (line_name is None) or (line_name == 'ha'):
        #Get Halpha broad profiles (need to load all MCMC samples)
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:2] != 'Ha') or (gauss_names[3*p + 2][3:5] != 'br'):
                continue
            
            pvals.append(p)


        profiles = []
        for i in range(gauss_result_tot.shape[0]):

            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
    
    
        profiles = np.vstack(profiles)

        prof_med = np.median(profiles, axis=0)
        prof_err_lo = prof_med - np.percentile(profiles, 16, axis=0)
        prof_err_hi = np.percentile(profiles, 84, axis=0) - prof_med

        colnames = ['wavelength', 'profile', 'err_lo', 'err_hi']            
        profile_info = Table( [qi.wave, prof_med, prof_err_lo, prof_err_hi], names=colnames)
        profile_info.write( epoch_dir + 'Ha_br_profile.csv', overwrite=True )
    
    
    ####################################################################
    ####################################################################
    # MgII
    
    if (line_name is None) or (line_name == 'mg2'):
        #Get MgII broad profiles (need to load all MCMC samples)
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:4] != 'MgII') or (gauss_names[3*p + 2][5:7] != 'br'):
                continue
            
            pvals.append(p)


        profiles = []
        for i in range(gauss_result_tot.shape[0]):

            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
        
        
        profiles = np.vstack(profiles)

        prof_med = np.median(profiles, axis=0)
        prof_err_lo = prof_med - np.percentile(profiles, 16, axis=0)
        prof_err_hi = np.percentile(profiles, 84, axis=0) - prof_med

        colnames = ['wavelength', 'profile', 'err_lo', 'err_hi']            
        profile_info = Table( [qi.wave, prof_med, prof_err_lo, prof_err_hi], names=colnames)
        profile_info.write( epoch_dir + 'MgII_br_profile.csv', overwrite=True )


    ####################################################################
    ####################################################################
    # CIV
    
    if (line_name is None) or (line_name == 'c4'):
        #Get MgII broad profiles (need to load all MCMC samples)
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:3] != 'CIV') or (gauss_names[3*p + 2][4:6] != 'br'):
                continue
            
            pvals.append(p)


        profiles = []
        for i in range(gauss_result_tot.shape[0]):

            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
        
        
        profiles = np.vstack(profiles)

        prof_med = np.median(profiles, axis=0)
        prof_err_lo = prof_med - np.percentile(profiles, 16, axis=0)
        prof_err_hi = np.percentile(profiles, 84, axis=0) - prof_med

        colnames = ['wavelength', 'profile', 'err_lo', 'err_hi']            
        profile_info = Table( [qi.wave, prof_med, prof_err_lo, prof_err_hi], names=colnames)
        profile_info.write( epoch_dir + 'CIV_br_profile.csv', overwrite=True )
    
    
    ####################################################################
    ####################################################################
    # Continuum
    
    continuum = qi.f_conti_model
    cont_info = Table( [qi.wave, continuum], names=['wavelength', 'flux'])
    cont_info.write( epoch_dir + 'continuum.csv', overwrite=True )
        
        
    ####################################################################
    ####################################################################
    # Raw broad profiles
    
    raw_br_prof = get_raw_br_prof(qi, line_name)
    raw_br_info = Table( [qi.wave, raw_br_prof, qi.err], names=['wavelength', 'flux', 'err'])
    raw_br_info.write( epoch_dir + 'raw_br_profile.csv', overwrite=True )
        
    return



def get_raw_br_prof(qi, line_name):

    #Get results    
    gauss_result_tot = qi.gauss_result_all
    gauss_result = qi.gauss_result[::2]
    gauss_names = qi.gauss_result_name[::2]
    
    ####################################################################
    ####################################################################
    
    if line_name == 'hb':
        
        #Get Hbeta narrow profile
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:2] != 'Hb') or (gauss_names[3*p + 2][3:5] != 'na'):
                continue
            
            pvals.append(p)
        
        profiles = []
        for i in range(gauss_result_tot.shape[0]):

            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
        
        profiles = np.vstack(profiles)        
        na_prof = np.median(profiles, axis=0)
        
        #Get continuum
        continuum = qi.f_conti_model
        
        #Get OIII
        o3_prof = np.zeros_like( qi.wave )
        for p in range( len(gauss_result)//3 ):
            if gauss_names[3*p + 2][:4] != 'OIII':
                continue
            
            o3_prof += qi.Onegauss( np.log(qi.wave), gauss_result[3*p:3*(p+1)] )
            
        
        #Get HeII
        he2_prof = np.zeros_like( qi.wave )
        for p in range( len(gauss_result)//3 ):
            if gauss_names[3*p + 2][:4] != 'HeII':
                continue
            
            he2_prof += qi.Onegauss( np.log(qi.wave), gauss_result[3*p:3*(p+1)] )
        
        
        raw_prof = qi.flux - (na_prof + continuum + o3_prof + he2_prof)
        
        
    ####################################################################
    ####################################################################
        
    if line_name == 'ha':
        
        #Get Halpha narrow profile
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:2] != 'Ha') or (gauss_names[3*p + 2][3:5] != 'na'):
                continue
            
            pvals.append(p)
            
        profiles = []
        for i in range(gauss_result_tot.shape[0]):
            
            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
            
        profiles = np.vstack(profiles)
        na_prof = np.median(profiles, axis=0)
        
        
        #Get continuum
        continuum = qi.f_conti_model
        
        #Get NII
        n2_prof = np.zeros_like( qi.wave )
        for p in range( len(gauss_result)//3 ):
            if gauss_names[3*p + 2][:3] != 'NII':
                continue
            
            n2_prof += qi.Onegauss( np.log(qi.wave), gauss_result[3*p:3*(p+1)] )
            
        
        #Get SII
        s2_prof = np.zeros_like( qi.wave )
        for p in range( len(gauss_result)//3 ):
            if gauss_names[3*p + 2][:3] != 'SII':
                continue
            
            s2_prof += qi.Onegauss( np.log(qi.wave), gauss_result[3*p:3*(p+1)] )
            
        
        #Get OI
        oi_prof = np.zeros_like( qi.wave )
        for p in range( len(gauss_result)//3 ):
            if gauss_names[3*p + 2][:3] != 'OI6':
                continue
            
            oi_prof += qi.Onegauss( np.log(qi.wave), gauss_result[3*p:3*(p+1)] )
            
        
        raw_prof = qi.flux - (na_prof + continuum + n2_prof + s2_prof + oi_prof)
    

    ####################################################################
    ####################################################################
    
    if line_name == 'mg2':
        
        #Get MgII narrow profile
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:4] != 'MgII') or (gauss_names[3*p + 2][5:7] != 'na'):
                continue
            
            pvals.append(p)
            
        profiles = []
        for i in range(gauss_result_tot.shape[0]):
                
            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
                
        profiles = np.vstack(profiles)
        na_prof = np.median(profiles, axis=0)
        
        
        #Get continuum
        continuum = qi.f_conti_model
        
        raw_prof = qi.flux - (na_prof + continuum)    
    

    ####################################################################
    ####################################################################
        
    if line_name == 'c4':
        
        #Get CIV narrow profile
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:3] != 'CIV') or (gauss_names[3*p + 2][4:6] != 'na'):
                continue
            
            pvals.append(p)
            
        profiles = []
        for i in range(gauss_result_tot.shape[0]):
                
            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
                
        profiles = np.vstack(profiles)
        na_prof = np.median(profiles, axis=0)
        
        
        #Get continuum
        continuum = qi.f_conti_model
        
        #Get HeII
        he2_prof = np.zeros_like( qi.wave )
        for p in range( len(gauss_result)//3 ):
            if gauss_names[3*p + 2][:4] != 'HeII':
                continue
            
            he2_prof += qi.Onegauss( np.log(qi.wave), gauss_result[3*p:3*(p+1)] )
            
        
        #Get OIII
        o3_prof = np.zeros_like( qi.wave )
        for p in range( len(gauss_result)//3 ):
            if gauss_names[3*p + 2][:4] != 'OIII':
                continue
            
            o3_prof += qi.Onegauss( np.log(qi.wave), gauss_result[3*p:3*(p+1)] )
        

        raw_prof = qi.flux - (na_prof + continuum + he2_prof + o3_prof)    
    
    
    return raw_prof


###############################################################################################
###############################################################################################
###############################################################################################


def run_all_fits(rmid, line_name, main_dir, prefix='', host=True, rej_abs_line=False, ncpu=None):
    
    fe2_dir = main_dir + 'rm{:03d}/'.format(rmid) + line_name + '/fe2/'
    res_dir = main_dir + 'rm{:03d}/'.format(rmid) + line_name + '/qsofit/'
    
    if host:
        host_dir = main_dir + 'rm{:03d}/host_flux/'.format(rmid)
    else:
        host_dir = None
    
    #Load data
    obj = Object(rmid)
    obj.get_fe2_params(line_name)

    #Make qsopar file
    header, newdata = make_qsopar(res_dir, oiii_wings=True)

    #Make bad run/rerun files
    with open(res_dir + 'bad_run.txt', 'w+') as f:
        f.write('#epoch\n')

    with open(res_dir + 'rerun.txt', 'w+') as f:
        f.write('#nrerun,epoch\n')
        

    #Make main result directory
    os.makedirs(res_dir, exist_ok=True)

    #Make individual epoch directories
    for i in range(obj.nepoch):
        epoch = obj.epochs[i]    
        dir_i = res_dir + 'epoch{:03d}/'.format(epoch)
        
        os.makedirs(dir_i, exist_ok=True)
        
    
    specific_job = partial(job, obj=obj, res_dir=res_dir, line_name=line_name, prefix=prefix, 
                           host_dir=host_dir, rej_abs_line=rej_abs_line)
    if ncpu is None:
        ncpu = obj.nepoch
    
    pool = mp.Pool(ncpu)
    pool.map(specific_job, range(obj.nepoch))
    pool.close()
    pool.join()

    return
