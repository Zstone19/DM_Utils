from dmutils.specfit import host_contribution as host
import dmutils.specfit.fe2_contribution as fe2
from dmutils.specfit.object import Object

import os
import glob
import numpy as np






#Get RMIDS
method = 'median'
# rmids = np.loadtxt('/data3/stone28/gem_targets/redo_host.txt', dtype=int, unpack=True)

# method = 'snr'
rmids = [160]


with open('/data3/stone28/2drm/sdssrm/bad_run_cont_fe2.txt', 'w+') as f:
    f.write('# GEMID \t line \n')



main_dir = '/data3/stone28/2drm/sdssrm/'


#Run
for rmid in rmids:
    
    if not os.path.exists(main_dir + 'rm{:03d}/'.format(rmid)):        
        print('')
        print('-------------------')
        print('RM{:03d} NOT FOUND'.format(rmid))
        print('-------------------')
        print('')
        
        continue
    
    
    print('')
    print('====================================')
    print('RUNNING RM{:03d}'.format(rmid))
    print('====================================')
    print('')
    

    obj = Object(rmid)
    line_names = obj.get_line_names()


    ################################################################################################
    #FIT HOST FLUX

    if ('ha' not in line_names) and ('hb' not in line_names):
        pass
    else:
        qsopar_dir = '/data3/stone28/2drm/sdssrm/'
        host_outdir = obj.main_dir + 'host_flux/'

        wl_arrs, flux_arrs = host.get_host_flux(obj, range(obj.nepoch), qsopar_dir, line_name=None,
                                                nburn=100, nsamp=200, nthin=10,
                                                rej_abs_line=False, linefit=False)

        host.save_host_fluxes(wl_arrs, flux_arrs, host_outdir)
        _ = host.get_best_host_flux(obj, host_outdir, line_name=None, method=method)


    ################################################################################################
    #FIT FeII FLUX
    for line_name in line_names:

        if line_name in ['ha', 'hb']:
            fix = ['fwhm_uv', 'fwhm_op', 'shift_uv', 'shift_op']
            host_dir = obj.main_dir + 'host_flux/'
        else:
            fix = ['fwhm', 'shift']
            host_dir = None



        try:
            qsopar_dir = '/data3/stone28/2drm/sdssrm/'
            fe2_outdir = obj.main_dir + line_name + '/fe2/'

            wl_fe, feii_arrs, cont_arrs = fe2.get_feii_flux(obj, range(obj.nepoch), 
                                                        qsopar_dir, 100, 200, 10,
                                                        fe2_outdir, line_name=line_name,
                                                        host_dir=host_dir,
                                                        linefit=False, mask_line=True)

            fe2.save_feii_fluxes( wl_fe, feii_arrs, cont_arrs, fe2_outdir, line_name=line_name)

            fe2.iterate_refitting( obj, fe2_outdir, qsopar_dir, 
                                100, 200, 10, line_name=line_name,
                                host_dir=host_dir,
                                all=True, fix=fix, method='both',
                                linefit=False, mask_line=True, niter=3)

        except Exception:
            print('ERROR: GEM{:03d} {}'.format(rmid, line_name))
            
            with open('/data3/stone28/2drm/sdssrm/bad_run_cont_fe2.txt', 'a') as f:
                f.write('{:03d} \t {}\n'.format(rmid, line_name))

