from dmutils.specfit.fit_lines import run_all_fits
    
for name in ['ha', 'hb']:
    run_all_fits(160, name, 
            fe2_dir='/data3/stone28/2drm/sdssrm/constants/fe2_' + name + '/',
            res_dir='/data3/stone28/2drm/sdssrm/fit_res_' + name + '/',
            host_dir='/data3/stone28/2drm/sdssrm/constants/host_fluxes/',
            prefix='')
