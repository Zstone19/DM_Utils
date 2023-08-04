from dmutils.specfit import host_contribution as host
import dmutils.specfit.fe2_contribution as fe2
from dmutils.specfit.object import Object




rmid = 160
obj = Object(160)

################################################################################################
#FIT HOST FLUX
qsopar_dir = '/data3/stone28/2drm/sdssrm/fit_res_mg2/'
host_outdir = '/data3/stone28/2drm/sdssrm/constants/host_fluxes/rm{:03d}/'.format(rmid)

wl_arrs, flux_arrs = host.get_host_flux(obj, range(obj.nepoch), qsopar_dir, line_name=None,
                                        nburn=100, nsamp=200, nthin=10,
                                        rej_abs_line=False, linefit=False)

host.save_host_fluxes(wl_arrs, flux_arrs, host_outdir)
best_epoch, snr = host.get_best_host_flux(obj, host_outdir, line_name=None)


################################################################################################
#FIT FeII FLUX
for line_name in ['ha', 'hb', 'mg2']:
    
    if line_name in ['ha', 'hb']:
        fix = ['fwhm_uv', 'fwhm_op', 'shift_uv', 'shift_op']
        host_dir = '/data3/stone28/2drm/sdssrm/constants/host_fluxes/'
    else:
        fix = ['fwhm', 'shift']
        host_dir = None


    if line_name == 'mg2':
        continue        

    qsopar_dir = '/data3/stone28/2drm/sdssrm/fit_res_' + line_name + '/'
    fe2_outdir = '/data3/stone28/2drm/sdssrm/constants/fe2_' + line_name + '/rm{:03d}/'.format(rmid)

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
