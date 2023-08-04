from dmutils.specfit import host_contribution as host
from dmutils.specfit.object import Object

rmid = 160
obj = Object(160)

line_name = 'hb'
qsopar_dir = '/data3/stone28/2drm/sdssrm/fit_res_' + line_name + '/'
output_dir = '/data3/stone28/2drm/sdssrm/constants/host_fluxes_' + line_name + '/rm{:03d}/'.format(rmid)

wl_arrs, flux_arrs = host.get_host_flux(obj, range(obj.nepoch), qsopar_dir, line_name=line_name,
                                        nburn=100, nsamp=200, nthin=10,
                                        rej_abs_line=False, linefit=False)

host.save_host_fluxes(wl_arrs, flux_arrs, output_dir)
best_epoch, snr = host.get_best_host_flux(obj, output_dir, line_name=line_name)


print(best_epoch)
