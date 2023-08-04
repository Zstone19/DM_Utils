import dmutils.specfit.fe2_contribution as fe2
from dmutils.specfit.object import Object

import time
import sys


rmid = int(sys.argv[1])
line_name = sys.argv[2]


start = time.time()

obj = Object(rmid)
qsopar_dir = '/data3/stone28/2drm/sdssrm/fit_res_mg2/rm{:03d}/'.format(rmid)
output_dir = '/data3/stone28/2drm/sdssrm/constants/fe2_' + line_name + '/rm{:03d}/'.format(rmid)

if line_name in ['ha', 'hb']:
  host_dir = '/data3/stone28/2drm/sdssrm/constants/host_fluxes/'
else:
  host_dir = None
  
  
if line_name in ['ha', 'hb']:
  fix = ['fwhm_uv', 'fwhm_op', 'shift_uv', 'shift_op']
else:
  fix = ['fwhm', 'shift']

#Initial fit
wl_fe, feii_arrs, cont_arrs = fe2.get_feii_flux(obj, range(obj.nepoch), 
                                               qsopar_dir, 100, 200, 10,
                                               output_dir, line_name=line_name,
                                               host_dir=host_dir,
                                               linefit=False, mask_line=True)
#Save FeII profiles
fe2.save_feii_fluxes( wl_fe, feii_arrs, cont_arrs, output_dir, line_name=line_name)

#Refit the bad epochs
fe2.iterate_refitting( obj, output_dir, qsopar_dir, 
                      100, 200, 10, line_name=line_name,
                      host_dir=host_dir,
                      all=True, fix=fix, method='both',
                      linefit=False, mask_line=True, niter=3)

print( 'Total time: {:.2f} seconds'.format(time.time()-start) )
