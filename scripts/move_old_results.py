import glob
import os
import shutil

import numpy as np





rmid = 160



#Move host flux files
old_dir = '/data3/stone28/2drm/sdssrm_OLD/constants/host_fluxes/rm160/'
new_dir = '/data3/stone28/2drm/sdssrm/rm160/host_flux/' 
for fname in glob.glob( old_dir + '*' ):
    shutil.copy( fname, new_dir )
    
    
for name in ['ha', 'hb', 'mg2']:
    
    #Move fe2 files
    old_dir = '/data3/stone28/2drm/sdssrm_OLD/constants/fe2_' + name + '/rm160/'
    new_dir = '/data3/stone28/2drm/sdssrm/rm160/' + name + '/fe2/'
    for fname in glob.glob( old_dir + '*' ):
        shutil.copy( fname, new_dir )
        
    #Move qsofit files
    old_dir = '/data3/stone28/2drm/sdssrm_OLD/fit_res_' + name + '/rm160/'
    new_dir = '/data3/stone28/2drm/sdssrm/rm160/' + name + '/qsofit/'
    for fname in glob.glob( old_dir + '*.*' ):
        shutil.copy( fname, new_dir )
        
    for dirname in glob.glob(old_dir + '*/'):
        new_subdir = new_dir + dirname.split('/')[-2] + '/'
        os.makedirs(new_subdir, exist_ok=True)
        for fname in glob.glob( dirname + '*.*' ):
            shutil.copy( fname, new_subdir )
            
            
    
    #Move profile files
    old_dir = '/data3/stone28/2drm/sdssrm_OLD/line_profs2/rm160/' + name + '/'
    new_dir = '/data3/stone28/2drm/sdssrm/rm160/' + name + '/profile/'
    for fname in glob.glob( old_dir + '*' ):
        shutil.copy( fname, new_dir )
            
    