import os
import glob
import shutil

import numpy as np
import tqdm


###############################################################################################

#Get all RMIDs
main_rm_dir = '/data2/yshen/sdssrm/public/'

rm_dirs = glob.glob(main_rm_dir + 'rm*/')
rmids = [ int(x.split('/')[-2][2:]) for x in rm_dirs]

sort_ind = np.argsort(rmids)
rm_dirs = np.array(rm_dirs)[sort_ind]
rmids = np.array(rmids)[sort_ind]



#Get all line names (for each RMID)
line_dict = {}

for i, rm_dir in enumerate(rm_dirs):
    line_files = glob.glob(rm_dir + '*_lc.txt' )
    line_file_names = [os.path.basename(x) for x in line_files]
    
    line_file_names.remove('cont_lc.txt')
    line_names = [x.split('_')[0] for x in line_file_names]
    
    line_dict[ '{:03d}'.format(rmids[i]) ] = line_names



#Get p0t files
p0t_dir = '/data2/yshen/sdssrm/public/prepspec/'
p0t_fnames = []
for rmid in rmids:
    p0t_fnames.append( p0t_dir + 'rm{:03d}/rm{:03d}_p0_t.dat'.format(rmid, rmid) )


#Get raw spec dirs (need to download separately from Yue's FTP link)
main_raw_spec_dir = '/data3/stone28/2drm/sdssrm_OLD/spec/'
raw_spec_dirs = []
for rmid in rmids:
    raw_spec_dirs.append( main_raw_spec_dir + 'RMID_{:03d}/'.format(rmid) )


#Get continuum lcs
cont_lc_fnames = []
for rm_dir in rm_dirs:
    cont_lc_fnames.append( rm_dir + 'cont_lc.txt' )


###############################################################################################

#Make main directory
main_dir = '/data3/stone28/2drm/sdssrm/'
os.makedirs(main_dir, exist_ok=True)

#Copy summary file
shutil.copy( main_rm_dir + 'summary.fits', main_dir + 'summary.fits' )

#Make directory for each RMID
for rmid in rmids:
    os.makedirs(main_dir + 'rm{:03d}/'.format(rmid), exist_ok=True)
    
for i, rmid in enumerate(rmids):
        
    #Copy p0t file
    shutil.copy( p0t_fnames[i], main_dir + 'rm{:03d}/p0t.dat'.format(rmid) )
    
    #Copy continuum file
    shutil.copy( cont_lc_fnames[i], main_dir + 'rm{:03d}/cont_lc.dat'.format(rmid) )
    
    #Copy raw spec files
    os.makedirs( main_dir + 'rm{:03d}/raw_spec/'.format(rmid), exist_ok=True)
    for fname in glob.glob( raw_spec_dirs[i] + '*' ):
        shutil.copy( fname, main_dir + 'rm{:03d}/raw_spec/'.format(rmid) )

    #Make host flux subdir
    os.makedirs( main_dir + 'rm{:03d}/host_flux/'.format(rmid), exist_ok=True )    

    
    #Make line name subdirectories
    line_names = line_dict[ '{:03d}'.format(rmid) ]
    for name in line_names:
        line_subdir = main_dir + 'rm{:03d}/'.format(rmid) + name + '/'
        os.makedirs( line_subdir, exist_ok=True )
        
        #Fit subdir
        os.makedirs( line_subdir + 'qsofit/', exist_ok=True )
        
        #FeII subdir
        os.makedirs( line_subdir + 'fe2/', exist_ok=True )
        
        #Line profile subdir
        os.makedirs( line_subdir + 'profile/', exist_ok=True )
        
