from dmutils.specfit.object import Object
from dmutils.specfit.fit_lines import make_qsopar, job
import numpy as np

from functools import partial
import os
import glob
import multiprocessing as mp



def run_all_fits(rmid, line_name, main_dir, prefix='', host=True, rej_abs_line=False, ncpu=None):
    
    res_dir = main_dir + 'rm{:03d}/'.format(rmid) + line_name + '/qsofit/'
    
    if host:
        host_dir = main_dir + 'rm{:03d}/host_flux/'.format(rmid)
    else:
        host_dir = None
    
    #Load data
    obj = Object(rmid, main_dir=main_dir)
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

#######################

main_dir = '/data3/stone28/2drm/sdssrm/'

bad_inds, bad_lines = np.loadtxt(main_dir + 'bad_run_const.txt', unpack=True, dtype=object)
bad_inds = bad_inds.astype(int)

# with open(main_dir + 'bad_run_qsofit.txt', 'w+') as f:
#     f.write('# GEMID\n')
    
# with open(main_dir + 'ran_newline.txt', 'w+') as f:
#     f.write('# GEMID \t line \n')



rmids = [160]


for rmid in rmids:

    if not os.path.exists(main_dir + 'GEM{:03d}/'.format(rmid)):
        with open(main_dir + 'cant_find.txt', 'a') as f:
            f.write('{:03d}'.format(rmid) + '\n')
        
        print('')
        print('-------------------')
        print('GEM{:03d} NOT FOUND'.format(rmid))
        print('-------------------')
        print('')
        
        continue


    print('')
    print('-------------------')
    print('FITTTING GEM{:03d}'.format(rmid))
    print('-------------------')
    print('')
    
    obj = Object(rmid, main_dir=main_dir)
    line_names = obj.get_line_names()

    for name in line_names:    
        
        if name in ['ha', 'hb']:
            host = True
        else:
            host = False
            
        if rmid in bad_inds:
            specific_inds = np.where(bad_inds == rmid)[0]
            specific_lines = bad_lines[specific_inds]
            
            if name in specific_lines:
                continue
                        
        try:
            run_all_fits(obj.rmid, name,
                    main_dir=main_dir,
                    host=host,
                    rej_abs_line=True,
                    prefix='')

        except Exception:
            #Retry 3 times

            success = False            
            for i in range(3):
                try:
                    run_all_fits(obj.rmid, name,
                        main_dir=main_dir,
                        host=host,
                        rej_abs_line=True,
                        prefix='')
                    
                    success = True
                except Exception:
                    continue

            if not success:
                with open(main_dir + 'bad_run_qsofit.txt', 'a') as f:
                    f.write('{:03d}'.format(rmid) + '\n')         

