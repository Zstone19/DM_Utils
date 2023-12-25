import sys
import os
import shutil

import numpy as np

if sys.argv[1] == '1':
    from dmutils.specfit.object import Object
    from pypetal.fromfile.run_toml import make_toml, run_from_toml1
elif sys.argv[1] == '2':
    from pypetal_jav.run_toml import run_from_toml_jav
elif sys.argv[1] == '3':
    from pypetal.fromfile.run_toml import run_from_toml2


def run_pypetal(gemid, output_dir, lctype='raw',
                lag_bounds=None,
                npyccf=3000, npyroa=[20000,15000], 
                threads=1, verbose=True):




    lag_dir = '/data3/stone28/2drm/sdssrm/rm{:03d}/lag/'.format(gemid)

    #Run initial pipeline
    if sys.argv[1] == '1':
        obj = Object(gemid)
        lines = obj.get_line_names()
        if os.path.exists(lag_dir) & obj.nepoch == 1:
            shutil.rmtree(lag_dir)
            return


        line_names = ['cont']
        for name in lines:
            line_names.append(name)

        output_dir = lag_dir + 'pypetal_{}/'.format(lctype)
        arg2 = [ lag_dir + 'cont_lc.csv' ]


        for name in line_names[1:]:
            arg2.append(lag_dir + lctype + '_' + name + '_lc.csv')


        general_params = {'file_fmt':'ascii', 'lag_bounds':lag_bounds, 'threads':threads, 'verbose':verbose}
        drw_rej_params = {'use_for_javelin': True}
        pyccf_params = {'nsim': npyccf}
        pyroa_params = {'together': False, 'nchain': npyroa[0], 'nburn': npyroa[1], 'subtract_mean':False, 'delay_dist':False}

        #            General         DRW rej         Detrend  pyCCF         pyZDCF  PyROA         JAVELIN   Weighting
        run_arr =   [                True,          False,   True,         False,  True,         True,     True]
        param_arr = [general_params, drw_rej_params, {},      pyccf_params, {},     pyroa_params, {},       {}]


        toml_dict = make_toml(output_dir, arg2,
                              run_arr, param_arr,
                              line_names=line_names, filename=lag_dir + lctype +'_pypetal.toml')
        del toml_dict

        res = run_from_toml1(lag_dir + lctype + '_pypetal.toml')
        del res


    #Run JAVELIN
    if sys.argv[1] == '2':     
        res = run_from_toml_jav(lag_dir + lctype + '_pypetal.toml')
        del res


    #Run weighting
    if sys.argv[1] == '3':
        res = run_from_toml2(lag_dir + lctype + '_pypetal.toml')
        del res



    return





if __name__ == '__main__':

    if sys.argv[1] == '1':
        fname = 'pyccf'
    elif sys.argv[1] == '2':
        fname = 'jav'

    if sys.argv[1] in ['1', '2']:        
        if not os.path.exists( '/data3/stone28/2drm/sdssrm/bad_pp_runs_{}.txt'.format(fname) ):
            with open('/data3/stone28/2drm/sdssrm/bad_pp_runs_{}.txt'.format(fname), 'w+') as f:
                f.write('# GEMID \t LCtype \t Error\n')
        
    
    if not os.path.exists('/data3/stone28/2drm/sdssrm/bad_pp_runs_cont.txt' ):
        with open('/data3/stone28/2drm/sdssrm/bad_pp_runs_cont.txt', 'w+') as f:
            f.write('# GEMID \t LCtype \t Error\n')
            
    
    if not os.path.exists('/data3/stone28/2drm/sdssrm/bad_pp_runs_weight.txt' ):
        with open('/data3/stone28/2drm/sdssrm/bad_pp_runs_weight.txt', 'w+') as f:
            f.write('# GEMID \t LCtype \t Error\n')






    bad_ids = np.loadtxt('/data3/stone28/gem_targets/bad_pp_runs_weight.txt', unpack=True,
                         skiprows=1, usecols=[0], dtype=str)
    bad_ids = np.unique([int(x[-3:]) for x in bad_ids])
    print(bad_ids)





    # arr = range(240, 400)
    arr = bad_ids
    for i in arr:
        if not os.path.exists('/data3/stone28/gem_targets/GEM{:03d}/lag/'.format(i) ):
            continue

        try:
            xc, yc, yerrc = np.loadtxt('/data3/stone28/gem_targets/GEM{:03d}/lag/cont_lc.csv'.format(i), unpack=True, delimiter=',', skiprows=1)
            if type(xc) is not np.ndarray:
                with open('/data3/stone28/gem_targets/bad_pp_runs_cont.txt', 'a') as f:
                    f.write('GEM{:03d} \t {} \t {}\n'.format(i, 'cont', 'No data'))
                continue
        except Exception as e:
            with open('/data3/stone28/gem_targets/bad_pp_runs_cont.txt', 'a') as f:
                f.write('GEM{:03d} \t {} \t {}\n'.format(i, 'cont', e))
            
            continue


        if sys.argv[1] == '1':
            if os.path.exists('/data3/stone28/gem_targets/GEM{:03d}/lag/pypetal_raw/'.format(i) ):
                shutil.rmtree('/data3/stone28/gem_targets/GEM{:03d}/lag/pypetal_raw/'.format(i) )

            if os.path.exists('/data3/stone28/gem_targets/GEM{:03d}/lag/pypetal_model/'.format(i) ):
                shutil.rmtree('/data3/stone28/gem_targets/GEM{:03d}/lag/pypetal_model/'.format(i) )





        print('')
        print('=====================')
        print('Running GEM{:03d}'.format(i))
        print('=====================')
        print('')


        for dt in ['raw', 'model']:
            if not os.path.exists('/data3/stone28/gem_targets/GEM{:03d}/lag/'.format(i) ):
                continue

            print('{}'.format(dt))


            if sys.argv[1] == '3':
                try:
                    run_pypetal(i, '/data3/stone28/gem_targets/GEM{:03d}/lag/'.format(i), lctype=dt,
                                lag_bounds=None,
                                npyccf=5000, npyroa=[20000,15000], 
                                threads=63, verbose=False)
                except Exception as e:
                    with open('/data3/stone28/gem_targets/bad_pp_runs_weight.txt', 'a') as f:
                        f.write('GEM{:03d} \t {} \t {}\n'.format(i, dt, e))
                    
                    
                    continue

            else:
                try:
                    run_pypetal(i, '/data3/stone28/gem_targets/GEM{:03d}/lag/'.format(i), lctype=dt,
                                lag_bounds=None,
                                npyccf=5000, npyroa=[20000,15000], 
                                threads=63, verbose=False)
                except Exception as e:
                    with open('/data3/stone28/gem_targets/bad_pp_runs_{}.txt'.format(fname), 'a') as f:
                        f.write('GEM{:03d} \t {} \t {}\n'.format(i, dt, e))


                    continue                

