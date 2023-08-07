import numpy as np
from astropy.table import Table

import os
import glob
import gzip

from dmutils import utils

class Object:
    
    def __init__(self, rmid):
        
        self.rmid = int(rmid)
        self.main_dir = '/data3/stone28/2drm/sdssrm/rm{:03d}/'.format(rmid)

        self.raw_spec_dir = self.main_dir + 'raw_spec/'
        self.p0_filename = self.main_dir + 'p0t.dat'
        self.summary_filename = self.main_dir + 'summary.fits'



        #Get p0 data
        self.lnp0_dat = Table.read(self.p0_filename, format='ascii', names=['mjd', 'lnp0', 'err'] )
        
        #Get RA and DEC
        dat = Table.read(self.summary_filename)
        self.ra = dat[rmid-1]['RA']
        self.dec = dat[rmid-1]['DEC']
        del dat
            
        #Get filenames for raw spectra
        self.raw_spec_filenames = glob.glob(self.raw_spec_dir + '*')

        #Get header info
        self.mjd = np.zeros_like(self.raw_spec_filenames, dtype=float)
        z_arr = np.zeros_like(self.raw_spec_filenames, dtype=float)
        self.epochs = np.zeros_like(self.raw_spec_filenames, dtype=int)
        self.plateid = np.zeros_like(self.raw_spec_filenames, dtype=int)
        self.fiberid = np.zeros_like(self.raw_spec_filenames, dtype=int)
        
        
        self.table_arr = []
        for i in range(len(self.raw_spec_filenames)):
            
            with gzip.open(self.raw_spec_filenames[i], mode="rt") as f:
                file_content = f.readlines()
                self.mjd[i] = float( file_content[0].split()[1].split('=')[1] )
                z_arr[i] = float( file_content[2].split()[1].split('=')[1] )  
                self.epochs[i] = int( file_content[3][-3:] )
                
                self.plateid[i] = int( file_content[4].split()[1].split('=')[1]  )
                self.fiberid[i] = int( file_content[4].split()[3].split('=')[1] )
                
                colnames = file_content[5].split()[1:]

            
            dat = Table.read(self.raw_spec_filenames[i], format='ascii', names=colnames)
            self.table_arr.append(dat) 
        

        #Sort by epoch
        sort_ind = np.argsort(self.epochs)
        self.table_arr = np.array(self.table_arr, dtype=object)[sort_ind]
        self.raw_spec_filenames = np.array(self.raw_spec_filenames)[sort_ind]
        self.mjd = self.mjd[sort_ind]
        self.epochs = self.epochs[sort_ind]
        self.plateid = self.plateid[sort_ind]
        self.fiberid = self.fiberid[sort_ind]
        z_arr = z_arr[sort_ind]
        
        self.nepoch = len(self.epochs)

        
        #Get redshift
        assert np.all( z_arr == z_arr[0] )
        self.z = z_arr[0]
        
        #Get p0
        self.lnp0_dat['p0'] = np.exp(self.lnp0_dat['lnp0'].tolist())
        self.lnp0_dat['mjd'] = np.array(self.lnp0_dat['mjd']) + 50000
        self.lnp0_dat['spec_mjd'] = self.mjd
        self.lnp0_dat.sort('mjd')
        
        self.p0 = np.array(self.lnp0_dat['p0'])

        
        #Divide by p0(t)
        for i in range(len(self.table_arr)):
            self.table_arr[i]['corrected_flux'] = np.array(self.table_arr[i]['Flux']) / self.lnp0_dat['p0'][i]
            self.table_arr[i]['corrected_err'] = np.array(self.table_arr[i]['Flux_Err']) / self.lnp0_dat['p0'][i]

        self.res_fnames = {}


    def get_fe2_params(self, line_name):
        self.fe2_params = Table.read( self.main_dir + '/' + line_name '/fe2/best_fit_params.dat', format='ascii' )
        return


    def get_fit_res(self, line_name):
        
        res_path = self.main_dir + '/' + line_name + '/qsofit/'        
        epoch_dirs = glob.glob(res_path + 'rm{:03d}/*/'.format(self.rmid) )
        epochs = np.array([ int(d[-4:-1]) for d in epoch_dirs ])
        
        assert len(epoch_dirs) == len(self.table_arr) == self.nepoch

            
        fit_files = []
        cont_files = []
        for d in epoch_dirs:
            fit_files.append( glob.glob(d + '*.fits')[0] )
            cont_files.append( glob.glob( d + 'continuum*' )[0] )


        #Sort by epoch
        sort_ind = np.argsort(epochs)
        self.fit_res_files = np.array(fit_files)[sort_ind]
        self.fit_cont_files = np.array(cont_files)[sort_ind]
        
        return 



    def get_line_profile_fits(self, line_name):
        
        line_path = self.main_dir + '/' + line_name + '/profile/'
        fnames = glob.glob(line_path + '*')
        epochs = [ int(x.split('.')[0][-3:]) for x in fnames ]
        
        sort_ind = np.argsort(epochs)
        self.res_fnames[line_name] = np.array(fnames)[sort_ind]
        
        return




    def make_brains_input(self, line_name, output_fname, nbin=None, tol=5e-2):
        
        #NOTE: Need to call get_line_profile_fits() first
        
        if line_name == 'ha':
            central_wl = 6564.61
        elif line_name == 'hb':
            central_wl = 4862.721
        elif line_name == 'mg2':
            central_wl = 2798.75


        fnames = self.res_fnames[line_name]
            
        utils.make_input_file(fnames, central_wl, self.mjd, self.z, output_fname, nbin=nbin, tol=tol)
        self.line2d_filename = output_fname

        return
    
                
        