import numpy as np
from astropy.table import Table

import glob
import gzip
import os

from dmutils import input

class Object:

    def __init__(self, rmid, main_dir='/data3/stone28/2drm/sdssrm/', processed=False, calibrate=None):

        self.rmid = int(rmid)
        self.main_dir = main_dir + 'rm{:03d}/'.format(rmid)

        self.raw_spec_dir = self.main_dir + 'raw_spec/'
        self.p0_filename = self.main_dir + 'p0t.dat'
        self.summary_filename = self.main_dir[:-6] + 'summary.fits'



        #Get p0 data
        self.lnp0_dat = Table.read(self.p0_filename, format='ascii', names=['mjd', 'lnp0', 'err'] )
        
        #Get RA and DEC
        dat = Table.read(self.summary_filename)
        self.ra = dat[rmid]['RA']
        self.dec = dat[rmid]['DEC']
        del dat

        #Get filenames for raw spectra
        self.raw_spec_filenames = glob.glob(self.raw_spec_dir + '*')

        #Get header info
        self.mjd = np.zeros_like(self.raw_spec_filenames, dtype=float)
        self.z_arr = np.zeros_like(self.raw_spec_filenames, dtype=float)
        self.epochs = np.zeros_like(self.raw_spec_filenames, dtype=int)
        self.plateid = np.zeros_like(self.raw_spec_filenames, dtype=int)
        self.fiberid = np.zeros_like(self.raw_spec_filenames, dtype=int)


        self.table_arr = []
        for i in range(len(self.raw_spec_filenames)):

            with gzip.open(self.raw_spec_filenames[i], mode="rt") as f:
                file_content = f.readlines()
                self.mjd[i] = float( file_content[0].split()[1].split('=')[1] )
                self.z_arr[i] = float( file_content[2].split()[1].split('=')[1] )  
                self.epochs[i] = int( file_content[3].split()[-1].split('=')[-1] )

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
        self.z_arr = self.z_arr[sort_ind]

        self.nepoch = len(self.epochs)


        #Get redshift
        if os.path.exists( self.main_dir + '/prepspec/dat/redshift_sdssrm{:04d}.dat'.format(self.rmid) ):
            self.z = float( np.loadtxt( self.main_dir + '/prepspec/dat/redshift_sdssrm{:04d}.dat'.format(self.rmid) ) )
        else:           
            self.z = np.median(self.z_arr)


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

        self.processed = False
        if processed:
            names = self.get_line_names()
            self.table_arr_og = self.table_arr.copy()

            self.table_arr = {}
            for name in names:
                self.table_arr[name] = []
                for i in range(self.nepoch):
                    self.table_arr[name].append( Table.read(self.main_dir + 'processed_spec/' + name + '/RM{:03d}_E{:03d}_{}_processed.csv'.format(self.rmid, i+1, name)) )


            self.processed = processed




        #Correct using OIII
        self.o3_corr = None
        self.calibrate = calibrate
        if self.calibrate is not None:
            self.use_oiii_corrections()



    def get_line_names(self):
        c4 = [1445, 1705]
        mg2 = [2700, 3090]
        hb = [4435, 5535]
        ha = [6400, 6800]
        ranges = [c4, mg2, hb, ha]
        names = ['c4', 'mg2', 'hb', 'ha']


        if self.processed:
            return list(self.table_arr.keys())

        else:
            wlmin = []
            wlmax = []
            for i in range(len(self.table_arr)):
                wl = np.array( self.table_arr[i]['Wave[vaccum]'] )/(1+self.z)
                wlmin.append( np.min(wl) )
                wlmax.append( np.max(wl) )

            wlmin = np.max(wlmin)
            wlmax = np.min(wlmax)


            line_names = []
            for i in range(len(ranges)):
                if len(ranges[i]) == 0:
                    continue

                if (wlmin < ranges[i][0]) & (wlmax > ranges[i][1]):
                    line_names.append(names[i])

            return line_names



    def get_fe2_params(self, line_name):
        self.fe2_params = Table.read( self.main_dir + '/' + line_name + '/fe2/best_fit_params.dat', format='ascii' )
        self.pl_params = Table.read( self.main_dir + '/' + line_name + '/fe2/best_fit_params_cont.dat', format='ascii' )
        return


    def use_oiii_corrections(self):
        
        if self.calibrate == 'PyQSOFit':
            if not os.path.exists(self.main_dir + 'calibrate/fit_prof/o3_corr.csv'):
                return

            self.o3_corr = Table.read(self.main_dir + 'calibrate/fit_prof/o3_corr.csv', format='ascii')
            self.o3_corr.sort('MJD')
        
            fluxcorr = self.o3_corr['FluxCorr']
            centercorr = self.o3_corr['CenterCorr']
            
        elif self.calibrate == 'PrepSpec':
            fluxcorr = self.p0
            centercorr = np.ones_like(self.p0)


        if self.processed:
            for line in self.table_arr.keys():
                for i in range(len(self.table_arr[line])):
                    self.table_arr[line][i]['Wave[vaccum]']  *= centercorr[i]
                    self.table_arr[line][i]['corrected_flux'] *= fluxcorr[i]
                    self.table_arr[line][i]['corrected_err'] *= fluxcorr[i]

        else:
            for i in range(len(self.table_arr)):
                self.table_arr[i]['Wave[vaccum]']  *= centercorr[i]
                self.table_arr[i]['corrected_flux'] *= fluxcorr[i]
                self.table_arr[i]['corrected_err'] *= fluxcorr[i]

        return


    def get_fit_res(self, line_name):

        res_path = self.main_dir + '/' + line_name + '/qsofit/'        
        epoch_dirs = glob.glob(res_path + 'rm{:03d}/*/'.format(self.rmid) )
        epochs = np.array([ int(d[-4:-1]) for d in epoch_dirs ])

        if self.processed:
            assert len(epoch_dirs) == len(self.table_arr[line_name]) == self.nepoch
        else:
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




    def make_brains_input(self, line_name, output_fname, 
                          nbin=None, bin_factor=None, dlambda=None, tol=5e-2, 
                          xmin=0, xmax=np.inf):

        #NOTE: Need to call get_line_profile_fits() first

        if line_name == 'ha':
            central_wl = 6564.61
            line_in = 'Ha_br'

        elif line_name == 'hb':
            central_wl = 4862.721
            line_in = 'Hb_br'

        elif line_name == 'mg2':
            central_wl = 2798.75
            line_in = 'MgII_br'

        elif line_name == 'c4':
            central_wl = 1550
            line_in = 'CIV_br'


        fnames = self.res_fnames[line_name]

        tol_fnames = []
        for i in range(self.nepoch):
            tol_fnames.append( self.main_dir + '/' + line_name + '/qsofit/epoch{:03d}/'.format(i+1) + line_in + '_profile.csv' )         
        tol_fnames = np.array(tol_fnames)


        mask = (self.mjd >= xmin) & (self.mjd <= xmax)
        
        if self.calibrate is None:
            ffactor = None
            cfactor = None
        elif self.calibrate == 'PyQSOFit':
            ffactor = self.o3_corr['FluxCorr'][mask].data
            cfactor = self.o3_corr['CenterCorr'][mask].data
        elif self.calibrate == 'PrepSpec':
            ffactor = self.p0[mask]
            cfactor = None
        

        input.make_input_file(fnames[mask], 
                             central_wl, self.mjd[mask], self.z, output_fname,
                             tol_fnames=tol_fnames[mask], tol_ffactor=ffactor, tol_cfactor=cfactor,
                             nbin=nbin, bin_factor=bin_factor, dlambda=dlambda,  tol=tol)
        self.line2d_filename = output_fname

        return
    