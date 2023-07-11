from astropy.table import Table
from astropy.io import fits
import numpy as np

import os
import glob
import gzip



###################################################################################################
# Utils for pyqsofit

def make_qsopar(path, ngauss_hb_br=2, fname='qsopar.fits', OIIItype='c'):

    """
    Create parameter file
    lambda    complexname  minwav maxwav linename ngauss inisca minsca maxsca inisig minsig maxsig voff vindex windex findex fvalue vary
    """

    recs = [(6564.61, r'H$\alpha$', 6400, 6800, 'Ha_br',   3, 0.1, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.015, 0, 0, 0, 0.05 , 1),
    (6564.61, r'H$\alpha$', 6400, 6800, 'Ha_na',   1, 0.1, 0.0, 1e10, 1e-3, 5e-4,   0.0017, 0.01,  1, 1, 0, 0.002, 1),
    (6549.85, r'H$\alpha$', 6400, 6800, 'NII6549', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 5e-3,  1, 1, 1, 0.001, 1),
    (6585.28, r'H$\alpha$', 6400, 6800, 'NII6585', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 5e-3,  1, 1, 1, 0.003, 1),
    (6718.29, r'H$\alpha$', 6400, 6800, 'SII6718', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 5e-3,  1, 1, 2, 0.001, 1),
    (6732.67, r'H$\alpha$', 6400, 6800, 'SII6732', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 5e-3,  1, 1, 2, 0.001, 1),

    (4862.68, r'H$\beta$', 4640, 5100, 'Hb_br',     ngauss_hb_br, 0.1, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.01, 0, 0, 0, 0.01 , 1),
    (4862.68, r'H$\beta$', 4640, 5100, 'Hb_na',     1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 0.01, 1, 1, 0, 0.002, 1),
    #(4687.02, r'H$\beta$', 4640, 5100, 'HeII4687_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.005, 0, 0, 0, 0.001, 1),
    #(4687.02, r'H$\beta$', 4640, 5100, 'HeII4687_na', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 0.005, 1, 1, 0, 0.001, 1),

    #(3934.78, 'CaII', 3900, 3960, 'CaII3934' , 2, 0.1, 0.0, 1e10, 1e-3, 3.333e-4, 0.0017, 0.01, 99, 0, 0, -0.001, 1),

    (3728.48, 'OII', 3650, 3800, 'OII3728', 1, 0.1, 0.0, 1e10, 1e-3, 3.333e-4, 0.0017, 0.01, 1, 1, 0, 0.001, 1),
        
    #(3426.84, 'NeV', 3380, 3480, 'NeV3426',    1, 0.1, 0.0, 1e10, 1e-3, 3.333e-4, 0.0017, 0.01, 0, 0, 0, 0.001, 1),
    #(3426.84, 'NeV', 3380, 3480, 'NeV3426_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.0025,   0.02,   0.01, 0, 0, 0, 0.001, 1),

    (2798.75, 'MgII', 2700, 2900, 'MgII_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 0, 0, 0, 0.05, 1),
    (2798.75, 'MgII', 2700, 2900, 'MgII_na', 2, 0.1, 0.0, 1e10, 1e-3, 5e-4, 0.0017, 0.01, 1, 1, 0, 0.002, 1),

    (1908.73, 'CIII', 1700, 1970, 'CIII_br',   2, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 99, 0, 0, 0.01, 1),
    #(1908.73, 'CIII', 1700, 1970, 'CIII_na',   1, 0.1, 0.0, 1e10, 1e-3, 5e-4,  0.0017, 0.01,  1, 1, 0, 0.002, 1),
    #(1892.03, 'CIII', 1700, 1970, 'SiIII1892', 1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.003, 1, 1, 0, 0.005, 1),
    #(1857.40, 'CIII', 1700, 1970, 'AlIII1857', 1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.003, 1, 1, 0, 0.005, 1),
    #(1816.98, 'CIII', 1700, 1970, 'SiII1816',  1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.0002, 1),
    #(1786.7,  'CIII', 1700, 1970, 'FeII1787',  1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.0002, 1),
    #(1750.26, 'CIII', 1700, 1970, 'NIII1750',  1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.001, 1),
    #(1718.55, 'CIII', 1700, 1900, 'NIV1718',   1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.001, 1),

    (1549.06, 'CIV', 1500, 1700, 'CIV_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05,   0.015, 0, 0, 0, 0.05 , 1),
    (1549.06, 'CIV', 1500, 1700, 'CIV_na', 1, 0.1, 0.0, 1e10, 1e-3, 5e-4,  0.0017, 0.01,  1, 1, 0, 0.002, 1),
    #(1640.42, 'CIV', 1500, 1700, 'HeII1640',    1, 0.1, 0.0, 1e10, 1e-3, 5e-4,   0.0017, 0.008, 1, 1, 0, 0.002, 1),
    #(1663.48, 'CIV', 1500, 1700, 'OIII1663',    1, 0.1, 0.0, 1e10, 1e-3, 5e-4,   0.0017, 0.008, 1, 1, 0, 0.002, 1),
    #(1640.42, 'CIV', 1500, 1700, 'HeII1640_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.0025, 0.02,   0.008, 1, 1, 0, 0.002, 1),
    #(1663.48, 'CIV', 1500, 1700, 'OIII1663_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.0025, 0.02,   0.008, 1, 1, 0, 0.002, 1),

    #(1402.06, 'SiIV', 1290, 1450, 'SiIV_OIV1', 1, 0.1, 0.0, 1e10, 5e-3, 0.002, 0.05,  0.015, 1, 1, 0, 0.05, 1),
    #(1396.76, 'SiIV', 1290, 1450, 'SiIV_OIV2', 1, 0.1, 0.0, 1e10, 5e-3, 0.002, 0.05,  0.015, 1, 1, 0, 0.05, 1),
    #(1335.30, 'SiIV', 1290, 1450, 'CII1335',   1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015, 0.01,  1, 1, 0, 0.001, 1),
    #(1304.35, 'SiIV', 1290, 1450, 'OI1304',    1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015, 0.01,  1, 1, 0, 0.001, 1),

    (1215.67, 'Lya', 1150, 1290, 'Lya_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05,   0.02, 0, 0, 0, 0.05 , 1),
    (1215.67, 'Lya', 1150, 1290, 'Lya_na', 1, 0.1, 0.0, 1e10, 1e-3, 5e-4,  0.0017, 0.01, 0, 0, 0, 0.002, 1)
    ]

    if OIIItype == 'c':
        recs.append( (4960.30, r'H$\beta$', 4640, 5100, 'OIII4959c', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 0.01, 1, 1, 0, 0.002, 1) )
        recs.append( (5008.24, r'H$\beta$', 4640, 5100, 'OIII5007c', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 0.01, 1, 1, 0, 0.004, 1) )
    elif OIIItype == 'w':
        recs.append( (4960.30, r'H$\beta$', 4640, 5100, 'OIII4959w',   1, 0.1, 0.0, 1e10, 3e-3, 2.3e-4, 0.004,  0.01,  2, 2, 0, 0.001, 1) )
        recs.append( (5008.24, r'H$\beta$', 4640, 5100, 'OIII5007w',   1, 0.1, 0.0, 1e10, 3e-3, 2.3e-4, 0.004,  0.01,  2, 2, 0, 0.002, 1) )



    newdata = np.rec.array( recs, 
    formats = 'float32,      a20,  float32, float32,      a20,  int32, float32, float32, float32, float32, float32, float32, float32,   int32,  int32,  int32,   float32, int32',
    names  =  ' lambda, compname,   minwav,  maxwav, linename, ngauss,  inisca,  minsca,  maxsca,  inisig,  minsig,  maxsig,  voff,     vindex, windex,  findex,  fvalue,  vary')


    # Header
    hdr = fits.Header()
    hdr['lambda'] = 'Vacuum Wavelength in Ang'
    hdr['minwav'] = 'Lower complex fitting wavelength range'
    hdr['maxwav'] = 'Upper complex fitting wavelength range'
    hdr['ngauss'] = 'Number of Gaussians for the line'

    # Can be set to negative for absorption lines if you want
    hdr['inisca'] = 'Initial guess of line scale [flux]'
    hdr['minsca'] = 'Lower range of line scale [flux]'
    hdr['maxsca'] = 'Upper range of line scale [flux]'

    hdr['inisig'] = 'Initial guess of linesigma [lnlambda]'
    hdr['minsig'] = 'Lower range of line sigma [lnlambda]'  
    hdr['maxsig'] = 'Upper range of line sigma [lnlambda]'

    hdr['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
    hdr['vindex'] = 'Entries w/ same NONZERO vindex constrained to have same velocity'
    hdr['windex'] = 'Entries w/ same NONZERO windex constrained to have same width'
    hdr['findex'] = 'Entries w/ same NONZERO findex have constrained flux ratios'
    hdr['fvalue'] = 'Relative scale factor for entries w/ same findex'

    hdr['vary'] = 'Whether or not to vary the line parameters (set to 0 to fix the line parameters to initial values)'

    # Save line info
    hdu = fits.BinTableHDU(data=newdata, header=hdr, name='data')
    hdu.writeto(os.path.join(path, fname), overwrite=True)
    
    return hdr, newdata



############################################################################################################
#Get input for and output of PyQSOFit

def get_spec_dat(rmid, spec_path, p0_path, summary_path, res_path=None, line_path=None):
    
    """Get the spectral data for a given object in the SDSSRM sample to be used for BRAINS. 
    
    Parameters
    ----------
    rmid : int
        The RMID of the object to be fit.
        
    spec_path : str
        The path to the directory containing the SDSSRM spectra.
        
    p0_path : str
        The path to the directory containing the p0 files.
        
    summary_path : str
        The path to the directory containing the summary.fits file.
        
    res_path : str, optional
        The path to the directory containing the results of the PyQSOFit emission line fits. If ``None``, 
        the files will not be read.
        
    line_path : str, optional
        The path to the directory containing the line files. If ``None``, the files will not be read.
    

    Returns:
    --------
    spec_prop : astropy.table.Table
        A table of properties for each spectrum for all epochs. This lists the MJD, redshift, epoch, as well as 
        the filenames for the spectra and (optionally) the fit emission lines
        
    table_arr: list
        A list of astropy.table.Table objects containing the spectral data for each epoch. The columns are
        wavelength, flux, flux error, corrected flux (using p0), and corrected error.
        
    ra: float
        The RA of the object.
        
    dec: float
        The DEC of the object.
    
    
    """
    
    spec_files = glob.glob(spec_path + 'RMID_{:03d}/*'.format(rmid) )
    lnp0_dat = Table.read(p0_path + 'rm{:03}/rm{:03}_p0_t.dat'.format(rmid,rmid), format='ascii', names=['mjd', 'lnp0', 'err'] )

    #Get RA and DEC
    dat = Table.read(summary_path + 'summary.fits')  
    ra = dat[rmid]['RA']
    dec = dat[rmid]['DEC']
    
    
    #Get header info
    mjd_arr = np.zeros_like(spec_files, dtype=float)
    z_arr = np.zeros_like(spec_files, dtype=float)
    epoch_arr = np.zeros_like(spec_files, dtype=int)
    plateid_arr = np.zeros_like(spec_files, dtype=int)
    fiberid_arr = np.zeros_like(spec_files, dtype=int)

    table_arr = []
    for i in range(len(spec_files)):
        
        with gzip.open(spec_files[i], mode="rt") as f:
            file_content = f.readlines()
            mjd_arr[i] = float( file_content[0].split()[1].split('=')[1] )
            z_arr[i] = float( file_content[2].split()[1].split('=')[1] )  
            epoch_arr[i] = int( file_content[3][-3:] )
            
            plateid_arr[i] = int( file_content[4].split()[1].split('=')[1]  )
            fiberid_arr[i] = int( file_content[4].split()[3].split('=')[1] )
            
            colnames = file_content[5].split()[1:]

        
        dat = Table.read(spec_files[i], format='ascii', names=colnames)
        table_arr.append(dat) 
    
    
    
    #Sort by epoch
    sort_ind = np.argsort(epoch_arr)
    table_arr = np.array(table_arr)[sort_ind]
    spec_files = np.array(spec_files)[sort_ind]

    mjd_arr = mjd_arr[sort_ind]
    z_arr = z_arr[sort_ind]
    epoch_arr = epoch_arr[sort_ind]
    plateid_arr = plateid_arr[sort_ind]
    fiberid_arr = fiberid_arr[sort_ind]
    
    
    #Get p0
    lnp0_dat['p0'] = np.exp(lnp0_dat['lnp0'].tolist())
    lnp0_dat['mjd'] = np.array(lnp0_dat['mjd']) + 50000
    lnp0_dat['spec_mjd'] = mjd_arr
    lnp0_dat.sort('mjd')
    
    
    #Divide by p0(t)
    for i in range(len(table_arr)):
        table_arr[i]['corrected_flux'] = np.array(table_arr[i]['Flux']) / lnp0_dat['p0'][i]
        table_arr[i]['corrected_err'] = np.array(table_arr[i]['Flux_Err']) / lnp0_dat['p0'][i]


    #Make an array for the properties of each spectrum
    spec_prop = Table([mjd_arr, epoch_arr, z_arr, plateid_arr, fiberid_arr, np.array(lnp0_dat['p0']), spec_files], 
                      names=['mjd', 'epoch', 'z', 'plateid', 'fiberid', 'p0', 'filename'])

    spec_prop.sort('mjd')
    
    
    
    if (res_path is not None) or (line_path is not None):
        fit_files, Hb_files, oIII_files, Ha_files, Mg2_files, cont_files, _ = get_fit_res(res_path, line_path, rmid)
        spec_prop['fit_file'] = fit_files
        spec_prop['Hb_file'] = Hb_files
        spec_prop['oIII_file'] = oIII_files
        spec_prop['Ha_file'] = Ha_files
        spec_prop['Mg2_file'] = Mg2_files
        spec_prop['cont_file'] = cont_files
    
    return spec_prop, table_arr, ra, dec




# Get the results of the pyQSOFit emission line fits (need to copy them from the original directory)
def get_fit_res(res_path, line_path, rmid):
    
    epoch_dirs = glob.glob(res_path + 'rm{:03d}/*/'.format(rmid) )
    epochs = np.array([ int(d[-4:-1]) for d in epoch_dirs ])
        
    fit_files = []
    oIII_files = []
    cont_files = []
    for d in epoch_dirs:
        fit_files.append( glob.glob(d + '*.fits')[0] )
        oIII_files.append( glob.glob(d + 'OIII*')[0] )
        cont_files.append( glob.glob( d + 'continuum*' )[0] )


    #Sort by epoch
    sort_ind = np.argsort(epochs)
    epochs = epochs[sort_ind]
    fit_files = np.array(fit_files)[sort_ind]
    oIII_files = np.array(oIII_files)[sort_ind]
    cont_files = np.array(cont_files)[sort_ind]
    
    
    
    
    Hb_files = glob.glob(line_path + 'rm{:03d}/hb/*.csv'.format(rmid) )
    Ha_files = glob.glob(line_path + 'rm{:03d}/ha/*.csv'.format(rmid) )
    Mg2_files = glob.glob(line_path + 'rm{:03d}/mg2/*.csv'.format(rmid) )
    epochs2 = np.array([ int(d[-4:-1]) for d in epoch_dirs ])
    
    #Sort by epoch
    sort_ind = np.argsort(epochs2)
    epochs2 = epochs2[sort_ind]
    Hb_files = np.array(Hb_files)[sort_ind]
    Ha_files = np.array(Ha_files)[sort_ind]
    Mg2_files = np.array(Mg2_files)[sort_ind]
    
    assert np.all(epochs == epochs2)
    
    return fit_files, Hb_files, oIII_files, Ha_files, Mg2_files, cont_files, epochs
        


############################################################################################################
#Get input for BRAINS

#Get the bounds of the emission line profile given the (unbinned) spectra
def get_prof_bounds(fnames, central_wl, tol=5e-2):

    left_bound = []
    right_bound = []
    
    for i in range(len(fnames)):
        ex_dat = Table.read(fnames[i])
        wl = ex_dat['wavelength'].tolist()
        prof = ex_dat['profile'].tolist()


        #Get boundaries    
        mid_ind = np.argmin( np.abs(np.array(wl) - central_wl) )
        bad_ind1 = np.argwhere( np.array(prof[:mid_ind]) < tol).T[0]
        bad_ind2 = np.argwhere( np.array(prof[mid_ind:]) < tol).T[0] + mid_ind

        if len(bad_ind1) == 0:
            left_ind = 0
        else:
            left_ind = np.max(bad_ind1)
        
        
        if len(bad_ind2) == 0:
            right_ind = len(wl) - 1
        else:
            right_ind = np.min(bad_ind2)
        
        left_bound.append(wl[left_ind])
        right_bound.append(wl[right_ind])    

    return np.min(left_bound), np.max(right_bound)





def make_input_file(fnames, central_wl, times, z, output_fname, nbin=None, tol=5e-2):
    
    #Wavelength is assumed to be in rest frame
    #Wavelength bins for each time are assumed not to be the same
    
    """Generate the input 2d line profile file for BRAINS. 
    This takes in a list of filenames for the 1d line profiles for each epoch of observation, 
    the central (rest-frame) wavelength for the line, the times of observation, 
    the redshift of the object, and the ouput filename. 
    NOTE: The wavelengths listed in the profile files should be in the rest-frame. 
    NOTE: The wavelengths in the output file will be in the observed-frame.
    
    The wavelengths bins for each epoch are not assumed to be the same. This will bin the spectra so they
    are all on the same wavelength grid. The general process is:
    - Create the wavelength grid
    - Bin the spectra onto this grid
    - If there are no points within a given bin, linearly interpolate the surrounding 5 datapoints
    - If the edge bins have no points within them, truncate the spectra to the first/last bin with points
    
    
    Parameters
    ----------
    fnames : list of strings
        List of filenames for the 1d line profiles for each epoch of observation
        
    central_wl : float
        Central wavelength of the line in the rest-frame
        
    times : list of floats
        List of times of observation
        
    z : float
        Redshift of the object
        
    output_fname : string
        Filename for the output file
        
    nbin : int, optional
        Number of wavelength bins to use. If not specified, will use the number of bins in the longest (i.e., most resolved) spectrum
    
    tol : float, optional
        Tolerance for determining the bounds of the line profile. If the profile dips below this value, it is considered to be outside the line profile.


    Returns
    -------
    None
    
    """
    
    wl_tot = []
    prof_tot = []
    err_tot = []
    bounds = get_prof_bounds(fnames, central_wl, tol=tol)

    for i in range(len(fnames)):
        hb_dat = Table.read(fnames[i], format='ascii.csv')
        
        wl_rest = hb_dat['wavelength']
        prof = hb_dat['profile']
        prof_err = (hb_dat['err_lo'] + hb_dat['err_hi'])/2    
        
        #Only use these bounds
        mask = (wl_rest > bounds[0]) & (wl_rest < bounds[1])
        wl_rest = np.array(wl_rest[mask])
        prof = np.array(prof[mask])
        prof_err = np.array(prof_err[mask])
        
        wl_tot.append(wl_rest)
        prof_tot.append(prof)
        err_tot.append(prof_err)


    #Go to observer frame
    for i in range(len(wl_tot)):
        wl_tot[i] = np.array(wl_tot[i]) * (1 + z)



    ####################################################
    #Need to rebin to a common wavelength grid
    if nbin is None:
        nbin = np.max( list(map(len, wl_tot))  )
    
    min_wl = np.min( list(map(np.min, wl_tot)) )
    max_wl = np.max( list(map(np.max, wl_tot)) )
    
    bin_centers = np.linspace(min_wl, max_wl, nbin )
    dlambda = bin_centers[1] - bin_centers[0]

    bin_edges = [bin_centers[0] - dlambda/2]
    for i in range(len(bin_centers)):
        bin_edges.append(bin_centers[i] + dlambda/2)



    wl_tot_rebin = np.zeros( (len(wl_tot), len(bin_centers)) )
    prof_tot_rebin = np.zeros( (len(wl_tot), len(bin_centers)) )
    err_tot_rebin = np.zeros( (len(wl_tot), len(bin_centers)) )



    prof_tot_bins = []
    err_tot_bins = []
    for i in range(len(wl_tot)):
        wl_tot_rebin[i,:] = bin_centers
        prof_tot_bins.append([])
        err_tot_bins.append([])
        
    for i in range(len(wl_tot)):
        for j in range(len(bin_centers)):
            prof_tot_bins[i].append([])
            err_tot_bins[i].append([])

    #Add values to the bins for each time/wavelength
    for n in range(len(wl_tot)):
        bin_ind = np.digitize(wl_tot[n], bin_edges)
            
        for j in range(len(bin_centers)):
            mask = (bin_ind-1 == j)
            
            prof_vals = prof_tot[n][mask]
            err_vals = err_tot[n][mask]

            prof_tot_bins[n][j] = np.concatenate( [ prof_tot_bins[n][j], prof_vals] )
            err_tot_bins[n][j] = np.concatenate( [ err_tot_bins[n][j], err_vals] )


    #Take the mean of the bins, if there are no values set to NaN
    for n in range(len(wl_tot)):
        for j in range(len(bin_centers)):
            
            if len(prof_tot_bins[n][j]) == 0:
                prof_tot_rebin[n,j] = np.nan
                err_tot_rebin[n,j] = np.nan
                continue

            
            mean_prof = np.mean(prof_tot_bins[n][j])
            err_prof = np.sqrt(np.sum(err_tot_bins[n][j]**2) + np.var(prof_tot_bins[n][j]))
                    
            prof_tot_rebin[n,j] = mean_prof
            err_tot_rebin[n,j] = err_prof
            
            
    
    #See if there are NaNs on the edges
    nan_ind = np.argwhere( np.isnan(prof_tot_rebin) )
    
    left_edge_nan = np.zeros( prof_tot_rebin.shape[0], dtype=bool )
    right_edge_nan = np.zeros( prof_tot_rebin.shape[0], dtype=bool )
    
    truncate_l = False
    truncate_r = False
    
    if len(nan_ind) == 0:
        pass
    else:        
        #Positions in the NaN indices array where the column index is 0 or the last column
        left_inds = np.argwhere( nan_ind[:,1] == 0 ).T[0]
        right_inds = np.argwhere( nan_ind[:,1] == prof_tot_rebin.shape[1]-1 ).T[0]
        
        #The row indices where this is true
        left_inds = np.unique( nan_ind[left_inds, 0] )
        right_inds = np.unique( nan_ind[right_inds, 0] )
        
        if len(left_inds) > 0:
            left_edge_nan[ left_inds ] = True
            truncate_l = True
        if len(right_inds) > 0:
            right_edge_nan[ right_inds ] = True    
            truncate_r = True



    #Truncate profile so there are no NaNs on the edges (on both sides)
    if truncate_l:
        new_left = 0
        
        for i in range(len(left_edge_nan)):
            if not left_edge_nan[i]:
                continue
            
            for j in range(prof_tot_rebin.shape[1]):
                if ( not np.isnan(prof_tot_rebin[i,j]) ):
                    
                    if j > new_left:
                        new_left = j
    
                    break
                    
        wl_tot_rebin = wl_tot_rebin[:, new_left:]
        prof_tot_rebin = prof_tot_rebin[:, new_left:]
        err_tot_rebin = err_tot_rebin[:, new_left:]


    if truncate_r:
        new_right = prof_tot_rebin.shape[1] - 1
        
        for i in range(len(left_edge_nan)):
            if not left_edge_nan[i]:
                continue
            
            for j in range(prof_tot_rebin.shape[1]-1, 0, -1):
                if ( not np.isnan(prof_tot_rebin[i,j]) ):
                    
                    if j < new_right:
                        new_left = j
    
                    break
        
        
        wl_tot_rebin = wl_tot_rebin[:, :new_right]
        prof_tot_rebin = prof_tot_rebin[:, :new_right]
        err_tot_rebin = err_tot_rebin[:, :new_right]



    #Linearly interpolate between points for wl bins with no data
    for i in range(len(wl_tot_rebin)):
        nan_ind = np.argwhere( np.isnan(prof_tot_rebin[i,:]) )
        
        if len(nan_ind) == 0:
            continue
        
        nfit = 5
        
        nan_ind = nan_ind.T[0]
        n = 0
        while n < len(nan_ind):
            n_interp = 1
            ind_lo = nan_ind[n] - 1

            m = nan_ind[n] + 1
            while m in nan_ind:
                n_interp += 1
                m += 1 

            ind_hi = m


            xfit = wl_tot_rebin[i, ind_lo-nfit:ind_hi+nfit]
            yfit = prof_tot_rebin[i, ind_lo-nfit:ind_hi+nfit]

            nan_mask = np.isnan(yfit)
            xfit = xfit[~nan_mask]
            yfit = yfit[~nan_mask]

            p = np.polyfit( xfit, yfit, 1)
            for j in range(1, n_interp+1):
                prof_tot_rebin[i, ind_lo+j] = np.polyval(p, wl_tot_rebin[i, ind_lo+j])
                err_tot_rebin[i, ind_lo+j] = np.sqrt( np.nanvar(prof_tot_rebin[i, ind_lo-nfit:ind_hi+nfit]) + np.nansum( err_tot_rebin[i, ind_lo-nfit:ind_hi+nfit]**2 ) )

            n += n_interp


    #Make sure there are no NaNs left
    assert np.all( np.isfinite(prof_tot_rebin) )
    
    #Write to file
    with open(output_fname, 'w+') as f:
        f.write('# {} {}\n'.format( wl_tot_rebin.shape[0], wl_tot_rebin.shape[1] ) )
        
        for i in range(wl_tot_rebin.shape[0]):
            f.write('# {:.5f}\n'.format(times[i]))
            
            for j in range(wl_tot_rebin.shape[1]):
                f.write('{:.6e} {:.6e} {:.6e}\n'.format(wl_tot_rebin[i,j], prof_tot_rebin[i,j], err_tot_rebin[i,j])  )
            
            f.write('\n')
            
            
            
    return

     
     
     
def read_input_file(fname):
        
    with open(fname, 'r') as f:
        lines = f.readlines()
        ne, nb = np.array( lines[0].split()[1:] ).astype(int)
            
        wl_tot = np.zeros((ne,nb))
        time_tot = np.zeros(ne)
        flux_tot = np.zeros((ne,nb))
        error_tot = np.zeros((ne,nb))

        time_ind = 1
        while len( lines[time_ind].split() ) == 0:
            time_ind += 1

        
        for i in range(ne):
            
            time_tot[i] =  float(lines[time_ind].split()[1])
            
            for j in range(0, nb):
                dat = np.array( lines[time_ind + 1 + j].split() ).astype(float)
                wl_tot[i,j] = dat[0]
                flux_tot[i,j] = dat[1]
                error_tot[i,j] = dat[2] 
                
                    
            time_ind += nb + 2
    
    
    return time_tot, wl_tot, flux_tot, error_tot




############################################################################################################
# Read BRAINS output files


def read_2d_tran(fname):
    
    with open(fname, 'r') as f:
        lines = f.readlines()
        ne, nb = np.array( lines[0].split()[1:] ).astype(int)
            
        wl_tot = np.zeros((ne,nb))
        time_tot = np.zeros(ne)
        flux_tot = np.zeros((ne,nb))

        time_ind = 1
        while len( lines[time_ind].split() ) == 0:
            time_ind += 1

        
        for i in range(ne):
            
            time_tot[i] =  float(lines[time_ind].split()[1])
            
            for j in range(0, nb):
                dat = np.array( lines[time_ind + 1 + j].split() ).astype(float)
                wl_tot[i,j] = dat[0]
                flux_tot[i,j] = dat[1]

                    
            time_ind += nb + 2

    return time_tot, wl_tot, flux_tot
