import numpy as np
import multiprocessing as mp
import os
from functools import partial

from astropy.table import Table
from astropy.io import fits

from dmutils.specfit.host_contribution import remove_host_flux
from dmutils.specfit.object import Object

from pyqsofit.PyQSOFit import QSOFit


##############################################################################
############################### USEFUL FUNCTIONS #############################
##############################################################################

def make_qsopar_old(path, fname='qsopar.fits', oiii_wings=False):

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



    (4862.68, r'H$\beta$', 4640, 5100, 'Hb_br',    3, 0.1, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.01, 0, 0, 0, 0.01 , 1),
    (4862.68, r'H$\beta$', 4640, 5100, 'Hb_na',    1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 0.01, 1, 1, 0, 0.002, 1),
    (4960.30, r'H$\beta$', 4640, 5100, 'OIII4959c', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 0.01, 1, 1, 0, 0.002, 1),
    (5008.24, r'H$\beta$', 4640, 5100, 'OIII5007c', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 0.01, 1, 1, 0, 0.004, 1),
    # (4862.68, r'H$\beta$', 4640., 5100.,'Hb_br',     3, 0.1, 0.0, 1e10, 5e-3, 0.002,  0.05,  0.01,   0,  0,  0,  0.01 , 1), #Qiaoya's
    # (4862.68, r'H$\beta$', 4640., 5100.,'Hb_na',     1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.002, 0.01,   1,  1,  0,  0.002, 1),
    (4687.02, r'H$\beta$', 4640, 5100, 'HeII4687_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.005, 0, 0, 0, 0.001, 1),
    (4687.02, r'H$\beta$', 4640, 5100, 'HeII4687_na', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 0.005, 1, 1, 0, 0.001, 1),


    #(3934.78, 'CaII', 3900, 3960, 'CaII3934' , 2, 0.1, 0.0, 1e10, 1e-3, 3.333e-4, 0.0017, 0.01, 99, 0, 0, -0.001, 1),

    (3728.48, 'OII', 3650, 3800, 'OII3728', 1, 0.1, 0.0, 1e10, 1e-3, 3.333e-4, 0.0017, 0.01, 1, 1, 0, 0.001, 1),

    #(3426.84, 'NeV', 3380, 3480, 'NeV3426',    1, 0.1, 0.0, 1e10, 1e-3, 3.333e-4, 0.0017, 0.01, 0, 0, 0, 0.001, 1),
    #(3426.84, 'NeV', 3380, 3480, 'NeV3426_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.0025,   0.02,   0.01, 0, 0, 0, 0.001, 1),

    # (2798.75, 'MgII', 2700, 2900, 'MgII_br', 2, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 0, 0, 0, 0.05, 1),
    # (2798.75, 'MgII', 2700, 2900, 'MgII_na', 1, 0.1, 0.0, 1e10, 1e-3, 5e-4, 0.0017, 0.01, 1, 1, 0, 0.002, 1),
    (2798.75, 'MgII', 2700., 2900., 'MgII_br', 2, 0.1, 0.0, 1e10, 5e-3, 0.002, 0.05, 0.0015, 0, 0, 0, 0.05, 1),  #Qiaoya's
    (2798.75, 'MgII', 2700., 2900., 'MgII_na', 1, 0.1, 0.0, 1e10, 1e-3, 5e-4, 0.002, 0.01,   1, 1, 0, 0.002, 1),

    (1908.73, 'CIII', 1700, 1970, 'CIII_br',   2, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 99, 0, 0, 0.01, 1),
    (1908.73, 'CIII', 1700, 1970, 'CIII_na',   1, 0.1, 0.0, 1e10, 1e-3, 5e-4,  0.0017, 0.01,  1, 1, 0, 0.002, 1),
    #(1892.03, 'CIII', 1700, 1970, 'SiIII1892', 1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.003, 1, 1, 0, 0.005, 1),
    #(1857.40, 'CIII', 1700, 1970, 'AlIII1857', 1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.003, 1, 1, 0, 0.005, 1),
    #(1816.98, 'CIII', 1700, 1970, 'SiII1816',  1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.0002, 1),
    #(1786.7,  'CIII', 1700, 1970, 'FeII1787',  1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.0002, 1),
    #(1750.26, 'CIII', 1700, 1970, 'NIII1750',  1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.001, 1),
    #(1718.55, 'CIII', 1700, 1900, 'NIV1718',   1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.001, 1),

    (1549.06, 'CIV', 1500, 1700, 'CIV_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05,   0.015, 0, 0, 0, 0.05 , 1),
    (1549.06, 'CIV', 1500, 1700, 'CIV_na', 1, 0.1, 0.0, 1e10, 1e-3, 5e-4,  0.0017, 0.01,  1, 1, 0, 0.002, 1),
    (1640.42, 'CIV', 1500, 1700, 'HeII1640',    1, 0.1, 0.0, 1e10, 1e-3, 5e-4,   0.0017, 0.008, 1, 1, 0, 0.002, 1),
    (1663.48, 'CIV', 1500, 1700, 'OIII1663',    1, 0.1, 0.0, 1e10, 1e-3, 5e-4,   0.0017, 0.008, 1, 1, 0, 0.002, 1),
    (1640.42, 'CIV', 1500, 1700, 'HeII1640_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.0025, 0.02,   0.008, 1, 1, 0, 0.002, 1),
    (1663.48, 'CIV', 1500, 1700, 'OIII1663_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.0025, 0.02,   0.008, 1, 1, 0, 0.002, 1),

    #(1402.06, 'SiIV', 1290, 1450, 'SiIV_OIV1', 1, 0.1, 0.0, 1e10, 5e-3, 0.002, 0.05,  0.015, 1, 1, 0, 0.05, 1),
    #(1396.76, 'SiIV', 1290, 1450, 'SiIV_OIV2', 1, 0.1, 0.0, 1e10, 5e-3, 0.002, 0.05,  0.015, 1, 1, 0, 0.05, 1),
    #(1335.30, 'SiIV', 1290, 1450, 'CII1335',   1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015, 0.01,  1, 1, 0, 0.001, 1),
    #(1304.35, 'SiIV', 1290, 1450, 'OI1304',    1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015, 0.01,  1, 1, 0, 0.001, 1),

    (1215.67, 'Lya', 1150, 1290, 'Lya_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05,   0.02, 0, 0, 0, 0.05 , 1),
    (1215.67, 'Lya', 1150, 1290, 'Lya_na', 1, 0.1, 0.0, 1e10, 1e-3, 5e-4,  0.0017, 0.01, 0, 0, 0, 0.002, 1)
    ]

    if oiii_wings:
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



def make_qsopar(path, fname='qsopar.fits', oiii_wings=True):    
    """
    Create parameter file
    lambda    complexname  minwav maxwav linename     ngauss inisca minsca maxsca inisig minsig    maxsig  inidw mindw   maxdw  vindex windex findex fvalue varysca, varysig, varydw
    """

    recs = [
    (6564.61, r'H$\alpha$', 6400, 6800, 'Ha_br',      3,     0.1,   0.0,   1e10,  5e-3,  0.004,    0.05,   0.00, -0.015, 0.015, 0,     0,     0,    0.05 , 1,        1,       1),
    (6564.61, r'H$\alpha$', 6400, 6800, 'Ha_na',      1,     0.1,   0.0,   1e10,  1e-3,  5e-4,     0.0017, 0.00, -0.01,  0.01,  1,     1,     0,    0.002, 1,        1,       1),
    (6549.85, r'H$\alpha$', 6400, 6800, 'NII6549',    1,     0.1,   0.0,   1e10,  1e-3,  2.3e-4,   0.0017, 0.00, -5e-3,  5e-3,  1,     1,     1,    0.001, 1,        1,       1),
    (6585.28, r'H$\alpha$', 6400, 6800, 'NII6585',    1,     0.1,   0.0,   1e10,  1e-3,  2.3e-4,   0.0017, 0.00, -5e-3,  5e-3,  1,     1,     1,    0.003, 1,        1,       1),
    (6718.29, r'H$\alpha$', 6400, 6800, 'SII6718',    1,     0.1,   0.0,   1e10,  1e-3,  2.3e-4,   0.0017, 0.00, -5e-3,  5e-3,  1,     1,     2,    0.001, 1,        1,       1),
    (6732.67, r'H$\alpha$', 6400, 6800, 'SII6732',    1,     0.1,   0.0,   1e10,  1e-3,  2.3e-4,   0.0017, 0.00, -5e-3,  5e-3,  1,     1,     2,    0.001, 1,        1,       1),

    (4862.68, r'H$\beta$', 4640, 5100, 'Hb_br',       3,     0.1,   0.0,   1e10,  5e-3,  0.004,    0.05,   0.00, -0.01,  0.01,  0,     0,     0,    0.01 , 1,        1,       1),
    (4862.68, r'H$\beta$', 4640, 5100, 'Hb_na',       1,     0.1,   0.0,   1e10,  1e-3,  2.3e-4,   0.0017, 0.00, -0.01,  0.01,  1,     1,     0,    0.002, 1,        1,       1),
    (4960.30, r'H$\beta$', 4640, 5100, 'OIII4959c',   1,     0.1,   0.0,   1e10,  1e-3,  2.3e-4,   0.0017, 0.00, -0.01,  0.01,  1,     1,     1,    0.333, 1,        1,       1),
    (5008.24, r'H$\beta$', 4640, 5100, 'OIII5007c',   1,     0.1,   0.0,   1e10,  1e-3,  2.3e-4,   0.0017, 0.00, -0.01,  0.01,  1,     1,     1,    1.000, 1,        1,       1),
    (4687.02, r'H$\beta$', 4640, 5100, 'HeII4687_br', 1,     0.1,   0.0,   1e10,  5e-3,  0.004,    0.05,   0.00  -0.005, 0.005, 0,     0,     0,    0.001, 1,        1,       1),
    (4687.02, r'H$\beta$', 4640, 5100, 'HeII4687_na', 1,     0.1,   0.0,   1e10,  1e-3,  2.3e-4,   0.0017, 0.00  -0.005, 0.005, 1,     1,     0,    0.001, 1,        1,       1),

    (3728.48, 'OII',       3650, 3800, 'OII3728',     1,     0.1,   0.0,   1e10,  1e-3,  3.333e-4, 0.0017, 0.00  -0.01,  0.01,  1,     1,     0,    0.001, 1,        1,       1),

    (2798.75, 'MgII',      2700, 2900, 'MgII_br',     2,     0.1,   0.0,   1e10,  5e-3,  0.004,    0.05,   0.00  -0.015, 0.015, 0,     0,     0,    0.05,  1,        1,       1),
    (2798.75, 'MgII',      2700, 2900, 'MgII_na',     1,     0.1,   0.0,   1e10,  1e-3,  5e-4,     0.0017, 0.00  -0.01,  0.01,  1,     1,     0,    0.002, 1,        1,       1),

    (1908.73, 'CIII',      1700, 1970, 'CIII_br',     2,     0.1,   0.0,   1e10,  5e-3,  0.004,    0.05,   0.00  -0.015, 0.015, 99,    0,     0,    0.01,  1,        1,       1),
    (1908.73, 'CIII',      1700, 1970, 'CIII_na',     1,     0.1,   0.0,    1e10, 1e-3,  5e-4,     0.0017, 0.00, -0.01,  0.01,  1,     1,     0,    0.002, 1,        1,       1),
 
    (1549.06, 'CIV',       1500, 1700, 'CIV_br',      1,     0.1,   0.0,   1e10,  5e-3,  0.004,    0.05,   0.00  -0.015, 0.015, 0,     0,     0,    0.05 , 1,        1,       1),
    (1549.06, 'CIV',       1500, 1700, 'CIV_na',      1,     0.1,   0.0,   1e10,  1e-3,  5e-4,     0.0017, 0.00  -0.01,  0.01,  1,     1,     0,    0.002, 1,        1,       1),
    (1640.42, 'CIV',       1500, 1700, 'HeII1640',    1,     0.1,   0.0,   1e10,  1e-3,  5e-4,     0.0017, 0.00  -0.008, 0.008, 1,     1,     0,    0.002, 1,        1,       1),
    (1663.48, 'CIV',       1500, 1700, 'OIII1663',    1,     0.1,   0.0,   1e10,  1e-3,  5e-4,     0.0017, 0.00  -0.008, 0.008, 1,     1,     0,    0.002, 1,        1,       1),
    (1640.42, 'CIV',       1500, 1700, 'HeII1640_br', 1,     0.1,   0.0,   1e10,  5e-3,  0.0025,   0.02,   0.00  -0.008, 0.008, 1,     1,     0,    0.002, 1,        1,       1),
    (1663.48, 'CIV',       1500, 1700, 'OIII1663_br', 1,     0.1,   0.0,   1e10,  5e-3,  0.0025,   0.02,   0.00  -0.008, 0.008, 1,     1,     0,    0.002, 1,        1,       1),

    (1215.67, 'Lya',       1150, 1290, 'Lya_br',      1,     0.1,   0.0,   1e10,  5e-3,  0.004,    0.05,   0.00  -0.02,  0.02,  0,     0,     0,    0.05 , 1,        1,       1),
    (1215.67, 'Lya',       1150, 1290, 'Lya_na',      1,     0.1,   0.0,   1e10,  1e-3,  5e-4,     0.0017, 0.00  -0.01,  0.01,  0,     0,     0,    0.002, 1,        1,       1)]

    if oiii_wings:
        recs.append( (4960.30, r'H$\beta$', 4640, 5100, 'OIII4959w',   1,     0.1,   0.0,   1e10,  3e-3,  2.3e-4, 0.004,  0.00, -0.01,  0.01,  2,     2,     0,    0.001, 1,        1,       1) )
        recs.append( (5008.24, r'H$\beta$', 4640, 5100, 'OIII5007w',   1,     0.1,   0.0,   1e10,  3e-3,  2.3e-4, 0.004,  0.00, -0.01,  0.01,  2,     2,     0,    0.002, 1,        1,       1) )


    newdata = np.rec.array( recs,
                            formats = 'float32, a20,      float32, float32, a20,      int32,  float32, float32, float32, float32, float32, float32, float32, float32, float32, int32, int32,  int32, float32, int32,   int32,  int32',
                            names  =  'lambda, compname, minwav, maxwav, linename, ngauss, inisca, minsca, maxsca, inisig, minsig, maxsig, inidw,  mindw,  maxdw, vindex, windex, findex, fvalue, varysca, varsig, varydw')



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

    hdr['inidw'] = 'Initial guess of velocity offset [lnlambda]'
    hdr['mindw'] = 'Lower range of velocity offset [lnlambda]'
    hdr['maxdw'] = 'Upper range of velocity offset [lnlambda]'
    
    hdr['vindex'] = 'Entries w/ same NONZERO vindex constrained to have same velocity'
    hdr['windex'] = 'Entries w/ same NONZERO windex constrained to have same width'
    hdr['findex'] = 'Entries w/ same NONZERO findex have constrained flux ratios'
    hdr['fvalue'] = 'Relative scale factor for entries w/ same findex'

    hdr['varysca'] = 'Whether or not to vary the line scales (set to 0 to fix the line parameters to initial values)'
    hdr['varysig'] = 'Whether or not to vary the line width (set to 0 to fix the line parameters to initial values)'
    hdr['varydw'] = 'Whether or not to vary the line offsets (set to 0 to fix the line parameters to initial values)'

    # Save line info
    hdu = fits.BinTableHDU(data=newdata, header=hdr, name='data')
    hdu.writeto(os.path.join(path, fname), overwrite=True)

    return hdr, newdata



#NEED TO REWORK THIS !!!!!!!!!!!!!!!!
def find_optimal_ngauss(lam, flux, err, z, ra, dec, mjd, fitpath, 
                        qsopar_header, qsopar_dat, line_name='hb', 
                        ngauss_max=7, bic_tol=10):
    
    if line_name == 'ha':
        line_str = 'Ha_br'
    elif line_name == 'hb':
        line_str = 'Hb_br'
    elif line_name == 'mg2':
        line_str = 'MgII_br'
    elif line_name == 'c4':
        line_str = 'CIV_br'
    
    
    bic_last = np.inf
    for ngauss in range(1, ngauss_max):
        
        print(fr'Fitting {line_name} with {ngauss} components.')
        
        newdata_n = qsopar_dat.copy()
        for i, row in enumerate(newdata_n):
            if line_str in str(row['linename']):
                newdata_n[i]['ngauss'] = ngauss

        hdu = fits.BinTableHDU(data=newdata_n, header=qsopar_header, name='data')
        hdu.writeto(os.path.join(fitpath, 'qsopar.fits'), overwrite=True)
        
        
        q = QSOFit(lam, flux, err, z, ra=ra, dec=dec, mjd=int(mjd), path=fitpath)
    
        q.Fit(name=None, nsmooth=1, deredden=True, reject_badpix=False, wave_range=None, wave_mask=None, 
                decompose_host=True, npca_gal=5, npca_qso=20, 
                Fe_uv_op=True, poly=True, rej_abs_conti=False,
                MCMC=True, epsilon_jitter=1e-4, nburn=100, nsamp=200, nthin=10, linefit=True, 
                save_result=False, plot_fig=False, save_fig=False, plot_corner=False,
                verbose=False)
    
    
        if line_name == 'Ha':
            mask_bic = (q.line_result_name == '2_line_min_chi2')
        elif line_name == 'Hb':
            mask_bic = (q.line_result_name == '1_line_min_chi2')
        
        bic = float(q.line_result[mask_bic][0])
        print('Delta BIC: {:.1f}'.format(bic_last - bic) )
        
        if np.abs(bic_last - bic) < bic_tol:
            ngauss_best = ngauss - 1
            break
    
        bic_last = bic
    
    return ngauss_best




def check_bad_run(qi, line_name):
    
    rerun = False

    if line_name is None:
        #Get chi2 of Hbeta
        c = np.argwhere( qi.uniq_linecomp_sort == 'H$\\beta$' ).T[0][0]
        chi2_nu1 = float(qi.comp_result[c*7+4])
        
        #Get scale of OIII4959
        names = qi.line_result_name
        oiii_mask = (names == 'OIII4959c_1_scale')
        oiii_scale = float(qi.line_result[oiii_mask])
            
        #Get chi2 of MgII
        c = np.argwhere( qi.uniq_linecomp_sort == 'MgII' ).T[0][0]
        chi2_nu2 = float(qi.comp_result[c*7+4])
        
        #Get chi2 of CIV
        c = np.argwhere( qi.uniq_linecomp_sort == 'CIV' ).T[0][0]
        chi2_nu3 = float(qi.comp_result[c*7+4])
        
        #Get chi2 of Halpha
        c = np.argwhere( qi.uniq_linecomp_sort == 'H$\\alpha$' ).T[0][0]
        chi2_nu4 = float(qi.comp_result[c*7+4])        
        
        
        if (chi2_nu1 > 3) or (chi2_nu2 > 3) or (oiii_scale < 1) or (chi2_nu3 > 3) or (chi2_nu4 > 3):
            rerun = True


    elif line_name == 'hb':
        #Get chi2 of Hbeta
        c = np.argwhere( qi.uniq_linecomp_sort == 'H$\\beta$' ).T[0][0]
        chi2_nu1 = float(qi.comp_result[c*7+4])
        
        #Get scale of OIII4959
        names = qi.line_result_name
        oiii_mask = (names == 'OIII4959c_1_scale')
        oiii_scale = float(qi.line_result[oiii_mask])
        
        if (chi2_nu1 > 3) or (oiii_scale < 1):
            rerun = True

        
    elif line_name == 'mg2':
        #Get chi2 of MgII
        c = np.argwhere( qi.uniq_linecomp_sort == 'MgII' ).T[0][0]
        chi2_nu2 = float(qi.comp_result[c*7+4])
        
        if chi2_nu2 > 3:
            rerun = True
            
    elif line_name == 'c4':
        #Get chi2 of CIV
        c = np.argwhere( qi.uniq_linecomp_sort == 'CIV' ).T[0][0]
        chi2_nu3 = float(qi.comp_result[c*7+4])
        
        if chi2_nu3 > 3:
            rerun = True
        

    elif line_name == 'ha':
        c = np.argwhere( qi.uniq_linecomp_sort == 'H$\\alpha$' ).T[0][0]
        chi2_nu4 = float(qi.comp_result[c*7+4])
        
        if chi2_nu4 > 3:
            rerun = True

    return rerun




###############################################################################################
###############################################################################################
###############################################################################################


def run_pyqsofit(obj, ind, output_dir, qsopar_dir, line_name=None, prefix='', host_dir=None, rej_abs_line=False):
    print('Fitting epoch {}'.format(ind+1))

    if line_name not in ['mg2', 'c4']:
        assert host_dir is not None, 'host_dir must be specified for non-MgII lines.'        

    lam = np.array(obj.table_arr[ind]['Wave[vaccum]'])
    flux = np.array(obj.table_arr[ind]['corrected_flux'])
    err = np.array(obj.table_arr[ind]['corrected_err'])
    
    and_mask = np.array(obj.table_arr[ind]['ANDMASK'])
    or_mask = np.array(obj.table_arr[ind]['ORMASK'])

    epoch = obj.epochs[ind]
    mjd = obj.mjd[ind]
    plateid = obj.plateid[ind]
    fiberid = obj.fiberid[ind]
    
    
    decompose_host = True
    if line_name is None:
        wave_range = None
    elif line_name == 'mg2':
        wave_range = np.array([2200, 3090])
        decompose_host = False
        center = 2798
    elif line_name == 'c4':
        wave_range = np.array([1445, 1705])
        decompose_host = False
        center = 1549
    elif line_name == 'hb':
        wave_range = np.array([4435, 5535])
        center = 4861
    elif line_name == 'ha':
        wave_range = np.array([6100, 7000])
        center = 6563

    
    
    if line_name in ['mg2', 'c4']:
        fe_uv_params = np.array( list(obj.fe2_params[ind]) )[[1,3,5]]
        fe_op_params = None
    else:
        fe_uv_params = np.array( list(obj.fe2_params[ind]) )[[1,3,5]]
        fe_op_params = np.array( list(obj.fe2_params[ind]) )[[7,9,11]]

    if decompose_host:
        flux, lam, err, and_mask, or_mask = remove_host_flux(lam, flux, err, and_mask, or_mask,
                                                      host_dir + 'best_host_flux.dat', 
                                                      z=obj.z)
        

    poly = True
    if line_name is not None:
        poly = False
        
    masks = True        
    nburn = 100
    nsamp = 200
    nthin = 10
    
    #Don't use masks if they remove the line
    if line_name is not None:
        new_lam = lam.copy()
        mask_ind = np.where( (and_mask == 0) & (and_mask == 0) , True, False)
        new_lam = new_lam[mask_ind]
        
        if new_lam[0] > center:
            masks = False

    

    name = 'RM{:03d}e{:03d}'.format(obj.rmid, epoch) + prefix    
        
    try:
        qi = QSOFit(lam, flux, err, obj.z, ra=obj.ra, dec=obj.dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                    and_mask_in=and_mask, or_mask_in=or_mask)
        
        qi.Fit(name=name, nsmooth=1, deredden=True, 
                and_mask=masks, or_mask=masks,
            reject_badpix=False, wave_range=wave_range, wave_mask=None, 
            decompose_host=False, npca_gal=5, npca_qso=20, 
            Fe_uv_op=True, poly=poly,
            rej_abs_conti=False, rej_abs_line=rej_abs_line,
            MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=True, 
            Fe_uv_fix=fe_uv_params, Fe_op_fix=fe_op_params,
            save_result=True, plot_fig=True, save_fig=True, plot_corner=False, kwargs_plot={'save_fig_path':output_dir}, 
            save_fits_name=name+'_pyqsofit', save_fits_path=output_dir, verbose=False)
    except:
        masks = False
        
        qi = QSOFit(lam, flux, err, obj.z, ra=obj.ra, dec=obj.dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir,
                    and_mask_in=and_mask, or_mask_in=or_mask)
        
        qi.Fit(name=name, nsmooth=1, deredden=True, 
                and_mask=masks, or_mask=masks,
            reject_badpix=False, wave_range=wave_range, wave_mask=None, 
            decompose_host=False, npca_gal=5, npca_qso=20, 
            Fe_uv_op=True, poly=poly,
            rej_abs_conti=False, rej_abs_line=rej_abs_line,
            MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=True, 
            Fe_uv_fix=fe_uv_params, Fe_op_fix=fe_op_params,
            save_result=True, plot_fig=True, save_fig=True, plot_corner=False, kwargs_plot={'save_fig_path':output_dir}, 
            save_fits_name=name+'_pyqsofit', save_fits_path=output_dir, verbose=False)


    rerun1 = check_bad_run(qi, line_name)
    rerun = rerun1

    #If chi2nu is too high or it doesn't fit [OIII]4959, rerun a couple of times
    n = 0
    while rerun:

        n += 1
        qi = QSOFit(lam, flux, err, obj.z, ra=obj.ra, dec=obj.dec, plateid=plateid, mjd=int(mjd), fiberid=fiberid, path=qsopar_dir)

        qi.Fit(name=name, nsmooth=1, deredden=True, 
               and_mask=masks, or_mask=masks,
            reject_badpix=False, wave_range=wave_range, wave_mask=None, 
            decompose_host=False, npca_gal=5, npca_qso=20, 
            Fe_uv_op=True, poly=poly, 
            rej_abs_conti=False, rej_abs_line=rej_abs_line,
            MCMC=True, epsilon_jitter=1e-4, nburn=nburn, nsamp=nsamp, nthin=nthin, linefit=True, 
            Fe_uv_fix=fe_uv_params, Fe_op_fix=fe_op_params,
            save_result=True, plot_fig=True, save_fig=True, plot_corner=False, kwargs_plot={'save_fig_path':output_dir}, 
            save_fits_name=name+'_pyqsofit', save_fits_path=output_dir, verbose=False)


        rerun = check_bad_run(qi, line_name)

        if n > 5:
            break



    if rerun1:
        with open(qsopar_dir + 'rerun.txt', 'a') as f:
            f.write('{},{}\n'.format(n, epoch))

    if rerun:
        with open(qsopar_dir + 'bad_run.txt', 'a') as f:
            f.write('{}\n'.format(epoch))


    return qi


#Job function (per epoch)
def job(ind, obj, res_dir, line_name=None, prefix='', host_dir=None, rej_abs_line=False):

    epoch = obj.epochs[ind]    
    epoch_dir = res_dir + 'epoch{:03d}/'.format(epoch)
    qi = run_pyqsofit(obj, ind, epoch_dir, res_dir, line_name=line_name, prefix=prefix, 
                      host_dir=host_dir, rej_abs_line=rej_abs_line)
    
    #Get line fitting results
    gauss_result_tot = qi.gauss_result_all
    gauss_result = qi.gauss_result[::2]
    gauss_names = qi.gauss_result_name[::2]


    ####################################################################
    ####################################################################
    # Hbeta

    if (line_name is None) or (line_name == 'hb'):
        #Get Hbeta broad profiles (need to load all MCMC samples)
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:2] != 'Hb') or (gauss_names[3*p + 2][3:5] != 'br'):
                continue
            
            pvals.append(p)


        profiles = []
        for i in range(gauss_result_tot.shape[0]):

            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
        
        
        profiles = np.vstack(profiles)

        prof_med = np.median(profiles, axis=0)
        prof_err_lo = prof_med - np.percentile(profiles, 16, axis=0)
        prof_err_hi = np.percentile(profiles, 84, axis=0) - prof_med

        colnames = ['wavelength', 'profile', 'err_lo', 'err_hi']            
        profile_info = Table( [qi.wave, prof_med, prof_err_lo, prof_err_hi], names=colnames)
        profile_info.write( epoch_dir + 'Hb_br_profile.csv', overwrite=True )


    ####################################################################
    ####################################################################
    # OIII

    if (line_name is None) or (line_name == 'hb'):
        #Get OIII core and wing profiles
        profiles = []
        names = []
        for p in range( len(gauss_result)//3 ):
            if gauss_names[3*p + 2][:4] != 'OIII':
                continue
            
            
            profile = qi.Onegauss( np.log(qi.wave), gauss_result[3*p:3*(p+1)] )
            profiles.append( profile )
            names.append( gauss_names[3*p + 2][:9] )
            
        
        profile_info = np.vstack([ [qi.wave], profiles])
        colnames = ['wavelength']
        for i in range(len(profiles)):
            colnames.append( names[i] )
            
        profile_info = Table( profile_info.T, names=colnames)
        profile_info.write( epoch_dir + 'OIII_profile.csv', overwrite=True )
    

    ####################################################################   
    ####################################################################
    # Halpha
    
    if (line_name is None) or (line_name == 'ha'):
        #Get Halpha broad profiles (need to load all MCMC samples)
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:2] != 'Ha') or (gauss_names[3*p + 2][3:5] != 'br'):
                continue
            
            pvals.append(p)


        profiles = []
        for i in range(gauss_result_tot.shape[0]):

            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
    
    
        profiles = np.vstack(profiles)

        prof_med = np.median(profiles, axis=0)
        prof_err_lo = prof_med - np.percentile(profiles, 16, axis=0)
        prof_err_hi = np.percentile(profiles, 84, axis=0) - prof_med

        colnames = ['wavelength', 'profile', 'err_lo', 'err_hi']            
        profile_info = Table( [qi.wave, prof_med, prof_err_lo, prof_err_hi], names=colnames)
        profile_info.write( epoch_dir + 'Ha_br_profile.csv', overwrite=True )
    
    
    ####################################################################
    ####################################################################
    # MgII
    
    if (line_name is None) or (line_name == 'mg2'):
        #Get MgII broad profiles (need to load all MCMC samples)
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:4] != 'MgII') or (gauss_names[3*p + 2][5:7] != 'br'):
                continue
            
            pvals.append(p)


        profiles = []
        for i in range(gauss_result_tot.shape[0]):

            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
        
        
        profiles = np.vstack(profiles)

        prof_med = np.median(profiles, axis=0)
        prof_err_lo = prof_med - np.percentile(profiles, 16, axis=0)
        prof_err_hi = np.percentile(profiles, 84, axis=0) - prof_med

        colnames = ['wavelength', 'profile', 'err_lo', 'err_hi']            
        profile_info = Table( [qi.wave, prof_med, prof_err_lo, prof_err_hi], names=colnames)
        profile_info.write( epoch_dir + 'MgII_br_profile.csv', overwrite=True )


    ####################################################################
    ####################################################################
    # CIV
    
    if (line_name is None) or (line_name == 'c4'):
        #Get MgII broad profiles (need to load all MCMC samples)
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:3] != 'CIV') or (gauss_names[3*p + 2][4:6] != 'br'):
                continue
            
            pvals.append(p)


        profiles = []
        for i in range(gauss_result_tot.shape[0]):

            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
        
        
        profiles = np.vstack(profiles)

        prof_med = np.median(profiles, axis=0)
        prof_err_lo = prof_med - np.percentile(profiles, 16, axis=0)
        prof_err_hi = np.percentile(profiles, 84, axis=0) - prof_med

        colnames = ['wavelength', 'profile', 'err_lo', 'err_hi']            
        profile_info = Table( [qi.wave, prof_med, prof_err_lo, prof_err_hi], names=colnames)
        profile_info.write( epoch_dir + 'CIV_br_profile.csv', overwrite=True )
    
    
    ####################################################################
    ####################################################################
    # Continuum
    
    continuum = qi.f_conti_model
    cont_info = Table( [qi.wave, continuum], names=['wavelength', 'flux'])
    cont_info.write( epoch_dir + 'continuum.csv', overwrite=True )
        
        
    ####################################################################
    ####################################################################
    # Raw broad profiles
    
    raw_br_prof = get_raw_br_prof(qi, line_name)
    raw_br_info = Table( [qi.wave, raw_br_prof, qi.err], names=['wavelength', 'flux', 'err'])
    raw_br_info.write( epoch_dir + 'raw_br_profile.csv', overwrite=True )
        
    return



def get_raw_br_prof(qi, line_name):

    #Get results    
    gauss_result_tot = qi.gauss_result_all
    gauss_result = qi.gauss_result[::2]
    gauss_names = qi.gauss_result_name[::2]
    
    ####################################################################
    ####################################################################
    
    if line_name == 'hb':
        
        #Get Hbeta narrow profile
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:2] != 'Hb') or (gauss_names[3*p + 2][3:5] != 'na'):
                continue
            
            pvals.append(p)
        
        profiles = []
        for i in range(gauss_result_tot.shape[0]):

            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
        
        profiles = np.vstack(profiles)        
        na_prof = np.median(profiles, axis=0)
        
        #Get continuum
        continuum = qi.f_conti_model
        
        #Get OIII
        o3_prof = np.zeros_like( qi.wave )
        for p in range( len(gauss_result)//3 ):
            if gauss_names[3*p + 2][:4] != 'OIII':
                continue
            
            o3_prof += qi.Onegauss( np.log(qi.wave), gauss_result[3*p:3*(p+1)] )
            
        
        #Get HeII
        he2_prof = np.zeros_like( qi.wave )
        for p in range( len(gauss_result)//3 ):
            if gauss_names[3*p + 2][:4] != 'HeII':
                continue
            
            he2_prof += qi.Onegauss( np.log(qi.wave), gauss_result[3*p:3*(p+1)] )
        
        
        raw_prof = qi.flux - (na_prof + continuum + o3_prof + he2_prof)
        
        
    ####################################################################
    ####################################################################
        
    if line_name == 'ha':
        
        #Get Halpha narrow profile
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:2] != 'Ha') or (gauss_names[3*p + 2][3:5] != 'na'):
                continue
            
            pvals.append(p)
            
        profiles = []
        for i in range(gauss_result_tot.shape[0]):
            
            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
            
        profiles = np.vstack(profiles)
        na_prof = np.median(profiles, axis=0)
        
        
        #Get continuum
        continuum = qi.f_conti_model
        
        #Get NII
        n2_prof = np.zeros_like( qi.wave )
        for p in range( len(gauss_result)//3 ):
            if gauss_names[3*p + 2][:3] != 'NII':
                continue
            
            n2_prof += qi.Onegauss( np.log(qi.wave), gauss_result[3*p:3*(p+1)] )
            
        
        #Get SII
        s2_prof = np.zeros_like( qi.wave )
        for p in range( len(gauss_result)//3 ):
            if gauss_names[3*p + 2][:3] != 'SII':
                continue
            
            s2_prof += qi.Onegauss( np.log(qi.wave), gauss_result[3*p:3*(p+1)] )
            
        
        raw_prof = qi.flux - (na_prof + continuum + n2_prof + s2_prof)
    

    ####################################################################
    ####################################################################
    
    if line_name == 'mg2':
        
        #Get MgII narrow profile
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:4] != 'MgII') or (gauss_names[3*p + 2][5:7] != 'na'):
                continue
            
            pvals.append(p)
            
        profiles = []
        for i in range(gauss_result_tot.shape[0]):
                
            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
                
        profiles = np.vstack(profiles)
        na_prof = np.median(profiles, axis=0)
        
        
        #Get continuum
        continuum = qi.f_conti_model
        
        raw_prof = qi.flux - (na_prof + continuum)    
    

    ####################################################################
    ####################################################################
        
    if line_name == 'c4':
        
        #Get CIV narrow profile
        pvals = []
        for p in range( len(gauss_result)//3 ):
            if (gauss_names[3*p + 2][:3] != 'CIV') or (gauss_names[3*p + 2][4:6] != 'na'):
                continue
            
            pvals.append(p)
            
        profiles = []
        for i in range(gauss_result_tot.shape[0]):
                
            profile = np.zeros_like( qi.wave )
            for p in pvals:
                profile += qi.Onegauss( np.log(qi.wave), gauss_result_tot[i, 3*p:3*(p+1)] )
        
            profiles.append(profile)
                
        profiles = np.vstack(profiles)
        na_prof = np.median(profiles, axis=0)
        
        
        #Get continuum
        continuum = qi.f_conti_model
        
        #Get HeII
        he2_prof = np.zeros_like( qi.wave )
        for p in range( len(gauss_result)//3 ):
            if gauss_names[3*p + 2][:4] != 'HeII':
                continue
            
            he2_prof += qi.Onegauss( np.log(qi.wave), gauss_result[3*p:3*(p+1)] )
            
        
        #Get OIII
        o3_prof = np.zeros_like( qi.wave )
        for p in range( len(gauss_result)//3 ):
            if gauss_names[3*p + 2][:4] != 'OIII':
                continue
            
            o3_prof += qi.Onegauss( np.log(qi.wave), gauss_result[3*p:3*(p+1)] )
        

        raw_prof = qi.flux - (na_prof + continuum + he2_prof + o3_prof)    
    
    
    return raw_prof


###############################################################################################
###############################################################################################
###############################################################################################


def run_all_fits(rmid, line_name, main_dir, prefix='', host=True, rej_abs_line=False, ncpu=None):
    
    fe2_dir = main_dir + 'rm{:03d}/'.format(rmid) + line_name + '/fe2/'
    res_dir = main_dir + 'rm{:03d}/'.format(rmid) + line_name + '/qsofit/'
    
    if host:
        host_dir = main_dir + 'rm{:03d}/host_flux/'.format(rmid)
    else:
        host_dir = None
    
    #Load data
    obj = Object(rmid)
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
