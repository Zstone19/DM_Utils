import numpy as np
from astropy.io import fits
import os

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



def make_qsopar(path, fname='qsopar.fits', oiii_wings=True,
                maxsig_ha_na=0.0017, maxdw_hb_na=.01, maxdw_hb_br=.01,
                minsca_hb_br=0., maxsig_hb_br=.05):
    """
    Create parameter file
    lambda    complexname  minwav maxwav linename     ngauss inisca minsca maxsca inisig        minsig    maxsig        inidw mindw   maxdw  vindex windex findex fvalue varysca    varysig  varydw
    """

    if maxsig_ha_na <= 1e-3:
        inisig_ha_na = (maxsig_ha_na + 5e-4)/2
    else:
        inisig_ha_na = 1e-3
        
        
    inisca_hb_br = .1
    if minsca_hb_br > inisca_hb_br:
        minsca_hb_br = inisca_hb_br + .1


    recs = [
    (6564.61, r'H$\alpha$', 6400, 6800, 'Ha_br',      3,     0.1,            0.0,            1e10,  5e-3,         0.004,    0.05,         0.00, -0.015,        0.015,        0,     0,     0,    0.05 ,  1,        1,       1),
    (6564.61, r'H$\alpha$', 6400, 6800, 'Ha_na',      1,     0.1,            0.0,            1e10,  inisig_ha_na, 5e-4,     maxsig_ha_na, 0.00, -0.01,         0.01,         1,     1,     0,    0.002,  1,        1,       1),
    (6549.85, r'H$\alpha$', 6400, 6800, 'NII6549',    1,     0.1,            0.0,            1e10,  inisig_ha_na, 2.3e-4,   maxsig_ha_na, 0.00, -5e-3,         5e-3,         1,     1,     1,    0.001,  1,        1,       1),
    (6585.28, r'H$\alpha$', 6400, 6800, 'NII6585',    1,     0.1,            0.0,            1e10,  inisig_ha_na, 2.3e-4,   maxsig_ha_na, 0.00, -5e-3,         5e-3,         1,     1,     1,    0.003,  1,        1,       1),
    (6718.29, r'H$\alpha$', 6400, 6800, 'SII6718',    1,     0.1,            0.0,            1e10,  inisig_ha_na, 2.3e-4,   maxsig_ha_na, 0.00, -5e-3,         5e-3,         1,     1,     2,    0.001,  1,        1,       1),
    (6732.67, r'H$\alpha$', 6400, 6800, 'SII6732',    1,     0.1,            0.0,            1e10,  inisig_ha_na, 2.3e-4,   maxsig_ha_na, 0.00, -5e-3,         5e-3,         1,     1,     2,    0.001,  1,        1,       1),

    (6302.05, 'OI',         6250, 6350, 'OI6302',     1,     0.1,            0.0,            1e10,  1e-3,         2.3e-4,   0.0017,       0.0,  -5e-3,         5e-3,         1,     0,     0,    0.001,  1,        1,       1),

    (4862.68, r'H$\beta$', 4640, 5100, 'Hb_br',       3,     inisca_hb_br,   minsca_hb_br,   1e10,  5e-3,         0.004,    maxsig_hb_br, 0.00, -maxdw_hb_br,  maxdw_hb_br,  0,     0,     0,    0.01 ,  1,        1,       1),
    (4862.68, r'H$\beta$', 4640, 5100, 'Hb_na',       1,     0.1,            0.0,            1e10,  1e-3,         2.3e-4,   0.0017,       0.00, -maxdw_hb_na,  maxdw_hb_na,  1,     1,     0,    0.002,  1,        1,       1),
    (4960.30, r'H$\beta$', 4640, 5100, 'OIII4959c',   1,     0.1,            0.0,            1e10,  1e-3,         2.3e-4,   0.0017,       0.00, -maxdw_hb_na,  maxdw_hb_na,  1,     1,     1,    0.333,  1,        1,       1),
    (5008.24, r'H$\beta$', 4640, 5100, 'OIII5007c',   1,     0.1,            0.0,            1e10,  1e-3,         2.3e-4,   0.0017,       0.00, -maxdw_hb_na,  maxdw_hb_na,  1,     1,     1,    1.000,  1,        1,       1),
    (4687.02, r'H$\beta$', 4640, 5100, 'HeII4687_br', 1,     0.1,            0.0,            1e10,  5e-3,         0.004,    0.05,         0.00, -0.005,        0.005,        0,     0,     0,    0.001,  1,        1,       1),
    (4687.02, r'H$\beta$', 4640, 5100, 'HeII4687_na', 1,     0.1,            0.0,            1e10,  1e-3,         2.3e-4,   0.0017,       0.00, -0.005,        0.005,        1,     1,     0,    0.001,  1,        1,       1),

    (3728.48, 'OII',       3650, 3800, 'OII3728',     1,     0.1,            0.0,            1e10,  1e-3,         3.333e-4, 0.0017,       0.00, -0.01,         0.01,         1,     1,     0,    0.001,  1,        1,       1),

    (2798.75, 'MgII',      2700, 2900, 'MgII_br',     2,     0.1,            0.0,            1e10,  5e-3,         0.004,    0.05,         0.00, -0.015,        0.015,        0,     0,     0,    0.05,   1,        1,       1),
    (2798.75, 'MgII',      2700, 2900, 'MgII_na',     1,     0.1,            0.0,            1e10,  1e-3,         5e-4,     0.0017,       0.00, -0.01,         0.01,         1,     1,     0,    0.002,  1,        1,       1),

    (1908.73, 'CIII',      1700, 1970, 'CIII_br',     2,     0.1,            0.0,            1e10,  5e-3,         0.004,    0.05,         0.00, -0.015,        0.015,        99,    0,     0,    0.01,   1,        1,       1),
    (1908.73, 'CIII',      1700, 1970, 'CIII_na',     1,     0.1,            0.0,            1e10, 1e-3,          5e-4,     0.0017,       0.00, -0.01,         0.01,         1,     1,     0,    0.002,  1,        1,       1),
 
    (1549.06, 'CIV',       1500, 1700, 'CIV_br',      1,     0.1,            0.0,            1e10,  5e-3,         0.004,    0.05,         0.00, -0.015,        0.015,        0,     0,     0,    0.05 ,  1,        1,       1),
    (1549.06, 'CIV',       1500, 1700, 'CIV_na',      1,     0.1,            0.0,            1e10,  1e-3,         5e-4,     0.0017,       0.00, -0.01,         0.01,         1,     1,     0,    0.002,  1,        1,       1),
    (1640.42, 'CIV',       1500, 1700, 'HeII1640',    1,     0.1,            0.0,            1e10,  1e-3,         5e-4,     0.0017,       0.00, -0.008,        0.008,        1,     1,     0,    0.002,  1,        1,       1),
    (1663.48, 'CIV',       1500, 1700, 'OIII1663',    1,     0.1,            0.0,            1e10,  1e-3,         5e-4,     0.0017,       0.00, -0.008,        0.008,        1,     1,     0,    0.002,  1,        1,       1),
    (1640.42, 'CIV',       1500, 1700, 'HeII1640_br', 1,     0.1,            0.0,            1e10,  5e-3,         0.0025,   0.02,         0.00, -0.008,        0.008,        1,     1,     0,    0.002,  1,        1,       1),
    (1663.48, 'CIV',       1500, 1700, 'OIII1663_br', 1,     0.1,            0.0,            1e10,  5e-3,         0.0025,   0.02,         0.00, -0.008,        0.008,        1,     1,     0,    0.002,  1,        1,       1),

    (1215.67, 'Lya',       1150, 1290, 'Lya_br',      1,     0.1,            0.0,            1e10,  5e-3,         0.004,    0.05,         0.00, -0.02,         0.02,         0,     0,     0,    0.05 ,  1,        1,       1),
    (1215.67, 'Lya',       1150, 1290, 'Lya_na',      1,     0.1,            0.0,            1e10,  1e-3,         5e-4,     0.0017,       0.00, -0.01,         0.01,         0,     0,     0,    0.002,  1,        1,       1)]

    if oiii_wings:
        recs.append( (4960.30, r'H$\beta$', 4640, 5100, 'OIII4959w',   1,     0.1,   0.0,   1e10,  3e-3,         2.3e-4,  0.004,        0.00, -0.01,         0.01,         2,     2,     0,    0.001, 1,        1,       1) )
        recs.append( (5008.24, r'H$\beta$', 4640, 5100, 'OIII5007w',   1,     0.1,   0.0,   1e10,  3e-3,         2.3e-4,  0.004,        0.00, -0.01,         0.01,         2,     2,     0,    0.002, 1,        1,       1) )
       

    newdata = np.rec.array( recs,
                            formats = 'float32, a20,      float32, float32, a20,      int32,  float32, float32, float32, float32, float32, float32, float32, float32, float32, int32, int32,  int32, float32, int32,   int32,  int32',
                            names  =  'lambda, compname, minwav, maxwav, linename, ngauss, inisca, minsca, maxsca, inisig, minsig, maxsig, inidw,  mindw,  maxdw, vindex, windex, findex, fvalue, varysca, varysig, varydw')



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





def make_qsopar_hb(path, fname='qsopar.fits', oiii_wings=True,
                   maxdw_hb_na=.01, maxdw_hb_br=.01, maxdw_o3w=.01,
                   minsca_hb_br=0., maxsig_hb_br=.05, maxsig_he2_br=.05):
    """
    Create parameter file
    lambda    complexname  minwav maxwav linename     ngauss inisca minsca                  maxsca inisig         minsig  maxsig        inidw mindw   maxdw  vindex windex findex fvalue varysca    varysig  varydw
    """
        
        
    inisca_hb_br = .1
    if minsca_hb_br > inisca_hb_br:
        minsca_hb_br = inisca_hb_br + .1


    recs = [
    (4862.68, r'H$\beta$', 4640, 5100, 'Hb_br',       3,     inisca_hb_br,   minsca_hb_br,   1e10,  5e-3,         0.004,    maxsig_hb_br,  0.00, -maxdw_hb_br,  maxdw_hb_br,  0,     0,     0,    0.01 ,  1,        1,       1),
    (4862.68, r'H$\beta$', 4640, 5100, 'Hb_na',       1,     0.1,            0.0,            1e10,  1e-3,         2.3e-4,   0.0017,        0.00, -maxdw_hb_na,  maxdw_hb_na,  1,     1,     0,    0.002,  1,        1,       1),
    (4960.30, r'H$\beta$', 4640, 5100, 'OIII4959c',   1,     0.1,            0.0,            1e10,  1e-3,         2.3e-4,   0.0017,        0.00, -maxdw_hb_na,  maxdw_hb_na,  1,     1,     1,    0.333,  1,        1,       1),
    (5008.24, r'H$\beta$', 4640, 5100, 'OIII5007c',   1,     0.1,            0.0,            1e10,  1e-3,         2.3e-4,   0.0017,        0.00, -maxdw_hb_na,  maxdw_hb_na,  1,     1,     1,    1.000,  1,        1,       1),
    (4687.02, r'H$\beta$', 4640, 5100, 'HeII4687_br', 1,     0.1,            0.0,            1e10,  5e-3,         0.004,    maxsig_he2_br, 0.00, -0.005,        0.005,        0,     0,     0,    0.001,  1,        1,       1),
    (4687.02, r'H$\beta$', 4640, 5100, 'HeII4687_na', 1,     0.1,            0.0,            1e10,  1e-3,         2.3e-4,   0.0017,        0.00, -0.005,        0.005,        1,     1,     0,    0.001,  1,        1,       1),

    (3728.48, 'OII',       3650, 3800, 'OII3728',     1,     0.1,            0.0,            1e10,  1e-3,         3.333e-4, 0.0017,        0.00, -0.01,         0.01,         1,     1,     0,    0.001,  1,        1,       1)]

    if oiii_wings:
        recs.append( (4960.30, r'H$\beta$', 4640, 5100, 'OIII4959w',   1,     0.1,   0.0,    1e10,  3e-3,         2.3e-4,   0.004,         0.00, -maxdw_o3w,    maxdw_o3w,    2,     2,     0,    0.001,  1,        1,       1) )
        recs.append( (5008.24, r'H$\beta$', 4640, 5100, 'OIII5007w',   1,     0.1,   0.0,    1e10,  3e-3,         2.3e-4,   0.004,         0.00, -maxdw_o3w,    maxdw_o3w,    2,     2,     0,    0.002,  1,        1,       1) )
       

    newdata = np.rec.array( recs,
                            formats = 'float32, a20,      float32, float32, a20,      int32,  float32, float32, float32, float32, float32, float32, float32, float32, float32, int32, int32,  int32, float32, int32,   int32,  int32',
                            names  =  'lambda, compname, minwav, maxwav, linename, ngauss, inisca, minsca, maxsca, inisig, minsig, maxsig, inidw,  mindw,  maxdw, vindex, windex, findex, fvalue, varysca, varysig, varydw')



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


def make_qsopar_mg2(path, fname='qsopar.fits'):
    """
    Create parameter file
    lambda    complexname  minwav maxwav linename     ngauss inisca minsca maxsca inisig        minsig    maxsig        inidw mindw   maxdw  vindex windex findex fvalue varysca    varysig  varydw
    """


    recs = [
    (3728.48, 'OII',       3650, 3800, 'OII3728',     1,     0.1,            0.0,            1e10,  1e-3,         3.333e-4, 0.0017,       0.00, -0.01,         0.01,         1,     1,     0,    0.001,  1,        1,       1),

    (2798.75, 'MgII',      2700, 2900, 'MgII_br',     2,     0.1,            0.0,            1e10,  5e-3,         0.004,    0.05,         0.00, -0.015,        0.015,        0,     0,     0,    0.05,   1,        1,       1),
    (2798.75, 'MgII',      2700, 2900, 'MgII_na',     1,     0.1,            0.0,            1e10,  1e-3,         5e-4,     0.0017,       0.00, -0.01,         0.01,         1,     1,     0,    0.002,  1,        1,       1),

    (1908.73, 'CIII',      1700, 1970, 'CIII_br',     2,     0.1,            0.0,            1e10,  5e-3,         0.004,    0.05,         0.00, -0.015,        0.015,        99,    0,     0,    0.01,   1,        1,       1),
    (1908.73, 'CIII',      1700, 1970, 'CIII_na',     1,     0.1,            0.0,            1e10, 1e-3,          5e-4,     0.0017,       0.00, -0.01,         0.01,         1,     1,     0,    0.002,  1,        1,       1)]       

    newdata = np.rec.array( recs,
                            formats = 'float32, a20,      float32, float32, a20,      int32,  float32, float32, float32, float32, float32, float32, float32, float32, float32, int32, int32,  int32, float32, int32,   int32,  int32',
                            names  =  'lambda, compname, minwav, maxwav, linename, ngauss, inisca, minsca, maxsca, inisig, minsig, maxsig, inidw,  mindw,  maxdw, vindex, windex, findex, fvalue, varysca, varysig, varydw')



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


def make_qsopar_ha_fixed(path, o3_dw, o3_sig, fname='qsopar.fits',
                        maxsig_ha_na=0.0017, maxdw_ha_na=5e-3, 
                        fix_dw=True, fix_sig=True,
                        maxsig_ha_br=.05, minsca_ha_br=0.):
    """
    Create parameter file
    lambda    complexname  minwav maxwav linename     ngauss inisca minsca maxsca inisig        minsig    maxsig        inidw  mindw   maxdw  vindex windex findex fvalue varysca    varysig       varydw
    """
    
    
    if maxsig_ha_br <= 5e-3:
        inisig_ha_br = (maxsig_ha_br + .004)/2
    else:
        inisig_ha_br = 5e-3
    
    
    if minsca_ha_br > .1:
        inisca_ha_br = minsca_ha_br + .1
    else:
        inisca_ha_br = .1




    if fix_sig:
        inisig_ha_na = o3_sig

    else:
        if maxsig_ha_na <= 1e-3:
            inisig_ha_na = (maxsig_ha_na + 5e-4)/2
        else:
            inisig_ha_na = 1e-3


    if fix_dw:
        inidw_ha_na = o3_dw
    else:
        inidw_ha_na = 0.0







    if fix_sig:
        fixval1 = 0
    else:
        fixval1 = 1

    if fix_dw:
        fixval2 = 0
    else:
        fixval2 = 1
        
        
        
        



    recs = [
    (6564.61, r'H$\alpha$', 6400, 6800, 'Ha_br',      3,     inisca_ha_br,   minsca_ha_br,   1e10,  inisig_ha_br, 0.004,    maxsig_ha_br, 0.0,         -0.015,         0.015,        0,    0,     0,    0.05 ,  1,        1,            1),
    (6564.61, r'H$\alpha$', 6400, 6800, 'Ha_na',      1,     0.1,            0.0,            1e10,  inisig_ha_na, 5e-4,     maxsig_ha_na, inidw_ha_na, -maxdw_ha_na,   maxdw_ha_na,  1,    1,     0,    0.002,  1,        fixval1,       fixval2),
    (6549.85, r'H$\alpha$', 6400, 6800, 'NII6549',    1,     0.1,            0.0,            1e10,  inisig_ha_na, 2.3e-4,   maxsig_ha_na, inidw_ha_na, -maxdw_ha_na,   maxdw_ha_na,  1,    1,     1,    0.001,  1,        fixval1,       fixval2),
    (6585.28, r'H$\alpha$', 6400, 6800, 'NII6585',    1,     0.1,            0.0,            1e10,  inisig_ha_na, 2.3e-4,   maxsig_ha_na, inidw_ha_na, -maxdw_ha_na,   maxdw_ha_na,  1,    1,     1,    0.003,  1,        fixval1,       fixval2),
    (6718.29, r'H$\alpha$', 6400, 6800, 'SII6718',    1,     0.1,            0.0,            1e10,  inisig_ha_na, 2.3e-4,   maxsig_ha_na, inidw_ha_na, -maxdw_ha_na,   maxdw_ha_na,  1,    1,     2,    0.001,  1,        fixval1,       fixval2),
    (6732.67, r'H$\alpha$', 6400, 6800, 'SII6732',    1,     0.1,            0.0,            1e10,  inisig_ha_na, 2.3e-4,   maxsig_ha_na, inidw_ha_na, -maxdw_ha_na,   maxdw_ha_na,  1,    1,     2,    0.001,  1,        fixval1,       fixval2),      

    (6302.05, 'OI',         6250, 6350, 'OI6302',     1,     0.1,            0.0,            1e10,  1e-3,         2.3e-4,   0.0017,       0.0,         -5e-3,          5e-3,         1,    0,     0,    0.001,  1,        1,            1)]

    newdata = np.rec.array( recs,
                            formats = 'float32, a20,      float32, float32, a20,      int32,  float32, float32, float32, float32, float32, float32, float32, float32, float32, int32, int32,  int32, float32, int32,   int32,  int32',
                            names  =  'lambda, compname, minwav, maxwav, linename, ngauss, inisca, minsca, maxsca, inisig, minsig, maxsig, inidw,  mindw,  maxdw, vindex, windex, findex, fvalue, varysca, varysig, varydw')



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
