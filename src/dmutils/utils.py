from astropy.table import Table
import numpy as np
from scipy.interpolate import splev, splrep

############################################################################################################
#Get input for BRAINS

#Get the bounds of the emission line profile given the (unbinned) spectra
def get_prof_bounds(fnames, central_wl, tol=5e-2):

    left_bound = []
    right_bound = []
    
    for i in range(len(fnames)):
        ex_dat = Table.read(fnames[i])
        wl = ex_dat['wavelength'].tolist()
        prof = ex_dat['flux'].tolist()


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





def make_input_file(fnames, central_wl, times, z, output_fname, 
                    time_bounds=None, wl_bounds=None, 
                    nbin=None, tol=5e-2):
    
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
    - If the edge bins have no points within them, truncate the spectra to the first/last bin with points
    - Fit a spline to fill the bins with no points
    
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
        
    time_bounds: list of floats, optional
        List of the bounds of the time bins. If not specified, will use the first and last times in the list of times
        
    wl_bounds : list of floats, optional
        List of the bounds of the wavelength bins. If not specified, will use the first and last wavelengths in the list of wavelengths
        
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
    
    if wl_bounds is not None:
        if wl_bounds[0] > bounds[0]:
            bounds[0] = wl_bounds[0]
        if wl_bounds[1] < bounds[1]:
            bounds[1] = wl_bounds[1]
    

    if time_bounds is not None:
        time_mask = (times > time_bounds[0]) & (times < time_bounds[1])
    else:
        time_mask = ~np.zeros(len(times), dtype=bool)



    for i in range(len(fnames)):
        if not time_mask[i]:
            continue

        hb_dat = Table.read(fnames[i], format='ascii.csv')
        
        wl_rest = hb_dat['wavelength']
        prof = hb_dat['flux']
        prof_err = hb_dat['err']    
        
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


    #Truncate time array if needed
    times = np.array(times)[time_mask]

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


    #Interpolate over NaNs
    for n in range(len(wl_tot)):
        nan_mask = np.isnan(prof_tot_rebin[n])
        
        if len(np.argwhere(nan_mask).T[0]) == 0:
            continue
        
        xgood = wl_tot_rebin[n][~nan_mask]
        ygood = prof_tot_rebin[n][~nan_mask]
        
        xtot = wl_tot_rebin[n]
        
        
        #This is with B-Splines
        spl = splrep(xgood, ygood, s=0)
        prof_tot_rebin[n] = splev(xtot, spl)        
        
        # #This is with interp1d
        # func = interp1d(xgood, ygood, kind='cubic')


    #Deal with NaNs in the error
    for i in range(len(err_tot_rebin)):
        nan_ind = np.argwhere( np.isnan(err_tot_rebin[i]) ).T[0]
        
        
        j = 0
        while j < len(nan_ind):

            left_ind = nan_ind[j] - 1
            
            n_interp = 1
            m = nan_ind[j] + 1
            while m in nan_ind:
                n_interp += 1
                m += 1
                
            right_ind = m
                        
            err_tot_rebin[i,left_ind+1:right_ind] = np.sqrt( (err_tot_rebin[i,left_ind] + err_tot_rebin[i,right_ind])**2 /4 + np.var(prof_tot_rebin[i,left_ind+1:right_ind]) )
            
            j += n_interp


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
