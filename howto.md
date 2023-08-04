# How To Use DM Utils to Produce BRAINS-Ready Line Data

## Step 1: Organize Data
- Use the ``Object`` class in ``dmutils.specfit.object`` to gather information from SDSS-RM spectra for a given object (input the RMID)
- Must have raw spectrum files processed by PrepSpec (i.e., multiplied by $p_0(t)$)
  - Currently, assumes that you have the same directories for the $p_0$ files and raw spectra files as I do(with the same name). Though, this can be changed.
- This will obtain all of the raw spectral data, and decorrect the $p_0(t)$ correction performed by PrepSpec


## Step 2: Remove Host-Galaxy Contamination
- Using PyQSOFit, the ``dmutils.specfit.host_contribution`` module contains functions to fit for the host-galaxy spectrum using PCA analysis
  - This will use PCA analysis to fit for the host galaxy for all epochs (over the whole spectrum), and use the epoch with the best SNR as the host galaxy contribution to subtract later on
  - Need a directory with the qsopar file for PyQSOFit, and a directory to store the host flux profiles
  - This will save all host flux profiles as ``host_flux_epoch{}.dat`` for each epoch, and the best SNR epoch as ``best_host_flux.dat`` to use later on

NOTE: This only needs to be done for wavelengths where host-galaxy contribution is significant, and will not work for UV lines.


## Step 3: Remove FeII Contamination
- Using PyQSOFit, the ``dmutils.specfit.fe2_contribution`` module contains functions to fit for the FeII profile
  - If a line name is given, this will only fit for the profile around the line, else it will fit the whole spectrum
  - This will use a set FeII template, and shift, widen, and normalize it
  - If host-galaxy subtraction is needed, it will be subtracted before this fitting (given the host-flux directory)
  - The process of fitting the FeII template is as follows:
    - Fit all epochs separately, save their best-fit parameters
    - Refit all epochs, fiixing the FWHM and shift to the median value of the initial fits
    - See which epoch fits deviate significantly from the median, and fix these epochs to have the meidan value for FWHM, shift, and normalization
    - This process can be changed if needed

## Step 4: Fit Emission Line(s)
- After host-galaxy and FeII template fitting, now fit for (broad) emission lines
  - If no line is specified, it will fit the whole spectrum
- For a given line:
  - Remove the host galaxy contribution if needed
  - Fix the FeII template parameters to those found in Step 3
  - Use the parameters specified in the qsopar.fits file to fit a given emission line
    - If the $\chi^2$ is too large, the line will be refit (this can happen up to 5 times and will be recorded)

## Step 5: Move Line Profiles
- If all fitted profiles seem to be satisfactory, we can now move the line profiles to a common directory
- Use the ``dmutils.specfit.move_line_profs`` module to do this, specifying an input and output directory


## Step 6: Make Input Data Files for BRAINS
- Now that all of the broad line profiles are in a common directory, we can use the ``make_brains_input`` function of the ``Object`` class to make the 2D line input file for BRAINS
- Separately, you can create the input continuum light curve file
