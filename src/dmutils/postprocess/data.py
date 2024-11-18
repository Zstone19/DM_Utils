import os
import re
import sys
import copy
import configparser as cp

import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz

from astropy.table import Table
from dmutils.input import read_input_file
from numba import njit

###########################################################################
###########################################################################
###########################################################################
# From bplotlib

class Param:
  """
  load param file
  """
  def __init__(self, fname):
    self.param_file = fname
    self._param_parser(self.param_file)
  
  def _param_parser(self, fname):
    """
    parse parameter file
    """
    config = cp.RawConfigParser(delimiters=' ', comment_prefixes='%', inline_comment_prefixes='%', 
    default_section=cp.DEFAULTSECT, empty_lines_in_values=False)
  
    with open(fname) as f:
      file_content = '[dump]\n' + f.read()

    config.read_string(file_content)
    
    # check the absolute path
    if os.path.isabs(config['dump']['filedir']) == False:
      raise Exception("FileDir in %s is not an absoulte path.\n"%self.param_file)
    
    self.param = config['dump']
  
  def set_param_file(self, fname):
    """
    set parameter file
    """
    self.param_file = fname
    self._param_parser(fname)
    return

class Options:
  """
  load options file
  """
  def __init__(self, file_dir, flag_dim):
     self._options_parser(file_dir, flag_dim)

  def _options_parser(self, file_dir, flag_dim):
    config = cp.RawConfigParser(delimiters=' ', comment_prefixes='#', inline_comment_prefixes='#', 
    default_section=cp.DEFAULTSECT, empty_lines_in_values=False)
    
    if flag_dim == '-2':
      self.option_file = ""
    elif flag_dim == '-1':
      self.option_file = file_dir + "/param/OPTIONSCON"
    elif flag_dim == '0':
      self.option_file = file_dir + "/param/OPTIONSCON"
    elif flag_dim == '1':
      self.option_file = file_dir + "/param/OPTIONS1D"
    elif flag_dim == '2':
      self.option_file = file_dir + "/param/OPTIONS2D"
    elif flag_dim == '3':
      self.option_file = file_dir + "/param/OPTIONSLP"
    elif flag_dim == '4':
      self.option_file = file_dir + "/param/OPTIONSSA"
    elif flag_dim == '5':
      self.option_file = file_dir + "/param/OPTIONSSA1D"
    elif flag_dim == '6':
      self.option_file = file_dir + "/param/OPTIONSSA2D"
    elif flag_dim == '7':
      self.option_file = file_dir + "/param/OPTIONSSARM"

    if self.option_file != "":
      with open(self.option_file) as f:
        file_content = '[dump]\n' + f.read()
  
      config.read_string(file_content)
      self.options = config['dump']
    else:
      self.options = None


class ParaName:
  """
  load parameter names
  """
  def __init__(self, file_dir, flag_dim):
    self._load_para_names(file_dir, flag_dim)

  def _load_para_names(self, file_dir, flag_dim):

    if flag_dim == '-2':
      self.para_names_file = ""
    elif flag_dim == '-1':
      self.para_names_file = ""
    elif flag_dim == '0':
      self.para_names_file = file_dir + "/data/para_names_con.txt"
    elif flag_dim == '1':
      self.para_names_file = file_dir + "/data/para_names_1d.txt"
    elif flag_dim == '2':
      self.para_names_file = file_dir + "/data/para_names_2d.txt"
    elif flag_dim == '3':
      self.para_names_file = file_dir + "/data/para_names_lp.txt"
    elif flag_dim == '4':
      self.para_names_file = file_dir + "/data/para_names_sa.txt"
    elif flag_dim == '5':
      self.para_names_file = file_dir + "/data/para_names_sa1d.txt"
    elif flag_dim == '6':
      self.para_names_file = file_dir + "/data/para_names_sa2d.txt"
    elif flag_dim == '7':
      self.para_names_file = file_dir + "/data/para_names_sarm.txt"
    else:
      raise Exception("Incorrect FlagDim.")

    names = np.genfromtxt(self.para_names_file, comments='#', \
      dtype=[int, 'S30', float, float, int, int, float], delimiter=[4, 30, 12, 12, 4, 5, 15])
    
    # the first line is header
    names = np.delete(names, 0)
    
    self.para_names = {}
    self.para_names['name'] = np.array([str(f.strip(), encoding='utf8') for f in names['f1']])
    self.para_names['min']  = names['f2']
    self.para_names['max']  = names['f3']
    self.para_names['prior']  = names['f4']
    self.para_names['fix']  = names['f5']
    self.para_names['val']  = names['f6']

    self.num_param_blrmodel_rm = 0
    self.num_param_blr_rm = 0
    self.num_param_rm_extra = 0
    self.num_param_sa = 0
    self.num_param_blrmodel_sa = 0
    self.num_param_sa_extra = 0
    
    self.num_param_blrmodel_rm = 0
    self.num_param_blrmodel_sa = 0
    self.num_param_sa_extra = 0
    for name in self.para_names['name']:
      if re.match("BLR_model", name):
        self.num_param_blrmodel_rm += 1
      
      if re.match("SA_BLR_model", name):
        self.num_param_blrmodel_sa += 1
      
      if re.match("SA_Extra_Par", name):
        self.num_param_sa_extra += 1
    
    self.num_param_sa = self.num_param_blrmodel_sa + self.num_param_sa_extra
    idx_con =  np.nonzero(self.para_names['name'] == 'sys_err_con')
    if len(idx_con[0]) > 0:
      self.num_param_rm_extra = idx_con[0][0] - self.num_param_blrmodel_rm - self.num_param_sa
    else:
      self.num_param_rm_extra = 0

    self.num_param_blr_rm = self.num_param_rm_extra + self.num_param_blrmodel_rm

    if 'time_series' in self.para_names['name']:
      idx = np.nonzero(self.para_names['name'] == 'time_series')
      self.num_param_con = idx[0][0] - self.num_param_sa - self.num_param_blr_rm
    else:
      self.num_param_con = 0

    #print(self.num_param_blr_rm, self.num_param_sa, self.num_param_con)
  
  def print_blrmodel_para_names(self):
    """
    print blr model parameter names
    """
    for i in range(self.num_param_blrmodel_rm):
      print("{:3d} {}".format(i, self.para_names['name'][i]))
  
  def locate_bhmass(self):
    """
    locate the index of black hole mass parameter
    """
    idx = -1
    for i in range(self.num_param_blrmodel_rm):
      if re.match("BLR_model_ln\(Mbh\)", self.para_names['name'][i]):
        idx = i 
        break
    
    if idx == -1:
      raise ValueError("No black hole mass parameter")
    else:
      return idx


###########################################################################
###########################################################################
###########################################################################
# Get best-fit parameters

def bestfit_kde(samps, weights, bw_method='scott', xvals=None, nkde=None):
    
    if xvals is None:
        if nkde is None:
            nkde = 1000
      
        rnum = samps.max() - samps.min()
        xvals = np.linspace(samps.min() - rnum, samps.max() + rnum, nkde)
  
    kde = gaussian_kde(samps, weights=weights, bw_method=bw_method)
    kde_vals = kde.evaluate(xvals)
    
    #Get percentiles
    cdf = cumtrapz(kde_vals, xvals, initial=0)
    p16 = xvals[ np.abs(cdf-0.16).argmin() ]
    p50 = xvals[ np.abs(cdf-0.50).argmin() ]
    p84 = xvals[ np.abs(cdf-0.84).argmin() ]
    
    return xvals, kde_vals, p50, p50-p16, p84-p50


def bestfit_median(samps, weights):
    p16 = weighted_percentile(samps, weights, .16)
    p50 = weighted_percentile(samps, weights, .50)
    p84 = weighted_percentile(samps, weights, .84)
    return p50, p50-p16, p84-p50

def bestfit_mean(samps, weights):
    mean = np.average(samps, weights=weights)
    std = np.sqrt(np.cov(samps, aweights=weights))
    return mean, std, std

def bestfit_map(samps, weights):
    val = samps[np.argmax(samps*weights)]
    return val, 0., 0.
  
  
###########################################################################
###########################################################################
###########################################################################
# Stats

def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    return np.interp(perc, cdf, data)
  
#@njit(fastmath=True)
def asymmetric_normal_draw(val, errlo, errhi, nsamp, lobound=None, hibound=None):
  
    if lobound is None:
        lobound = -np.inf
    if hibound is None:
        hibound = np.inf
    
    if nsamp == 1:
        return asymmetric_normal_draw_single(val, errlo, errhi, lobound, hibound)
    
    rng = np.random.uniform(0,1,nsamp)
    mask = rng < .5
    mask_ind = np.argwhere(mask).flatten()
    nomask_ind = np.argwhere(~mask).flatten()
    
    val_out = np.zeros(nsamp)
    val_out[mask] = np.random.normal(val, errlo, mask.sum())
    val_out[~mask] = np.random.normal(val, errhi, (~mask).sum())
    
    for i in range(len(mask_ind)):
        if (val_out[mask_ind[i]] > val) or (val_out[mask_ind[i]] < lobound):
            while (val_out[mask_ind[i]] > val) or (val_out[mask_ind[i]] < lobound):
                val_out[mask_ind[i]] = np.random.normal(val, errlo)
                

    for i in range(len(nomask_ind)):
        if (val_out[nomask_ind[i]] < val) or (val_out[nomask_ind[i]] > hibound):
            while (val_out[nomask_ind[i]] < val) or (val_out[nomask_ind[i]] > hibound):
                val_out[nomask_ind[i]] = np.random.normal(val, errhi)

    return val_out


@njit(fastmath=True)
def asymmetric_normal_draw_single(val, errlo, errhi, lobound=None, hibound=None):
    
    if lobound is None:
        lobound = -np.inf
    if hibound is None:
        hibound = np.inf
    
    
    rng = np.random.uniform(0,1)

    if rng < .5:
        val_out = np.random.normal(val, errlo)
        if (val_out > val) or (val_out < lobound):
            while (val_out > val) or (val_out < lobound):
                val_out = np.random.normal(val, errlo)
                
    else:
        val_out = np.random.normal(val, errhi)
        if (val_out < val) or (val_out > hibound):
            while (val_out < val) or (val_out > hibound):
                val_out = np.random.normal(val, errhi)

    return val_out


###########################################################################
###########################################################################
###########################################################################

def get_rset_tset(xline, xcon, r_input, t_input):
    rmin_set = 0
    
    
    tspan_data_con = xcon.max() - xcon.min()
    tspan_data = xline.max() - xcon.min()    
    tcad_data = np.max( [np.diff(xline).max(), np.diff(xcon).max(), tspan_data] )
    
    tset = tspan_data_con + (xcon[0] - xline[0])
    tset = max( 2*tcad_data, tset )
    
    
    
    dt = xcon[0] - tset
    if (r_input > 0):
        dt = max( dt , xline[0] - r_input*2 )
    elif (t_input > 0):
        dt = max( dt, xcon[0] - t_input )
    
    tset = xcon[0] - dt



    rmax_set = tspan_data/2
    
    rmax_set = min( rmax_set, (xline[0] - xcon[0] + tset)/2  )
    if r_input > 0:
        rmax_set = min( rmax_set, r_input )
        
    return rmin_set, rmax_set, tset

def get_paramfile_inputs(brains_dir):
    param = Param(brains_dir + '/param/param')    
    return param.param._parser._sections['dump']    



class DM_Data:
    
    def __init__(self, brains_dir):
        
        #All fluxes and velocities will be scaled
        #Use VEL_UNIT to convert to km/s
        #Use cont_scale and line_scale to convert to input flux units
        
        self.paramnames = np.array([ 'log_RBLR', 'BETA', 'F', 'INC', 'OPN', 'KAPPA', 'GAMMA', 'XI', 'log_MBH', 'F_ELLIP', 'F_FLOW',
                            'log_SIGR_CIRC', 'log_SIGTHE_CIRC', 'log_SIGR_RAD', 'log_SIGTHE_RAD', 'THETA_E', 'log_SIG_TURB',
                            'DELTA_V_LINE', 'SIGMA_SYS_LINE', 'SIGMA_SYS_CON', 'log_SIGMA_D', 'log_TAU_D', 'B', 'A', 'AG'])
        
        self.param_units = np.array(['log(lt-d)', '', 'RBLR', 'deg', 'deg', '', '', '', 'log(Msol)', '', '',
                            'log(km/s)', 'log(km/s)', 'log(km/s)', 'log(km/s)', 'deg', 'log(km/s)', 
                            'km/s', 'f_lambda', 'f_lambda', 'f_lambda', 'd', 'f_lambda', '', ''])
        
        ################################
        #Constants
        self.GRAVITY = 6.672e-8
        self.SOLAR_MASS = 1.989e33
        self.CVAL = 2.9979e10
        self.CM_PER_LD = self.CVAL*8.64e4
        self.VEL_UNIT = np.sqrt( self.GRAVITY * 1.0e6 * self.SOLAR_MASS / self.CM_PER_LD ) / 1.0e5
        self.C_UNIT = self.CVAL/1.0e5/self.VEL_UNIT
        
        self.EPS = sys.float_info.epsilon
        
        self.brains_dir = brains_dir
        self.paramfile_inputs = get_paramfile_inputs(self.brains_dir)
        
        self.z = float(self.paramfile_inputs['redshift'])
        self.central_wl = float(self.paramfile_inputs['linecenter'])
        self.ncloud = int(self.paramfile_inputs['ncloudpercore'])
        self.vel_per_cloud = int(self.paramfile_inputs['nvpercloud'])
        self.nrecon_cont = int(self.paramfile_inputs['nconrecon'])
        self.nrecon_line = int(self.paramfile_inputs['nlinerecon'])
        self.nrecon_vel = int(self.paramfile_inputs['nvelrecon'])
        
        self.flag_linecenter = int(self.paramfile_inputs['flaglinecenter'])
        self.flag_trend = int(self.paramfile_inputs['flagtrend'])
        self.flag_trend_diff = int(self.paramfile_inputs['flagtrenddiff'])
        self.flag_inst_res = int(self.paramfile_inputs['flaginstres'])
        self.flag_nl = int(self.paramfile_inputs['flagnarrowline'])
        self.flag_cont_err = int(self.paramfile_inputs['flagconsyserr'])
        self.flag_line_err = int(self.paramfile_inputs['flaglinesyserr'])
        
        self.inst_res = float(self.paramfile_inputs['instres'])
        self.inst_res_err = float(self.paramfile_inputs['instreserr'])
        self.r_input = float(self.paramfile_inputs['rcloudmax'])
        self.t_input = float(self.paramfile_inputs['timeback'])
        
        self.cloud_fname = self.brains_dir + '/' + self.paramfile_inputs['cloudsfileout']
        self.param_fname = self.brains_dir + 'param/param'
        
        
        tin, wlin, line2d_in, line2d_err_in = read_input_file(self.paramfile_inputs['filedir'] + '/' + self.paramfile_inputs['line2dfile'])
        self.xline = tin
        self.wl_vals = wlin[0]/(1+self.z)
        self.vel_line = self.C_UNIT*( self.wl_vals - self.central_wl )/self.central_wl
        self.vel_line_out = self.vel_line * self.VEL_UNIT #km/s
        
        self.line2D = line2d_in
        self.line2D_err = line2d_err_in
        self.line2D_out = self.line2D.copy()
        self.line2D_err_out = self.line2D_err.copy()
        
        
            #In observed-frame
        self.xcont, self.ycont, self.yerr_cont = np.loadtxt(self.paramfile_inputs['filedir'] + '/' + self.paramfile_inputs['continuumfile'], unpack=True)
        self.xcont_out = self.xcont.copy()
        self.xline_out = self.xline.copy()
        
            #Put in rest-frame
        self.xcont /= (1+self.z)
        self.xline /= (1+self.z)

        

        self.ycont_out = self.ycont.copy()
        self.yerr_cont_out = self.yerr_cont.copy()
    
        
        self.r_input = float(self.paramfile_inputs['rcloudmax'])
        self.t_input = float(self.paramfile_inputs['timeback'])
        self.rmin, self.rmax, self.timeback = get_rset_tset(self.xline, self.xcont, self.r_input, self.t_input)

        # self.ntau = len(bp.results['tau_rec'][0])
        self.ntau = int(self.paramfile_inputs['ntau'])
        
        ################################
        # Get nparams
        self.nq = 1 + self.flag_trend
        self.ntrend = self.nq
        self.ndrw = 3
        self.nresp = 2
        
        if self.flag_trend_diff > 0:
            self.ndifftrend = 1
        else:
            self.ndifftrend = 0
            
        self.nvar = self.ndrw + self.ntrend + self.nresp + self.ndifftrend
        
        self.nnlr = 0
        if self.flag_nl >= 2:
            self.nnlr = 3
        
        self.nres = 1
        if self.flag_inst_res > 1:
            self.nres = len(self.xline)
            
        self.nlinecenter = 0
        if self.flag_linecenter > 0:
            self.nlinecenter = 1
        elif self.flag_linecenter < 0:
            self.nlinecenter = len(self.xline)

            
        ################################
        #Get xcont_recon
        tspan = self.xcont.max() - self.xcont.min()
        xcont_recon_min = self.xcont.min() - self.timeback - 10.
        
        xcont_recon_max = self.xcont.max() + max(.05*tspan, 20.)
        xcont_recon_max = max( xcont_recon_max, self.xline.max() + 10. )

        dt = (xcont_recon_max - xcont_recon_min)/(self.nrecon_cont-1)
        self.xcont_recon = np.array([ xcont_recon_min + j*dt for j in range(self.nrecon_cont) ])

        
        ################################
        #Get xline_recon
        xline_recon_min = self.xline.min() - min( .1*(self.xline.max() - self.xline.min()), 10. )
        if self.t_input <= 0.:
            xline_recon_min = max( xline_recon_min, xcont_recon_min + self.timeback )

        xline_recon_max = self.xline.max() + min( .1*(self.xline.max() - self.xline.min()), 10. )
        xline_recon_max = min( xline_recon_max, xcont_recon_max - 1. )
        
        dt = (xline_recon_max - xline_recon_min)/(self.nrecon_line-1)
        self.xline_recon = np.array([ xline_recon_min + j*dt for j in range(self.nrecon_line) ])
        
        
        ################################
        # Get pow_xcont, xcont_med
        tspan_cont = self.xcont.max() - self.xcont.min()    
        tspan = self.xline.max() - self.xcont.min()
        self.xcont_med = .5*(self.xcont.max() + self.xcont.min())
        
        self.pow_xcont = np.zeros(self.ndifftrend)
        for i in range(self.ndifftrend):
            self.pow_xcont[i] = ( (self.xcont.max() - self.xcont_med)**(i+2) - (self.xcont.min() - self.xcont_med)**(i+2) )/(i+2)/tspan_cont

        
        ################################
        # Get idx_resp
        
        #For P14 model
        self.nblrmodel = 17
        self.nblr = self.nblrmodel + self.nnlr + self.nres + self.nlinecenter + 1
        
        self.nparams = self.nrecon_cont + self.nblr + self.nvar
        self.idx_resp = self.nblr + self.ndrw + self.ntrend
        self.idx_difftrend = self.idx_resp + self.nresp
        self.idx_linecenter = self.nblrmodel + self.nnlr + self.nres
        
        ################################
        # Get line LC
        
        self.yline = np.trapz( self.line2D, x=self.vel_line, axis=1 )
        self.yerr_line = np.sqrt(   np.trapz( self.line2D_err**2, x=self.vel_line, axis=1 )   )
    
        self.yline_out = np.trapz( self.line2D, x=self.wl_vals*(1+self.z), axis=1 )
        self.yerr_line_out = np.sqrt( np.trapz( self.line2D_err**2, x=self.wl_vals*(1+self.z), axis=1 ) )
        
        self.line_err_mean = np.mean(self.line2D_err)            
            
        ################################
        # Rescale light curves
        
        self.cont_err_mean = np.mean(self.yerr_cont)
        
        cont_avg = np.mean(self.ycont)
        self.cont_scale = 1/cont_avg
        
        line_avg = np.mean(self.yline)
        self.line_scale = 1/line_avg
        
        self.ycont *= self.cont_scale
        self.yerr_cont *= self.cont_scale
        self.cont_err_mean *= self.cont_scale
        
        self.yline *= self.line_scale
        self.yerr_line *= self.line_scale
        self.line2D *= self.line_scale
        self.line2D_err *= self.line_scale
        self.line_err_mean *= self.line_scale
                    
        ################################
        # Extend velocities
        
        self.nvel_data_incr = 5
        self.nvel_data_ext = len(self.vel_line) + 2*self.nvel_data_incr

        dv = np.diff(self.vel_line)[0]
        self.vel_line_ext = np.zeros(self.nvel_data_ext)
        for i in range(self.nvel_data_incr+1):
            self.vel_line_ext[i] = self.vel_line[0] - (self.nvel_data_incr - i)*dv
            self.vel_line_ext[-1-i] = self.vel_line[-1] + (self.nvel_data_incr - i)*dv

        for i in range(len(self.vel_line)):
            self.vel_line_ext[i+self.nvel_data_incr] = self.vel_line[i]
            
        self.vel_line_ext_out = self.vel_line_ext * self.VEL_UNIT
        
        
        ################################
        # Rescale inst res
        self.inst_res_out = self.inst_res
        self.inst_res_err_out = self.inst_res_err
        
        self.inst_res /= self.VEL_UNIT
        self.inst_res_err /= self.VEL_UNIT
        
        
        ################################
        # Set unset variables
        self.model_params = None
        self.model_params_physical = None
        self.kde_dists = None
        self.kde_dists_physical = None
        self.param_bounds = None
        self.param_bounds_physical = None
        self.param_prior_types = None
        self.sample_tot = None
        self.sample_info = None
        self.sample_size = None
        
        
        self.nkde = 1000
        self.bw_method='scott'
        
        self.paramnames_tot = list(self.paramnames.copy())
        self.param_units_tot = list(self.param_units.copy())
        for i in range(self.nrecon_cont):
            self.paramnames_tot.append('TS{}'.format(i-self.nblrmodel))
            self.param_units_tot.append('f_lambda')
        
        
        
    def get_model_params(self, weights_file=None, valtype='mean'):
            
        if valtype == 'mean':
          func = bestfit_mean
        elif valtype == 'median':
          func = bestfit_median
        elif valtype == 'kde':
          func = bestfit_kde
        elif valtype == 'map':
          func = bestfit_map
          
  
        self.samples_tot = np.loadtxt(self.brains_dir + 'data/posterior_sample_2d.txt')
        if self.samples_tot.ndim == 1:
            self.sample_tot = np.reshape(self.samples_tot, (1, self.samples_tot.shape[0]))
        
        self.sample_info = np.loadtxt(self.brains_dir + 'data/posterior_sample_info_2d.txt')
        self.sample_size = len(self.samples_tot)
        
        if weights_file is not None:
            weights = np.loadtxt(weights_file)
        else:
            weights = np.ones(self.sample_size)
        
        
        self.model_params = np.zeros((self.nparams, 3))
        self.kde_dists = np.zeros((self.nparams, 2, self.nkde))
        for i in range(self.nparams):
            if i < self.nblr+self.nvar:
                if (self.paramnames[i] == 'SIGMA_SYS_CON') and (~self.flag_cont_err):
                    continue

            if valtype == 'kde':
                xvals, kde_vals, p50, errlo, errhi = bestfit_kde(self.samples_tot[:,i], weights, bw_method=self.bw_method, nkde=self.nkde)
              
                self.model_params[i] = [p50, errlo, errhi]
                self.kde_dists[i,0] = xvals
                self.kde_dists[i,1] = kde_vals
              
            else:
                self.model_params[i] = func(self.samples_tot[:,i], weights)        
        
        ################################
        # Priors
        
        prior_bounds0 = []
        prior_bounds1 = []
        prior_types = []
        with open(self.brains_dir + 'data/para_names_2d.txt', 'r') as f:
            lines = f.readlines()
            
            for n in range(7, 32):
                vals = lines[n][-50:].strip().split(' ')
                vals = [x for x in vals if x]

                prior_bound0 = float(vals[0])
                prior_bound1 = float(vals[1])
                prior_type = int(vals[2])

                prior_bounds0.append(prior_bound0)
                prior_bounds1.append(prior_bound1)
                prior_types.append(prior_type)
                
        prior_bounds0 = np.array(prior_bounds0)
        prior_bounds1 = np.array(prior_bounds1)
        prior_types = np.array(prior_types)
                
                
        self.param_bounds = np.zeros((self.nparams, 2))
        self.param_prior_types = np.zeros(self.nparams, dtype='U50')
        for i in range(self.nblr+self.nvar):
          
            if prior_types[i] == 2:
                self.param_prior_types[i] = 'Uniform'
            elif prior_types[i] == 1:
                self.param_prior_types[i] = 'Gaussian'
            
            if self.param_prior_types[i] == 'Uniform':
                self.param_bounds[i] = [prior_bounds0[i], prior_bounds1[i]]
            else:
                self.param_bounds[i] = [-np.inf, np.inf]
            
        
        for i in range(self.nblr+self.nvar, self.nparams):
            self.param_bounds[i] = [-np.inf, np.inf]
            self.param_prior_types[i] = 'Gaussian'


        return
        


    def get_model_params_physical(self, weights_file=None, valtype='mean'):
            
        if valtype == 'mean':
          func = bestfit_mean
        elif valtype == 'median':
          func = bestfit_median
        elif valtype == 'kde':
          func = bestfit_kde
        elif valtype == 'map':
          func = bestfit_map
      

        self.samples_tot = np.loadtxt(self.brains_dir + 'data/posterior_sample_2d.txt')
        if self.samples_tot.ndim == 1:
            self.sample_tot = np.reshape(self.samples_tot, (1, self.samples_tot.shape[0]))
        
        self.sample_info = np.loadtxt(self.brains_dir + 'data/posterior_sample_info_2d.txt')
        self.sample_size = len(self.samples_tot)
        
        
        self.samples_tot_physical = np.zeros_like(self.samples_tot)
        
        if weights_file is not None:
            weights = np.loadtxt(weights_file)
        else:
            weights = np.ones(self.sample_size)
      

        self.model_params_physical = np.zeros((self.nparams, 3))
        self.kde_dists_physical = np.zeros((self.nparams, 2, self.nkde))
        for i in range(self.nparams):
            if i >= self.nblr+self.nvar:
                samps = self.samples_tot[:,i]
            elif self.paramnames[i] == 'log_MBH':
                samps = self.samples_tot[:,i]/np.log(10) + 6 
            elif 'log' in self.paramnames[i]:
                samps = self.samples_tot[:,i]/np.log(10)
            elif self.paramnames[i] == 'INC':
                samps = np.arccos(self.samples_tot[:,i])*180/np.pi
            elif self.paramnames[i] == 'SIGMA_SYS_CON':
                if (~self.flag_cont_err):
                    continue
              
                samps = (np.exp(self.samples_tot[:,i]) - 1)*self.cont_err_mean
            elif self.paramnames[i] == 'SIGMA_SYS_LINE':
                vel_vals = (self.CVAL/1e5) * ( self.wl_vals/(1+self.z) - self.central_wl )/self.central_wl
                dv = np.abs(vel_vals[1] - vel_vals[0])/self.VEL_UNIT

                line_lc_err = np.sqrt( np.sum(self.line2D_err**2, axis=0) )*dv
                samps = (np.exp(self.samples_tot[:,i]) - 1)*np.mean(line_lc_err)
            elif self.paramnames[i] == 'DELTA_V_LINE':
                samps = self.samples_tot[:,i]*self.inst_res_err + self.inst_res
            else:
                samps = self.samples_tot[:,i]
          
          
            self.samples_tot_physical[:,i] = samps.copy()
          
            
            if valtype == 'kde':
                xvals, kde_vals, p50, errlo, errhi = bestfit_kde(samps, weights, bw_method=self.bw_method, nkde=self.nkde)
                
                self.model_params_physical[i] = [p50, errlo, errhi]
                self.kde_dists_physical[i,0] = xvals.copy()
                self.kde_dists_physical[i,1] = kde_vals.copy()
            else:
                self.model_params_physical[i] = func(samps, weights)  
            
            
            
            
            
        ################################
        # Priors
        
        prior_bounds0 = []
        prior_bounds1 = []
        prior_types = []
        with open(self.brains_dir + 'data/para_names_2d.txt', 'r') as f:
            lines = f.readlines()
            
            for n in range(7, 32):
                vals = lines[n][-50:].strip().split(' ')
                vals = [x for x in vals if x]

                prior_bound0 = float(vals[0])
                prior_bound1 = float(vals[1])
                prior_type = int(vals[2])

                prior_bounds0.append(prior_bound0)
                prior_bounds1.append(prior_bound1)
                prior_types.append(prior_type)
                
        prior_bounds0 = np.array(prior_bounds0)
        prior_bounds1 = np.array(prior_bounds1)
        prior_types = np.array(prior_types)
                
                
                
        self.param_bounds_physical = np.zeros((self.nparams, 2))
        self.param_prior_types = np.zeros(self.nparams, dtype='U50')
        for i in range(self.nblr+self.nvar):
          
            if prior_types[i] == 2:
                self.param_prior_types[i] = 'Uniform'
            elif prior_types[i] == 1:
                self.param_prior_types[i] = 'Gaussian'
            
            
            
            
            if self.paramnames[i] == 'log_MBH':
                lb = prior_bounds0[i]/np.log(10) + 6
                ub = prior_bounds1[i]/np.log(10) + 6
            elif 'log' in self.paramnames[i]:
                lb = prior_bounds0[i]/np.log(10)
                ub = prior_bounds1[i]/np.log(10)
            elif self.paramnames[i] == 'INC':
                lb = 0.
                ub = 90.
            elif self.paramnames[i] == 'SIGMA_SYS_CON':
                lb = (np.exp(prior_bounds0[i]) - 1)*self.cont_err_mean
                ub = (np.exp(prior_bounds1[i]) - 1)*self.cont_err_mean
            elif self.paramnames[i] == 'SIGMA_SYS_LINE':
                vel_vals = (self.CVAL/1e5) * ( self.wl_vals/(1+self.z) - self.central_wl )/self.central_wl
                dv = np.abs(vel_vals[1] - vel_vals[0])/self.VEL_UNIT

                line_lc_err = np.sqrt( np.sum(self.line2D_err**2, axis=0) )*dv
                lb = (np.exp(prior_bounds0[i]) - 1)*np.mean(line_lc_err)
                ub = (np.exp(prior_bounds1[i]) - 1)*np.mean(line_lc_err)
            elif self.paramnames[i] == 'DELTA_V_LINE':
                lb = prior_bounds0[i]*self.inst_res_err + self.inst_res
                ub = prior_bounds1[i]*self.inst_res_err + self.inst_res
            else:
                lb = prior_bounds0[i]
                ub = prior_bounds1[i]
            
            
            
            if self.param_prior_types[i] == 'Uniform':
                self.param_bounds_physical[i] = [lb, ub]
            else:
                self.param_bounds_physical[i] = [-np.inf, np.inf]
            
        
        for i in range(self.nblr+self.nvar, self.nparams):
            self.param_bounds_physical[i] = [-np.inf, np.inf]
            self.param_prior_types[i] = 'Gaussian'
    
    
        return 
      
      
      
    def sample_model_params(self, dtype='raw', nsamp=1000):
        if dtype == 'raw':     
            if self.model_params is None:
                self.get_model_params()
               
            mp = self.model_params
            mp_bounds = self.param_bounds
            
        elif dtype == 'physical':
            if self.model_params_physical is None:
                self.get_model_params_physical()
          
            mp = self.model_params_physical
            mp_bounds = self.param_bounds_physical





        vals_out = np.zeros((nsamp, self.nparams))
        
        for i in range(self.nparams):
            vals_out[:,i] = asymmetric_normal_draw(mp[i,0], mp[i,1], mp[i,2], nsamp, 
                                                   mp_bounds[i,0], mp_bounds[i,1])
            
        return vals_out
      
      
    def save_model_params(self, fname, dtype='raw'):
        if dtype == 'raw':
            if self.kde_dists is None:
                save_kde = False
            else:
                save_kde = True
                
        elif dtype == 'physical':
            if self.kde_dists_physical is None:
                save_kde = False
            else:
                save_kde = True 
        
        
        if dtype == 'raw':     
            if self.model_params is None:
                self.get_model_params()
             
            mp = self.model_params
            mp_bounds = self.param_bounds
            mp_priors = self.param_prior_types
            units = ['']*len(self.param_units_tot)
            
            kde_dists = self.kde_dists
            

        elif dtype == 'physical':
            if self.model_params_physical is None:
                self.get_model_params_physical()
        
            mp = self.model_params_physical
            mp_bounds = self.param_bounds_physical
            mp_priors = self.param_prior_types
            units = self.param_units_tot
            
            kde_dists = self.kde_dists_physical
          
          
        if save_kde:
            tab = Table([self.paramnames_tot, mp, mp_bounds, units, mp_priors, kde_dists],
                     names=['PARAMETER', 'VALUE', 'BOUNDS', 'UNIT', 'PRIOR_TYPE', 'KDE_DIST'])
      
        else:            
            tab = Table([self.paramnames_tot, mp, mp_bounds, units, mp_priors],
                        names=['PARAMETER', 'VALUE', 'BOUNDS', 'UNIT', 'PRIOR_TYPE'])
            
            
        tab.write(fname, overwrite=True)

        return
      
      
    def load_model_params(self, fname, dtype='raw'):
        tab = Table.read(fname)
        
        if 'KDE_DIST' in tab.colnames:
            load_kde = True
        else:
            load_kde = False
        
        mp = tab['VALUE'].data
        mp_bounds = tab['BOUNDS'].data
        mp_priors = tab['PRIOR_TYPE'].data
        
        if load_kde:
            kde_dists = np.zeros((self.nparams, 2, self.nkde))
            for i in range(self.nparams):
                kde_dists[i,0] = tab['KDE_DIST'][i][0]
                kde_dists[i,1] = tab['KDE_DIST'][i][1]
        
        if dtype == 'raw':
            self.model_params = mp
            self.param_bounds = mp_bounds
            if load_kde:
                self.kde_dists = kde_dists
            
        elif dtype == 'physical':
            self.model_params_physical = mp
            self.param_bounds_physical = mp_bounds
            if load_kde:
                self.kde_dists_physical = kde_dists
          
        self.param_prior_types = mp_priors
        return
      
      
    def load_kde_dists(self, fname, dtype='raw'):
        tab = Table.read(fname)
        
        kde_dists = np.zeros((self.nparams, 2, 1000))
        for i in range(self.nparams):
            kde_dists[i,0] = tab['XVALS'][i]
            kde_dists[i,1] = tab['KDE_VALS'][i]
          
        if dtype == 'raw':
            self.kde_dists = kde_dists
        elif dtype == 'physical':
            self.kde_dists_physical = kde_dists
          
        return