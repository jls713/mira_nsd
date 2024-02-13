#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: model.py
# Description: Model for the Mira sample

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import emcee
import numpyro
numpyro.enable_x64(True)
numpyro.set_platform("gpu")
from numpyro_model import *
from jax.scipy.special import logsumexp as jax_logsumexp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from df_jax import binney_df_jax_spline, quasiisothermal_df_jax_spline
from kde import KDE_fft_ND

import agama
agama.setUnits(length=1, velocity=1, mass=1e10)   # 1 kpc, 1 km/s, 1e10 Msun

import sormani_potential

## Some assumptions/defaults
gc_dist = 8.275 # GC distance
sspread_gl=0.2  # Width of distance sampling distribution 
vscale=200.     # Width of velocity sampling distribution for denominator
vspread=vscale*np.ones(3)
vmeans=np.array([-250.,0.,0.]) # means of velocity sampling distribution
## Small shifts required to transform between (l,b) and the system where SgrA* is at centre
shift_sgrA = np.deg2rad(np.array([0., -0.05576432, -0.04616002]))*gc_dist

def make_default_galaxy_model():
    # Create the combined Sormani+22 and Axisymmetric Portail+ potential
    sPot = agama.Potential(agama.Potential('sormani_2021_potential.agama'), 
                        agama.Potential('portail_2017_noCNC_axi.agama'))

    # The initial potential used for the iterations of the original Sormani+22 model
    init_potential = agama.Potential('sormani_2021_initial_potential.agama')

    # Build AGAMA galaxy model
    sDF = sormani_potential.sormani_DF(init_potential)

    return agama.GalaxyModel(sPot,sDF)

class Data_Driven_Model(object):
    def __init__(self):
        self.central_density = 90.
        self.scalelength = np.deg2rad(1.8) # in rad
        data = pd.read_csv('background_bulge.csv')
        self.kde = KDE_fft_ND(data[['pml','pmb']].values*4.74, gridsize='auto')
    def __call__(self, l, b, pmlb):
        return self.central_density*np.exp(-np.abs(b-shift_sgrA[2]/gc_dist)/self.scalelength)*self.kde(pmlb)
    def density(self, l, b):
        return self.central_density*np.exp(-np.abs(b-shift_sgrA[2]/gc_dist)/self.scalelength)

default_galaxy_model = make_default_galaxy_model()
default_action_finder = default_galaxy_model.af
default_nsd_mass = default_galaxy_model.df.totalMass()
data_driven_model = Data_Driven_Model()

def flip(x,deg=False):
    """
    flip coordinate from (0,360) to (-180,180)
    """
    if deg is True:
        return x - 360.*(x>180.)
    else:
        return x - 2.*np.pi*(x>np.pi)
    
def lnG(x, mu, s):
    """
    Log of Gaussian distribution

    Args:
        x (float or array): Value(s) at which to evaluate the log of the Gaussian
        mu (float): Mean of the Gaussian
        s (float): Standard deviation of the Gaussian
    """
    return -.5*(x-mu)**2/s**2-.5*np.log(2.*np.pi*s**2)

def agama_GalactocentricFromGalactic(coords):
    """
    Convert from Galactic to Galactocentric coordinates using AGAMA

    Args:
        coords (array): Array of Galactic coordinates (l,b,d,pml,pmb,vlos) in units of (deg,deg,kpc,km/s/kpc,km/s/kpc,km/s)

    Returns:
        array: Array of Galactocentric coordinates (x,y,z,vx,vy,vz) in units of (kpc,kpc,kpc,km/s,km/s,km/s)
    """
    Xcoords = np.array(agama.getGalactocentricFromGalactic(lon=coords[:,0],
                                                           lat=coords[:,1],
                                                           dist=coords[:,2], 
                                                           pmlon=coords[:,3], 
                                                           pmlat=coords[:,4], 
                                                           vlos=coords[:,5],
                                                           galcen_distance=gc_dist,
                                                           galcen_v_sun=(11.1, 6.411*4.74*gc_dist, 7.25))
                                                          ).T
    
    Xcoords[:,:3]-=shift_sgrA
    Xcoords[:,3:5]*=-1
    return Xcoords

def agama_GalacticFromGalactocentric(coords):
    """
    Convert from Galactocentric to Galactic coordinates using AGAMA

    Args:
        coords (array): Array of Galactocentric coordinates (x,y,z,vx,vy,vz) in units of (kpc,kpc,kpc,km/s,km/s,km/s)

    Returns:
        array: Array of Galactic coordinates (l,b,d,pml,pmb,vlos) in units of (deg,deg,kpc,km/s/kpc,km/s/kpc,km/s)
    """
    coords[:,:3]+=shift_sgrA
    coords[:,3:5]*=-1
    
    Xcoords = np.array(agama.getGalacticFromGalactocentric(x=coords[:,0],
                                                           y=coords[:,1],
                                                           z=coords[:,2], 
                                                           vx=coords[:,3], 
                                                           vy=coords[:,4], 
                                                           vz=coords[:,5],
                                                           galcen_distance=gc_dist,
                                                           galcen_v_sun=(11.1, 6.411*4.74*gc_dist, 7.25))
                                                          ).T
    return Xcoords

def find_frequencies(actions, pot, kR=0.25, kz=0.25, Jtildemin=10.):
    """
    Find the epicyclic frequencies and circular radius given a set of actions

    Args:
        actions (array): Array of actions
        pot (agama.Potential): AGAMA potential
        kR (float): Radial frequency ratio parameter
        kz (float): Vertical frequency ratio parameter
        Jtildemin (float): Minimum value of Jtilde to consider bound

    Returns:
        array: Array of epicyclic frequencies and circular radius
    """
    
    Jtilde = jnp.abs(actions[:,2]) + kR * actions[:,0] + kz * actions[:,1]
    Rc = pot.Rcirc(L=np.array(Jtilde))
    xyz = np.column_stack((Rc, Rc*0., Rc*0.))
    force, deriv=pot.forceDeriv(xyz)
    
    kappa=np.sqrt(-deriv[:,0]-3*force[:,0]/Rc)
    nu=np.sqrt(-deriv[:,2])
    Omega = np.array(Jtilde) / Rc**2

    return np.column_stack((kappa, nu, Omega, Rc))

def clean_actions(actions, lnweights, Ndata, Nsamples):
    '''
    If any problematic actions (unbound), set to the median and then assign large weight so no contribution in sums

    Args:
        actions (array): Array of actions
        lnweights (array): Array of log weights
        Ndata (int): Number of stars
        Nsamples (int): Number of samples
    
    Returns:
        array: Cleaned actions
    '''
    clean = np.any(np.isnan(actions),axis=1)
    actions[clean]=np.nanmedian(actions,axis=0)
    lnweights[clean.reshape(Ndata, Nsamples)]=100000.
    return actions, lnweights

def safelog(x):
    y = np.ones_like(x)*-700
    y[~((x==0.)|np.isnan(x))]=np.log(x[~((x==0.)|np.isnan(x))])
    return y

def generate_coordinate_samples(data, sspread, N_num_samples, 
                                include_radial_velocities=False, 
                                include_pml=True,
                                ):
    '''
    Generate samples of the coordinates and velocities for numerator computation

    Args:
        data (dict): Dictionary of data
        sspread (float): Spread of the distance sampling distribution
        N_num_samples (int): Number of samples to generate
        include_radial_velocities (bool): Whether to include radial velocities
        include_pml (bool): Whether to include proper motions

    Returns:
        num_coords, dist_lnweights, vel_lnweights, pm_lnweights (array): 
            Arrays of numerator coordinates, distance log weights, velocity log weights, and proper motion log weights    
    '''

    Ndata = len(data['l'])    

    ## Sampling distribution for the missing distances and line-of-sight velocities
    s_samples = np.abs(np.random.normal(gc_dist, sspread, (Ndata, N_num_samples))) # abs to avoid negative distances -- hacky
    vlos_samples = np.random.normal(vmeans[2], vscale, (Ndata, N_num_samples))

    if include_radial_velocities:
        # If we have radial velocities, use them instead of gaussian sampling
        print('Warning -- background model not yet implemented for radial velocities')
        vel = data['maser_SiO_rv'].values.copy() # Use SiO RVs if available
        vel[vel!=vel] = data['maser_OH_rv'].values[vel!=vel] # Otherwise use OH RVs
        where_vel = (vel==vel) 
        vlos_samples[where_vel] = np.random.normal(vel[where_vel], 
                                                   10.*np.ones(np.count_nonzero(where_vel)),  # assign uncertainty as 10 km/s
                                                   (N_num_samples,np.count_nonzero(where_vel))).T
        vel_lnweights = lnG(vlos_samples, vmeans[2], vscale)*(vel!=vel)[:,np.newaxis]
    else:
        vel_lnweights = lnG(vlos_samples, vmeans[2], vscale)

    if include_pml:
        # Sample proper motions using the covariance matrix
        key = jax.random.PRNGKey(42)
        pm_rslt=jax.random.multivariate_normal(key,
                                            jnp.vstack([data['pml'].values, data['pmb'].values]).T,
                                            jnp.array([[data['epml'].values**2,
                                                        data['pml_pmb_corr'].values*data['epml'].values*data['epmb'].values],
                                                        [data['pml_pmb_corr'].values*data['epml'].values*data['epmb'].values,
                                                        data['epmb'].values**2]]).T,
                                            shape=(N_num_samples,Ndata))
        pm_lnweights = np.zeros((Ndata, N_num_samples))
    else: 
        # If don't want mu_l, sample from a Gaussian but then include weights
        pm_rslt = np.vstack([np.random.normal(vmeans[0]/gc_dist,  vscale/gc_dist,      (N_num_samples, Ndata))[:,:,np.newaxis].T,
                                   np.random.normal(data['pmb'].values, data['epmb'].values, (N_num_samples, Ndata))[:,:,np.newaxis].T]).T
        pm_lnweights = lnG(pm_rslt[:,:,0].T, vmeans[0]/gc_dist, vscale/gc_dist)

    dist_lnweights = lnG(s_samples, gc_dist, sspread)

    num_coords = np.column_stack([np.repeat(data['l_rad'].values, N_num_samples), 
                                  np.repeat(data['b_rad'].values, N_num_samples),
                                  s_samples.flatten(),
                                  np.array(pm_rslt[:,:,0]).T.flatten(),
                                  np.array(pm_rslt[:,:,1]).T.flatten(),
                                  vlos_samples.flatten()])
    
    return num_coords, dist_lnweights, vel_lnweights, pm_lnweights

    
def generate_numerator_and_denominator_samples(data, 
                                               bar_model, 
                                               SF=None, 
                                               galaxy_model=default_galaxy_model, 
                                               N_num_samples=100, 
                                               N_denom_samples=100, 
                                               sspread=sspread_gl, 
                                               include_radial_velocities=False,
                                               include_pml=True,
                                               nsd_mass=default_nsd_mass,
                                               use_grid_integral=False):
    '''
    Generate the samples required to perform the per-star marginalization over the uncertainties in the proper motions.
    
    Args:
        data (DataFrame): DataFrame of data
        bar_model: Bar model from Portail+17
        SF (SelectionFunction): Selection function
        galaxy_model (class): Default galaxy model
        N_num_samples (int): Number of samples to use for the numerator
        N_denom_samples (int): Number of samples to use for the denominator
        sspread (float): Width of the distance sampling distribution
        include_selection_function (bool): Whether to include the selection function
        include_radial_velocities (bool): Whether to include the radial velocities
        include_pml (bool): Whether to include the longitudinal proper motion
        nsd_mass (float): Mass of the NSD in 10^10 Msun (default ~0.09)
        use_grid_integral (bool): Whether to use compute the distance integral using a regular grid (slower) or a Monte Carlo integral (faster) (default False -- useful for testing)
    Returns:
        dict: Dictionary of samples
    '''
    
    np.random.seed(42)

    if SF is None:
        include_selection_function=False
    else:
        include_selection_function=True

    Ndata = len(data['l'])    
    
    ## Numerator samples
    ## =================

    # ## Sampling distribution for the missing distances and line-of-sight velocities
    num_coords, dist_lnweights, vel_lnweights, pm_lnweights = generate_coordinate_samples(data,sspread,N_num_samples,include_radial_velocities,include_pml)
    num_lnweights = dist_lnweights + vel_lnweights + pm_lnweights

    # Convert to actions, clean and add Jacobian
    num_actions = default_galaxy_model.af(agama_GalactocentricFromGalactic(num_coords))
    num_actions = np.column_stack([num_actions, find_frequencies(num_actions, default_galaxy_model.potential)])
    num_actions, num_lnweights = clean_actions(num_actions, num_lnweights, Ndata, N_num_samples)
    # We divide by the weighting terms so they are the negative of the Jacobian
    num_lnweights += (- 4.*np.log(num_coords[:,2]) - np.log(np.cos(num_coords[:,1]))).reshape(Ndata, N_num_samples) # Jacobian
    num_lnweights += np.log(N_num_samples) # weight by number of samples

    # Add selection function
    if include_selection_function:
        num_lnweights -= np.log(SF.S(np.rad2deg(num_coords[:,0]), 
                                     np.rad2deg(num_coords[:,1]), 
                                     num_coords[:,2],
                                     np.repeat(data['period'].values, N_num_samples)
                                     ).reshape((Ndata, N_num_samples)))
        
    num_nsdweights = jax_logsumexp(safelog(galaxy_model.df(num_actions[:,:3]).reshape(Ndata, N_num_samples))-num_lnweights,axis=1)
    num_lnweights -= np.log(nsd_mass) # weight by mass of NSD (must be after previous line or mass included twice -- because galaxy_model includes NSD mass)
    
    # Bar model density
    # Weights not necessary in bar model as already marginalised over distance and line-of-sight velocity
    # But weights necessary if we want to discount pml

    num_ddweights = np.log(np.sum((data_driven_model(num_coords[:,0],
                                                     num_coords[:,1],
                                                     num_coords[:,3:5],
                                                     )/np.exp(pm_lnweights).flatten()).reshape((Ndata, N_num_samples)),axis=1)) - np.log(N_num_samples)
    
    if include_selection_function:

        max_dist=20.
        if use_grid_integral:
            # Do integral over distance as linearly-spaced grid
            Nd=500
            dist_grid = np.linspace(1., max_dist, Nd)
            delta_s = dist_grid[1]-dist_grid[0]
            Vcoords = np.column_stack([np.tile(dist_grid, Ndata * N_num_samples),
                                       np.repeat(num_coords[:,3], Nd),
                                       np.repeat(num_coords[:,4], Nd)])
            num_pFweights = np.log(np.sum((SF.S(np.repeat(np.rad2deg(num_coords[:,0]),Nd),
                                                np.repeat(np.rad2deg(num_coords[:,1]),Nd),
                                                Vcoords[:,0], 
                                                np.repeat(np.repeat(data['period'].values, N_num_samples), Nd))*
                                           bar_model.evaluate(np.repeat(np.rad2deg(num_coords[:,0]),Nd),
                                                              np.repeat(np.rad2deg(num_coords[:,1]),Nd),
                                                              Vcoords, include_density=True)
                                            /np.repeat(np.exp(pm_lnweights), Nd)
                                          ).reshape((Ndata, N_num_samples*Nd)),axis=1)) \
                            - np.log(N_num_samples) + np.log(delta_s)
            
            # Assume that the data-driven model is flat in distance
            SFint = np.sum((SF.S(np.repeat(data['l'].values, Nd),np.repeat(data['b'].values, Nd),
                                 np.tile(dist_grid, Ndata), np.repeat(data['period'].values, Nd))*delta_s).reshape((Ndata, Nd)),axis=1)/max_dist 
            num_ddweights += np.log(SFint)

        else:        
            # Do integral using MC sampling -- use 10 times as many samples as above and broader distance spread
            sspread_bar = 3.0
            num_coords_bar, dist_lnweights_bar, _,pm_lnweights_bar = generate_coordinate_samples(data,sspread_bar,
                                                                                                 N_num_samples*10,
                                                                                                 False,
                                                                                                 include_pml)
            num_pFweights = np.log(np.sum((SF.S(np.rad2deg(num_coords_bar[:,0]),
                                                np.rad2deg(num_coords_bar[:,1]),
                                                num_coords_bar[:,2], 
                                                np.repeat(data['period'].values, N_num_samples*10))*
                                           bar_model.evaluate(np.rad2deg(num_coords_bar[:,0]),
                                                              np.rad2deg(num_coords_bar[:,1]),
                                                              num_coords_bar[:,2:5],
                                                              include_density=True)
                                            /np.exp(pm_lnweights_bar.flatten())
                                            /np.exp(dist_lnweights_bar.flatten())).reshape((Ndata, N_num_samples*10)),axis=1)) \
                                - np.log(N_num_samples*10)
            
            # Assume that the data-driven model is flat in distance
            SFint = np.sum((SF.S(np.rad2deg(num_coords_bar[:,0]),
                                 np.rad2deg(num_coords_bar[:,1]),
                                 num_coords_bar[:,2], np.repeat(data['period'].values, N_num_samples*10))
                            /np.exp(pm_lnweights_bar.flatten())).reshape((Ndata, N_num_samples*10)),axis=1)/max_dist/(N_num_samples*10)
            num_ddweights += np.log(SFint)

    else:
        num_pFweights = np.log(np.sum((bar_model.evaluate(np.rad2deg(num_coords[:,0]), 
                                                        np.rad2deg(num_coords[:,1]), 
                                                        num_coords[:,3:5],
                                                        include_density=True
                                                        )/np.exp(pm_lnweights.flatten())).reshape((Ndata, N_num_samples)),axis=1)) - np.log(N_num_samples)

    ## Denominator samples -- similar to above but proper motions (and rvs) also sampled
    ## ===================
    
    s_samples = np.random.normal(gc_dist, sspread, (Ndata, N_denom_samples))
    v_samples = np.random.normal(vmeans,  vspread, (Ndata, N_denom_samples, 3))
    denom_lnweights = lnG(s_samples, gc_dist, sspread)
    denom_lnweights += np.sum(lnG(v_samples, vmeans[np.newaxis,np.newaxis,:], vspread[np.newaxis,np.newaxis,:]), axis=-1)
    
    denom_coords = np.column_stack([np.repeat(data['l_rad'].values, N_denom_samples), 
                                    np.repeat(data['b_rad'].values, N_denom_samples),
                                    s_samples.flatten(),
                                    v_samples[:,:,0].flatten()/s_samples.flatten(),
                                    v_samples[:,:,1].flatten()/s_samples.flatten(), 
                                    v_samples[:,:,2].flatten()])
    
    denom_actions = default_galaxy_model.af(agama_GalactocentricFromGalactic(denom_coords))
    denom_actions = np.column_stack([denom_actions, find_frequencies(denom_actions, default_galaxy_model.potential)])
    denom_actions, denom_lnweights = clean_actions(denom_actions, denom_lnweights, Ndata, N_denom_samples)
    # We divide by the weighting terms so they are the negative of the Jacobian
    denom_lnweights += - 2.*np.log(s_samples) - np.log(np.cos(denom_coords[:,1].reshape(Ndata, N_denom_samples))) # Jacobian
    denom_lnweights += np.log(N_denom_samples) # weight by number of samples

    if include_selection_function:
        denom_lnweights -= np.log(SF.S(np.rad2deg(denom_coords[:,0]), np.rad2deg(denom_coords[:,1]), denom_coords[:,2],
                                    np.repeat(data['period'].values, N_denom_samples)).reshape((Ndata, N_denom_samples)))
        
    denom_nsdweights = jax_logsumexp(safelog(galaxy_model.df(denom_actions[:,:3]).reshape(Ndata, N_denom_samples))-denom_lnweights,axis=1)

    denom_lnweights -= np.log(nsd_mass) # weight by mass of NSD (must be after previous line or mass included twice)
    
    denom_ddweights = np.log(data_driven_model.density(data['l_rad'].values,
                                                       data['b_rad'].values))
    
    if include_selection_function:
        if use_grid_integral:
            # Do integral over distance as linearly-spaced grid -- MC samples require a Gaussian of around ~3 kpc width
            Nd=500
            dist_grid = np.linspace(0.5, 20., Nd)
            delta_s = dist_grid[1]-dist_grid[0]
            
            denom_pFweights = np.log(np.sum((SF.S(np.repeat(data['l'].values, Nd),
                                                  np.repeat(data['b'].values, Nd),
                                                  np.tile(dist_grid, Ndata), 
                                                  np.repeat(data['period'].values, Nd))*
                                             bar_model.density(np.repeat(data['l'].values, Nd), 
                                                               np.repeat(data['b'].values, Nd),
                                                               np.tile(dist_grid, Ndata))).reshape((Ndata, Nd)),axis=1)) + np.log(delta_s)
        else:
            s_bar_samples = np.abs(np.random.normal(gc_dist, sspread_bar, (Ndata, N_denom_samples*10))) # abs to avoid negative distances -- hacky
            denom_coords = np.column_stack([np.repeat(data['l'].values, N_denom_samples*10), 
                                            np.repeat(data['b'].values, N_denom_samples*10),
                                            s_bar_samples.flatten()])
            denom_pFweights = np.log(np.sum((SF.S(denom_coords[:,0],
                                                  denom_coords[:,1],
                                                  denom_coords[:,2], 
                                                  np.repeat(data['period'].values, N_denom_samples*10))
                                            *bar_model.density(denom_coords[:,0],
                                                               denom_coords[:,1],
                                                               denom_coords[:,2])
                                            / np.exp(lnG(denom_coords[:,2], gc_dist, sspread_bar))).reshape((Ndata, N_denom_samples*10)),
                                            axis=1)/(N_denom_samples*10))
        denom_ddweights += np.log(SFint)

    else:
        denom_pFweights = np.log(bar_model.density(np.rad2deg(data['l_rad'].values), 
                                                   np.rad2deg(data['b_rad'].values)))
    

    # Return dictionary of results
    return {'num_actions':jnp.array(num_actions).reshape(num_lnweights.shape+(7,)), 
            'num_ln_weights': jnp.array(num_lnweights), 
            'num_log10P':jnp.array(np.log10(np.repeat(data['period'].values, N_num_samples))).reshape(num_lnweights.shape),
            'denom_actions':jnp.array(denom_actions).reshape(denom_lnweights.shape+(7,)), 
            'denom_ln_weights': jnp.array(denom_lnweights), 
            'denom_log10P': jnp.array(np.log10(np.repeat(data['period'].values, N_denom_samples))).reshape(denom_lnweights.shape),
            'num_pFweights': jnp.array(num_pFweights),
            'denom_pFweights': jnp.array(denom_pFweights),
            'num_nsdweights': jnp.array(num_nsdweights),
            'denom_nsdweights': jnp.array(denom_nsdweights),
            'num_ddweights': jnp.array(num_ddweights),
            'denom_ddweights': jnp.array(denom_ddweights)}
    

@partial(jax.jit, static_argnums=(2,3,4,6,))
def logL_numpyro(data, params, 
                 use_s22_weights=True, 
                 df_type='quasiisothermal', 
                 include_background_weight=True,
                 aux_knots=None,
                 background_model='p17'):
    """
    Log-likelihood function for the model.

    I restructured this function to have kwargs rather than a dictionary of parameters,
    because then it is jit-able. The dictionary of parameters can be created outside
    the function and passed in as a static argument. However, it doesn't seem to lead
    to a speed-up.

    Parameters
    ----------
    data : dict
        Dictionary of data (result of generate_numerator_and_denominator_samples)
    params : dict
        Dictionary of model parameters
    use_s22_weights : bool, optional
        Whether to use the S22 weights, by default True
    df_type : str, optional
        Type of DF to use, by default 'quasiisothermal'
    include_background_weight : bool, optional
        Whether to include the background weight, by default True
    aux_knots : array, optional
        Array of knots for the spline, by default None
    background_model : str, optional
        Which background model to use, by default 'p17'
    
    Returns
    -------
    logL : float
        Log-likelihood
    """

    # merge params and aux_params into a single dictionary
    # params = {**params, **aux_params}
    if 'aux_knots' not in params.keys():
        params['aux_knots'] = aux_knots

    if not use_s22_weights:
        if df_type == 'exponential':
            bdf = binney_df_jax_spline(params['ln_Rdisk_coeffs'], 
                                    params['ln_Hdisk_coeffs'], 
                                    params['ln_deltaR_coeffs'],
                                    ln_Jv0=params['ln_Jv0'], 
                                    ln_Jd0=params['ln_Jd0'],
                                    aux_knots=params['aux_knots'], 
                                    mass=1., 
                                    vO=params['vO'], 
                                    )
        elif df_type == 'quasiisothermal':
            bdf = quasiisothermal_df_jax_spline(params['ln_Rdisk'], 
                                                params['ln_Hdisk'], 
                                                params['ln_sigmaR0'],
                                                params['ln_RsigmaR'],
                                                aux_knots=params['aux_knots'],
                                                )

        num   = jax_logsumexp(bdf(data['num_actions'],   data['num_log10P'],   log=True) - data['num_ln_weights'],   axis=1)
        denom = jax_logsumexp(bdf(data['denom_actions'], data['denom_log10P'], log=True) - data['denom_ln_weights'], axis=1)

    else:
        num, denom = data['num_nsdweights'], data['denom_nsdweights']

    if include_background_weight:
        if params['aux_knots'] is not None:
            ln_w_P = InterpolatedUnivariateSpline(params['aux_knots'], params['ln_w_P'], k=3)
        else:
            ln_w_P = lambda x: params['ln_w_P']

        which = {'p17':'pF', 'dd':'dd'}[background_model]
        num =   jnp.logaddexp(num,   ln_w_P(data['num_log10P'][:,0])   + data[f'num_{which}weights'])
        denom = jnp.logaddexp(denom, ln_w_P(data['denom_log10P'][:,0]) + data[f'denom_{which}weights'])

    return num-denom

def sampling_lnL_emcee(params, bdf, galaxy_model=default_galaxy_model):
    """
    Function to sample the likelihood using emcee

    Parameters
    ----------
    params : array_like
        Array of parameters to sample the likelihood at.
    bdf : callable
        Function to compute the distribution function.
    galaxy_model : callable
        Galaxy model to compute actions and frequencies.

    Returns
    -------
    outputs : array_like
        Array of log-likelihoods.
    """
    
    actions = galaxy_model.af(agama_GalactocentricFromGalactic(params[:,:6]))
    # check if bdf is type quasiisothermal_df_jax_spline
    if type(bdf)==quasiisothermal_df_jax_spline:
        actions = np.vstack([actions.T, find_frequencies(actions, galaxy_model.potential).T]).T

    # Safety checks
    clean = ~np.any(np.isnan(actions),axis=1)
    outputs = -np.ones(params.shape[0])*np.inf
    lBDF = np.array(bdf(jnp.array(actions), params[:,-1], log=True))
    clean &= (~np.isnan(lBDF)&~np.isinf(lBDF)&~(params[:,-1]<2.)&~(params[:,-1]>3.)&~
              (params[:,0]<-np.pi)&~(params[:,0]>np.pi)&~
              (params[:,1]<-np.pi/2.)&~(params[:,1]>np.pi/2.)&~(params[:,2]<0.))
    
    outputs[clean] = 4.*np.log(params[clean,2]) + np.log(np.cos(params[clean,1])) + lBDF[clean]

    return outputs

def generate_samples(model,
                     bar_model=None,
                     galaxy_model=default_galaxy_model, 
                     N_emcee_iterations=1000, 
                     data = None, 
                     Nsamples=10000,
                     add_errors=False,
                     nsd_mass=default_nsd_mass,
                     SF=None,
                     onskylimit=1.6):
    """
    Function to generate samples from the posterior using emcee

    Parameters
    ----------
    model: either numpyro MCMC object or scipy.optimize.minimize object
        Object from which to generate samples.
    bar_model : Portail bar model
        Bar model to use.
    galaxy_model : callable
        Galaxy model to use.
    N_emcee_iterations : int, optional
        Number of iterations to run emcee for. The default is 1000.
    data : pandas DataFrame, optional
        Data to use for the likelihood. The default is None.
    Nsamples : int, optional
        Number of samples to use for the likelihood. The default is 10000.
    add_errors : bool, optional
        Include proper motion errors in the likelihood. The default is False.
    nsd_mass : float, optional
        Mass of the nuclear star disc in 10^10 Msun. The default is ~0.09
    SF : selection function, optional
        Selection function to use. The default is None.

    Returns
    -------
    final_chain : array_like
        Array of samples from the posterior.
    final_logL : array_like
        Array of log-likelihoods for the samples.
    """

    if model.aux_parameters['use_s22_weights']:
        
        print('Using S22 weights')

        s22_samples_x, samples_m = galaxy_model.sample(Nsamples*10)
        samples = agama_GalacticFromGalactocentric(s22_samples_x)
        samples = np.vstack([samples.T, np.random.uniform(2.,3.,len(samples))]).T

    else:    
        nwalkers = 500
        ndim = 7
        if model is not None:
            best_params = {k:np.nanmedian(model.samples()[k],axis=0) for k in model.samples().keys()}
        elif model.type=='minimization':
            model.aux_parameters['aux_knots']=None
            if model.aux_parameters['df_type'] == 'quasiisothermal': 
                best_params = {k:np.float64(model.x[i]) 
                            for i,k in enumerate(['ln_Rdisk', 'ln_Hdisk', 
                                                'ln_sigmaR0', 'ln_RsigmaR'])}
            else:
                best_params = {k:np.float64(model.x[i]) 
                            for i,k in enumerate(['ln_Rdisk', 'ln_Hdisk', 'ln_deltaR',
                                                    'ln_Jv0', 'ln_Jd0'])}
        print(best_params)

        params = {**best_params, **model.aux_parameters}

        if model.aux_parameters['df_type'] == 'quasiisothermal':
            bdf = quasiisothermal_df_jax_spline(params['ln_Rdisk'], 
                                    params['ln_Hdisk'], 
                                    params['ln_sigmaR0'],
                                    params['ln_RsigmaR'],
                                    aux_knots=params['aux_knots'], 
                                    mass=1.)
        else:
            bdf = binney_df_jax_spline(params['ln_Rdisk'], 
                                    params['ln_Hdisk'], 
                                    params['ln_deltaR'],
                                    ln_Jv0=params['ln_Jv0'], 
                                    ln_Jd0=params['ln_Jd0'],
                                    aux_knots=params['aux_knots'], 
                                    mass=1., 
                                    vO=params['vO'])
        
        em = emcee.EnsembleSampler(nwalkers, ndim, sampling_lnL_emcee, args=(bdf,galaxy_model,), vectorize=True)

        p0 = np.vstack([np.random.uniform(-np.deg2rad(1.5), np.deg2rad(1.5), nwalkers),
                        np.random.uniform(-np.deg2rad(1.5), np.deg2rad(1.5), nwalkers),
                        np.random.normal(gc_dist, sspread_gl,nwalkers),
                        np.random.normal(-6.*4.74,4.*4.74,nwalkers),
                        np.random.normal(0.,2.*4.74,nwalkers),
                        np.random.normal(0.,150.,nwalkers),
                        np.random.uniform(2.,3.,nwalkers)]).T
        
        mc=em.run_mcmc(p0, N_emcee_iterations, thin=5)

        samples = em.flatchain[-Nsamples*10:,:]
        samples_m = np.ones_like(samples[:,0])*nsd_mass/len(samples)
    
    if model.aux_parameters['include_background_weight'] and bar_model is not None:
        
        print('Including background')

        fltr =  (np.abs(bar_model.lbr[:,0])<np.deg2rad(onskylimit))
        fltr &= (np.abs(bar_model.lbr[:,1])<np.deg2rad(onskylimit))

        bar_samples = np.vstack([bar_model.lbr[fltr].T, 
                                    np.random.uniform(2.,3.,np.count_nonzero(fltr)),
                                    np.ones(np.count_nonzero(fltr))]).T
        
        if bar_model.use_vtransverse:
            bar_samples[:,3:5] /= bar_samples[:,2:3]
        
        samples = np.vstack([samples.T, np.zeros(len(samples))]).T # add identifiers
        joint_samples = np.vstack([samples, bar_samples])

        if 'aux_knots' in model.aux_parameters and model.aux_parameters['aux_knots'] is not None and model.type=='mcmc':
            print('Spline model for relative weight')
            ln_w_P_c = jnp.nanmedian(model.samples()['ln_w_P'],axis=0)
            ln_w_P = InterpolatedUnivariateSpline(model.aux_parameters['aux_knots'], 
                                                    ln_w_P_c, k=3)
        else:
            if model.type=='mcmc':
                print('Constant model for relative weight')
                ln_w_P = lambda x: np.nanmedian(model.samples()['ln_w_P'])*np.ones(len(x))
            elif model.type=='minimization':
                print('Minimization result for relative weight')
                ln_w_P = lambda x: model.x[-1]*np.ones(len(x))
            else:
                ln_w_P = lambda x: np.zeros(len(x))


        probs = np.concatenate([samples_m, bar_model.m[fltr]*np.exp(ln_w_P(bar_samples[:,6]))])
        if SF is not None:
            probs *= SF.S(joint_samples[:,0],joint_samples[:,1],joint_samples[:,2],joint_samples[:,6])

        probs /= np.sum(probs)
        samples = joint_samples[np.random.choice(len(joint_samples), size=Nsamples*5, p=probs)]

    if data is not None:
        # If we have data, we find nearest samples to the data in (l,b,log10P)
        period_scale = 0.05
        deg_scale = np.deg2rad(0.05)
        N_samples_per_star=100
        nearest = np.argsort((samples[:,6][:,np.newaxis]-np.log10(data['period'].values)[np.newaxis,:])**2./period_scale**2.
                             +(samples[:,0][:,np.newaxis]-flip(data['l_rad'].values,deg=False)[np.newaxis,:])**2./deg_scale**2.
                             +(samples[:,1][:,np.newaxis]-data['b_rad'].values[np.newaxis,:])**2./deg_scale**2,axis=0)[:N_samples_per_star].flatten()
        samples = samples[nearest]

        if add_errors:
            # If we want to add errors, we sample from the proper motion covariance matrix of the data
            key = jax.random.PRNGKey(42)
            pm_rslt=jax.random.multivariate_normal(key,jnp.zeros((len(data),2)),
                                        jnp.array([[data['epml'].values**2,data['pml_pmb_corr'].values*data['epml'].values*data['epmb'].values],
                                                [data['pml_pmb_corr'].values*data['epml'].values*data['epmb'].values,data['epmb'].values**2]]).T,
                                        shape=(N_samples_per_star,len(data)))
            
            samples[:,3]+=pm_rslt[:,:,0].flatten()
            samples[:,4]+=pm_rslt[:,:,1].flatten()

    else:
        samples = samples[-Nsamples:]
    
    return samples

def recompute_denom(mc, n_d):
    """
    Function to recompute the denominator of the likelihood using the samples

    Parameters
    ----------
    mc : numpyro MCMC object
        Object from which to generate samples.
    n_d : dict
        Dictionary of samples from the numerator and denominator.

    Returns
    -------
    logL : array_like
        Array of log-likelihoods for the samples.
    """
    
    bdf = quasiisothermal_df_jax_spline(jnp.nanmedian(mc.samples()['ln_Rdisk'],axis=0), 
                                        jnp.nanmedian(mc.samples()['ln_Hdisk'],axis=0), 
                                        jnp.nanmedian(mc.samples()['ln_sigmaR0'],axis=0), 
                                        jnp.nanmedian(mc.samples()['ln_RsigmaR'],axis=0),
                                        aux_knots=mc.aux_parameters['aux_knots']
                                        )
    return jax_logsumexp(bdf(n_d['denom_actions'], n_d['denom_log10P'], log=True) - n_d['denom_ln_weights'], axis=1)