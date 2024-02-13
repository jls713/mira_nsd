#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: selection_function.py
# Description: selection function for NSD modelling

from scipy.ndimage import map_coordinates
import numpy as np
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import quad
import pandas as pd

class schultheis_map_3d(object):
    """
    Schultheis et al. 2014 3D map of the extinction in the inner Galaxy.
    
    The map is available at http://cdsarc.u-strasbg.fr/viz-bin/cat/J/A%2BA/566/A120
    
    The map is in the Galactic coordinates, with the longitude in the range [-10,10],
    the latitude in the range [-10,5] and the distance in the range [0.25,10.25] kpc.
    The map is sampled on a 200x151x21 grid.
    """
    def __init__(self):
        sch3d = Table.read('../../nsd/selection_function/schultheis_3d_map.fit')
        sch3d['glon'] = sch3d['GLON']-360.*(sch3d['GLON']>180.)
        sch3d.sort(['glon','GLAT'])
        x=sch3d[['E_H-K_%02i'%s for s in np.linspace(0.,100.,21)]].as_array()
        self.ehk=x.view(np.float64).reshape(x.shape + (-1,)).reshape((200,151,21))
        e_x=sch3d[['e_E_H-K_%02i'%s for s in np.linspace(0.,100.,21)]].as_array()
        self.e_ehk=e_x.view(np.float64).reshape(e_x.shape + (-1,)).reshape((200,151,21))

    def query(self, l, b, dist):
        '''
        Query the extinction map at a given Galactic longitude, latitude and distance.
        
        Args:
            l (float): Galactic longitude in degrees
            b (float): Galactic latitude in degrees
            dist (float): distance in kpc
        
        Returns:
            float: E(H-Ks) extinction
        '''
        l = l-360.*(l>180.)
        return map_coordinates(self.ehk, np.vstack([(l+10.)/0.1, (b+10.)/0.1, (dist-0.25)/.5]),
                               mode='nearest',order=1)
    def query_spread(self, l, b, dist):
        '''
        Query the spread in the extinction map at a given Galactic longitude, latitude and distance.

        Args:
            l (float): Galactic longitude in degrees
            b (float): Galactic latitude in degrees
            dist (float): distance in kpc

        Returns:
            float: spread in the E(H-Ks) extinction
        '''
        l = l-360.*(l>180.)
        return map_coordinates(self.e_ehk, np.vstack([(l+10.)/0.1, (b+10.)/0.1, (dist-0.25)/.5]),
                               mode='nearest',order=1)
    
def plr(P):
    """
    Period-luminosity relation for LMC Mira variables from Sanders+ 2023

    Args:
        P (float): period in days

    Returns:
        float: absolute Ks magnitude
    """
    return -7.01-3.73*(np.log10(P)-2.3)*(np.log10(P)<2.6)+(-0.3*3.73-6.99*(np.log10(P)-2.6))*(np.log10(P)>=2.6)

def ks_c(P,c):
    """
    Period-colour relation for LMC Mira variables from Sanders+ 2023

    Args:
        P (float): period in days
        c (str): filter name
    
    Returns:
        float: colour (Ks-c)
    """
    abc = {'J':[-5.9,-3.11,-6.87],'H':[-6.69,-3.34,-6.86],
           '4_5':[-7.51,-3.83,-7.64], '3_6':[-7.41,-3.97,-7.37], 
           '5_8':[-7.68,-3.83,-7.81],
           '8_0':[-7.86,-3.9,-8.5]}
    return (-7.01-3.73*(np.log10(P)-2.3)*(np.log10(P)<2.6)+(-0.3*3.73-6.99*(np.log10(P)-2.6))*(np.log10(P)>=2.6))-\
            (abc[c][0]+abc[c][1]*(np.log10(P)-2.3)*(np.log10(P)<2.6)+(0.3*abc[c][1]+abc[c][2]*(np.log10(P)-2.6))*(np.log10(P)>=2.6))

class convolution_interp(object):
    def __init__(self, width_grid = np.logspace(np.log10(0.3),np.log10(10.),30)):
        xx = np.logspace(np.log10(0.01),np.log10(20.), 50)
        yy = np.array([[quad(lambda x: 1./(1.+np.exp(-x))*np.exp(-(xxx-x)**2/2./width**2)/np.sqrt(2.*np.pi*width**2), 
                             -np.max([60.,5.*width]),np.max([60.,5.*width]))[0] for xxx in xx]
        for width in width_grid])
        yy[(xx[np.newaxis,:]>0.)&(yy<0.5)]=1.
        yy[(xx[np.newaxis,:]<0.)&(yy>0.5)]=0.
        self.interp = RegularGridInterpolator((width_grid,xx), yy, bounds_error=False, fill_value=None)
    def __call__(self, width, x):
        return 1.*(x<0.)+np.sign(x)*self.interp((width,np.abs(x)))


class selection_function(object):
    """
    Selection function for the NSD Mira variables.

    The selection function is a function of the Galactic longitude, latitude, distance and period.
    """
    def __init__(self, plr=plr, width=None, data_file=None, ext_multiplier=1., high_cut_off = 20.,
                 ext_map=schultheis_map_3d):
        self.AK_EHK = 1.306*ext_multiplier
        self.C = convolution_interp()
        self.smap = ext_map()
        self.Sk = lambda k,P: .5*(1.-np.tanh((10.3-k)/.4))*.9*.5*(1.-np.tanh((k-high_cut_off)/.2))
        if data_file is not None:
            s = pd.read_csv(data_file)
            self.plr = interp1d(s['period'], s['plr'], bounds_error=False, fill_value='extrapolate')
            width = interp1d(s['period'], s['plr_width'], bounds_error=False, fill_value='extrapolate')
        else:
            self.plr = plr
        if width is not None:
            self.Sk = lambda k, P: self.C(width(P)/.4, (k-10.3)/.4)*.9*.5*(1.-np.tanh((k-high_cut_off)/.2))

    def S(self,l,b,s,P):
        """
        Selection function for the NSD Mira variables.

        Args:
            l (float): Galactic longitude in degrees.
            b (float): Galactic latitude in degrees.
            s (float): Distance in kpc.
            P (float): Period in days.
        
        Returns:
            float: Selection function.
        """
        return self.Sk(self.plr(P)+5.*np.log10(s*100.)+self.smap.query(l*np.ones_like(s),
                                                                       b*np.ones_like(s),s)*self.AK_EHK,
                                                                       P)
    def Snorm(self,l,b,s,P,gc_dist=8.275):
        """
        Normalized selection function for the NSD Mira variables.

        The selection function is normalized to the value at the Galactic center.

        Args:
            l (float): Galactic longitude in degrees.
            b (float): Galactic latitude in degrees.
            s (float): Distance in kpc.
            P (float): Period in days.
            gc_dist (float): Distance to the Galactic center in kpc.

        Returns:
            float: Normalized selection function.
        """
        return self.S(l,b,s,P)/self.S(l,b,gc_dist*np.ones_like(s),P)
    


def circumstellar_extinction(colour_val, colour, magnitude, model_grid, teff=2600., tinner=600.):
    '''
    Finds the circumstellar dust extinction given a colour excess for a given colour (c1-c2) and magnitude.

    Parameters
    ----------
    colour_val : float
        The colour excess value.
    colour : tuple
        The colour (c1-c2) for which the colour excess is given (c1,c2).
    magnitude : str
        The magnitude for which the colour excess is given.
    model_grid : joint_grids object
        The model grid to use.
    teff : float
        The effective temperature of the star.
    tinner : float
        The inner temperature of the dust.
    
    Returns
    -------
    float
        The circumstellar dust extinction.
    '''
    index1 = np.argmin(np.abs(model_grid.teff-teff))
    index2 = np.argmin(np.abs(model_grid.tinner-tinner))
    colour_arr = np.reshape(model_grid.colour_model(colour[0], colour[1], flat=True),(8, 5, 50))[index1, index2, :]
    mag_arr = np.reshape(model_grid.magnitudes_model(magnitude, flat=True),(8, 5, 50))[index1, index2, :]
    return interp1d(colour_arr, mag_arr, fill_value='extrapolate', bounds_error=False)(colour_val)