#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bar_model_interpolant.py
# Description: interpolating KDE estimates for the phase-space density of the Portail+ N-body model

from read_Portail import Portail
from kde import KDE_fft, KDE_fft_ND
import agama
import numpy as np
from multiprocessing import Pool
from itertools import repeat


class Portail_interp(Portail):
    """
    Class for interpolating KDE estimates for the phase-space density of the Portail+ N-body model

    Args:
        lrange (list): range of longitudes to interpolate over
        brange (float): range of latitudes to interpolate over
        N (int): number of points to interpolate over
        radius (float): radius to use for KDE
        include_vlos (bool): whether to include the line-of-sight velocity in the KDE
        shift_sgrA (np.array): shift to apply to the coordinates to put Sgr A* at the origin
        gc_dist (float): distance to the Galactic centre
        bw_scale (float): factor to scale the bandwidth by

    We work in a coordinate system shifted by (lshift, bshift). The density distribution is approximately
    symmetric about this centre so we only need to cover half of the b-space. This means all input coordinates
    are shifted.
    """
    def __init__(self, 
                 lrange_degree = [-1.5,1.5], 
                 brange_degree = 1.5,
                 N=20, 
                 radius_degree=0.1, 
                 include_vlos=False, 
                 include_distance=False,
                 shift_sgrA=np.deg2rad(np.array([0., -0.05576432, -0.04616002]))*8.275, 
                 gc_dist=8.275,
                 gc_vel=(11.1, 6.411*4.74*8.275, 7.25),
                 bw_scale=0.5,
                 parallel=False):
        
        super(Portail_interp, self).__init__()
        self.cut_stuff()
        self.symmetrise()
        self.realign()

        self.include_vlos = include_vlos
        self.include_distance = include_distance
        self.bw_scale = bw_scale

        lrange = np.deg2rad(lrange_degree)
        brange = np.deg2rad(brange_degree)
        self.radius = np.deg2rad(radius_degree)

        self.lbr=np.array(agama.getGalacticFromGalactocentric(self.y+shift_sgrA[0],
                                                              -self.x+shift_sgrA[1],
                                                              self.z+shift_sgrA[2],
                                                              100.*self.vy,
                                                              -100.*self.vx,
                                                              100.*self.vz, 
                                                              galcen_distance=gc_dist,
                                                              galcen_v_sun=gc_vel)).T
        
        self.lshift = shift_sgrA[1]/gc_dist
        self.bshift = shift_sgrA[2]/gc_dist

        self.lrange = np.linspace(lrange[0], lrange[1], N) - self.lshift
        self.brange = np.linspace(0., brange + np.abs(self.bshift), N)
        self.lbgrid = np.meshgrid(self.lrange, self.brange, indexing='ij')

        if parallel:
            p = Pool(32)
            self.pft = p.starmap(self.kde_fn, zip(repeat(self), 
                                                    self.lbgrid[0].flatten(), 
                                                    self.lbgrid[1].flatten()))
        else:
            self.pft = [KDE_fft_ND(self.lbr[self.fltr_fn(ll,bb),3-self.include_distance:5+self.include_vlos], 
                                   weights=self.m[self.fltr_fn(ll,bb)], bw_scale=self.bw_scale)
                                   for ll,bb in 
                                   zip(self.lbgrid[0].flatten(), self.lbgrid[1].flatten())]
        
        if include_distance:    
            self.dens_s = [KDE_fft(self.lbr[self.fltr_fn(ll,bb),2], 
                                 weights=self.m[self.fltr_fn(ll,bb)], bw_scale=self.bw_scale)
                                 for ll,bb in 
                                 zip(self.lbgrid[0].flatten(), self.lbgrid[1].flatten())]
            
        self.dens = np.reshape([np.sum(self.m[self.fltr_fn(ll,bb)]) / self.radius**2 / np.cos(bb) / np.pi
                    for ll,bb in zip(self.lbgrid[0].flatten(),self.lbgrid[1].flatten())], 
                    self.lbgrid[0].shape)

    def fltr_fn(self, ll, bb):
        return np.hypot((self.lbr[:,0]-self.lshift-ll)*np.cos(bb),
                        self.lbr[:,1]-self.bshift-bb)<self.radius

    @staticmethod
    def kde_fn(self,ll,bb):
        '''
        KDE function for parallelisation

        Args:
            self (Portail_interp): instance of Portail_interp
            ll (float): Galactic longitude
            bb (float): Galactic latitude
        '''
        return KDE_fft_ND(self.lbr[self.fltr_fn(ll,bb),3-self.include_distance:5+self.include_vlos],
                            weights=self.m[self.fltr_fn(ll,bb)], bw_scale=self.bw_scale)
    
    def find_bracket_indices(self, sorted_array, values):
        '''
        Find the indices of the bracketing values in a sorted array
        
        Args:
            sorted_array (np.array): sorted array to search
            values (np.array): values to search for
        
        Returns:
            np.array: indices of the bracketing values'''
        lower_indices = np.searchsorted(sorted_array, values, side='left')
        lower_indices[lower_indices == 0] += 1
        return lower_indices-1

    def evaluate(self, Lcoords, Bcoords, Vcoords, include_density=True, return_density=False):
        '''
        Evaluate the density at a given point in (l,b,v) space

        Args:
            Lcoords (float): longitude in degrees
            Bcoords (float): latitude in degrees
            Vcoords (float): proper motion in km/s/kpc
            include_density (bool): whether to include the density in the KDE estimate
            return_density (bool): whether to return the density as well as the KDE estimate
        
        Returns:
            float: KDE estimate at the given point
        '''

        Lcoords = np.deg2rad(Lcoords - 360.*(Lcoords>180.)) - self.lshift
        Bcoords = np.deg2rad(Bcoords) - self.bshift
        
        gg=np.reshape([self.pft[i](Vcoords) for i in range(len(self.pft))], 
                      (*self.lbgrid[0].shape,len(Vcoords)))
        
        ll = self.find_bracket_indices(self.lrange, Lcoords)
        bb = self.find_bracket_indices(self.brange, np.abs(Bcoords))
        delta_l = (Lcoords - self.lrange[ll])/(self.lrange[ll+1]-self.lrange[ll])
        delta_b = (np.abs(Bcoords) - self.brange[bb])/(self.brange[bb+1]-self.brange[bb])
        ar = np.arange(len(Vcoords))

        if include_density:
            inT = gg[ll,bb,ar]*self.dens[ll,bb]*(1-delta_l)*(1-delta_b) + \
                gg[ll+1,bb,ar]*self.dens[ll+1,bb]*delta_l*(1-delta_b) + \
                gg[ll,bb+1,ar]*self.dens[ll,bb+1]*(1-delta_l)*delta_b + \
                gg[ll+1,bb+1,ar]*self.dens[ll+1,bb+1]*delta_l*delta_b
        else:
            inT = gg[ll,bb,ar]*(1-delta_l)*(1-delta_b) + \
                gg[ll+1,bb,ar]*delta_l*(1-delta_b) + \
                gg[ll,bb+1,ar]*(1-delta_l)*delta_b + \
                gg[ll+1,bb+1,ar]*delta_l*delta_b
            
        if return_density:
            return inT, self.dens[ll  ,bb  ]*(1-delta_l)*(1-delta_b) + \
                        self.dens[ll+1,bb  ]*   delta_l *(1-delta_b) + \
                        self.dens[ll  ,bb+1]*(1-delta_l)*   delta_b + \
                        self.dens[ll+1,bb+1]*   delta_l *   delta_b

        return inT


    def density(self, Lcoords, Bcoords, Distcoords=None):
        '''
        Evaluate the density at a given point in (l,b) and optionally distance space

        Args:
            Lcoords (float): longitude in degrees
            Bcoords (float): latitude in degrees
            Distcoords (float): distance in kpc (optional -- only needed if include_distance=True)
        
        Returns:
            float: KDE estimate at the given point
        '''

        Lcoords = np.deg2rad(Lcoords - 360.*(Lcoords>180.)) - self.lshift
        Bcoords = np.deg2rad(Bcoords) - self.bshift
        
        ll = self.find_bracket_indices(self.lrange, Lcoords)
        bb = self.find_bracket_indices(self.brange, np.abs(Bcoords))
        delta_l = (Lcoords - self.lrange[ll])/(self.lrange[ll+1]-self.lrange[ll])
        delta_b = (np.abs(Bcoords) - self.brange[bb])/(self.brange[bb+1]-self.brange[bb])

        if Distcoords is not None:
            gg=np.reshape([self.dens_s[i](Distcoords) for i in range(len(self.pft))], 
                        (*self.lbgrid[0].shape,len(Distcoords)))        
            ar = np.arange(len(Distcoords))
            return gg[ll,bb,ar]*self.dens[ll,bb]*(1-delta_l)*(1-delta_b) + \
                gg[ll+1,bb,ar]*self.dens[ll+1,bb]*delta_l*(1-delta_b) + \
                gg[ll,bb+1,ar]*self.dens[ll,bb+1]*(1-delta_l)*delta_b + \
                gg[ll+1,bb+1,ar]*self.dens[ll+1,bb+1]*delta_l*delta_b
        else:
            return self.dens[ll,bb]*(1-delta_l)*(1-delta_b) + \
                self.dens[ll+1,bb]*delta_l*(1-delta_b) + \
                self.dens[ll,bb+1]*(1-delta_l)*delta_b + \
                self.dens[ll+1,bb+1]*delta_l*delta_b