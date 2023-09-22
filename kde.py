#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: kde.py
# Description: estimating the density of a sample using KDE

import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from KDEpy import FFTKDE


class KDE(object):
    def __init__(self):
        pass

    def __call__(self, xx):
        return self.interpolator(xx)

class KDE_scipy(KDE):

    def __init__(self, data, weights=None, **kde_kwargs):
        ggk = gaussian_kde(data, weights=weights,
                           bw_method='scott', **kde_kwargs)
        bw = ggk.covariance_factor()
        xmin, xmax = np.nanpercentile(data, [0., 100.])
        xx = np.linspace(xmin, xmax, np.int64((xmax - xmin) / (bw / 3.)))
        self.interpolator = interp1d(
            xx, ggk(xx), kind='linear', bounds_error=False, fill_value=0.0)

class KDE_fft(KDE):

    def znorm(self, x):
        return (x - self.shift) / self.scale
    
    def interpolator(self, xx):
        return self.interpolate(self.znorm(xx)) / self.scale

    def __init__(self, data, weights=None, gridsize = 'auto', 
                 force_normalization=True, bw_scale=1., **kde_kwargs):
        self.scale = np.std(data)
        self.shift = np.nanmedian(data)

        if weights is not None:
            neff = np.sum(weights)**2 / np.sum(weights**2)
        else:
            neff = len(data)

        bw = bw_scale*neff**-0.2
        ggk = FFTKDE(kernel='gaussian', bw=bw, **kde_kwargs)

        ps = ggk.kernel.practical_support(bw)
        
        if gridsize == 'auto':
            gridsize = np.int64(np.max(np.power(2, np.ceil(np.log2(np.ptp(data)/self.scale/(ps/2.))))))
        
        if((np.max(data) - np.min(data)) / self.scale / gridsize > ps/2.):
            print('''Warning: gridsize is too small for the data.''')
            if force_normalization:
                print('''The normalization is forced to be correct ''' 
                      '''so the result will approximately OK, '''
                      '''but check with a higher gridsize if you want '''
                      '''to be sure (or try using gridsize=auto))''')

        ggk = ggk.fit(self.znorm(data), weights).evaluate(gridsize)

        grid = ggk[1]
        if force_normalization:
            # if gridsize is small (and above warning issued), the normalization is not correct, 
            # so we do it manually just to be sure
            grid /= np.sum(ggk[1])*np.diff(ggk[0])[0]

        self.interpolate = interp1d(
            ggk[0], grid, kind='linear', 
            bounds_error=False, fill_value=0.0)

class KDE_fft_ND(KDE):

    def znorm(self, x):
        return np.dot(self.scale, (x - self.shift[np.newaxis, :]).T)

    def gridnorm(self, x):
        return (x - self.minimum[:, np.newaxis]) / self.scaling[:, np.newaxis]
    
    def interpolator(self, xx):
        return map_coordinates(self.grid, self.gridnorm(self.znorm(xx))) * self.norm

    def __init__(self, data, weights=None, gridsize = 2**8, 
                 force_normalization=True, bw_scale=1.):

        cov = np.cov(data, rowvar=False, bias=True)
        w, v = np.linalg.eig(cov)
        self.scale = np.dot(np.dot(v, np.diag(1. / np.sqrt(w))), v.T)
        self.shift = np.nanmedian(data, axis=0)
        self.norm = np.linalg.det(self.scale)

        if weights is not None:
            neff = np.sum(weights)**2 / np.sum(weights**2)
        else:
            neff = len(data)

        bw = bw_scale * neff**(-1. / (np.shape(data)[1] + 4))
        ggk = FFTKDE(kernel='gaussian', bw=bw)
        ps = ggk.kernel.practical_support(bw)
        
        if gridsize == 'auto':
            gridsize = np.int64(np.max(np.power(2, np.ceil(np.log2(np.ptp(self.znorm(data),axis=1)/(ps/2.))))))

        if np.any(np.ptp(self.znorm(data),axis=1) / gridsize > ps/2.):
            print('''Warning: gridsize is too small for the data.''')
            if force_normalization:
                print('''The normalization is forced to be correct ''' 
                      '''so the result will approximately OK, '''
                      '''but check with a higher gridsize if you want '''
                      '''to be sure (or try using gridsize=auto))''')

        ggk = ggk.fit(self.znorm(data).T, weights).evaluate(gridsize)

        self.grid = ggk[1].reshape([gridsize] * np.shape(data)[1])


        self.minimum = np.array([np.min(np.atleast_2d(ggk[0])[:, ii])
                                 for ii in range(np.shape(data)[1])])

        xyz = ggk[0].reshape([gridsize] * np.shape(data)[1]
                             + [np.shape(data)[1]])

            
        self.scaling = np.array([np.max(np.diff(xyz, axis=ii))
                                 for ii in range(np.shape(data)[1])])
        
        if force_normalization:
            # if gridsize is small (and above warning issued), the normalization is not correct, 
            # so we do it manually just to be sure
            self.grid /= np.sum(self.grid)*np.prod(self.scaling)
