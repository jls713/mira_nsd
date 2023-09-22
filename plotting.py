#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: plotting.py
# Description: plotting functions for NSD modelling project

from plotting_general import add_inner_ticks
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

# From Zhang & Sanders (2023)

def age_period(P, with_gc=True, use_nikzat=False):
    if use_nikzat:
        return 13.8 / (1 + np.power(P/310., 3.6))
    if not with_gc:
        return 6.9 * (1+np.tanh((330.-P)/400.))
    else:
        return 14.7/2. * (1+np.tanh((330.-P)/308.))
def dage_dperiod(P, with_gc=True, use_nikzat=False):
    if use_nikzat:
        return -3.6*np.power(P/310., 3.6)/13.8/P*np.power(age_period(P, False, True),2.)
    if not with_gc:
        return -0.01725 * np.cosh((330.-P)/400.)**(-2)
    else:
        return -0.02386 * np.cosh((330.-P)/308.)**(-2)
def period_age(a, with_gc=True, use_nikzat=False):
    if use_nikzat:
        return 310. * np.power(13.8/a - 1., 1./3.6)
    if not with_gc:
        return 330. - 400.*np.arctanh(a/6.9-1)
    else:
        return 330. - 308.*np.arctanh(a/14.7*2.-1)

def shade_bfa(with_gc=True, add_label=True):
    if not with_gc:
        plt.axvspan(np.log10(period_age(7.8)),np.log10(period_age(6.8)),color='gray',alpha=0.1)
        if add_label:
            plt.annotate('Bar\nformation\nepoch', (np.log10(period_age(7.3)),4e1), fontsize=17,ha='center',va='top')
    else:
        delta = 0.7
        plt.axvspan(np.log10(period_age(7.8+delta)),np.log10(period_age(6.8+delta)),color='gray',alpha=0.1)
        if add_label:
            plt.annotate('Bar\nformation\nepoch', (np.log10(period_age(7.3+delta)),4e1), fontsize=17,ha='center',va='top')

def format_period_axis(with_secondary_axis=True):
    plt.xlim(1.98,3.)
    plt.gca().set_xticks(np.log10(np.array([100,200,300,400,600,1000])))
    plt.gca().set_xticks(np.log10(np.array([150,250,350,450,500,550,600,650,750,800,850,900,950])),minor=True)
    plt.gca().set_xticklabels([100,200,300,400,600,1000])

    plt.gca().tick_params(axis="y",direction="in")
    plt.gca().tick_params(axis="x",direction="in")
    plt.gca().tick_params(axis="y",direction="in",which='minor')
    plt.gca().tick_params(axis="x",direction="in",which='minor')
    plt.gca().tick_params(axis='y', which='minor', bottom=False)
    plt.gca().yaxis.set_ticks_position('both')

    plt.xlabel(r'Period [d]')
    if with_secondary_axis:
        # add secondary axis with age
        ax2 = plt.gca().twiny()
        ax2.set_xlabel('Age [Gyr]', labelpad=10)
        ax2.xaxis.set_label_position('top')
        ax2.xaxis.tick_top()
        ax2.set_xticks(np.log10(period_age(np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.]))),labels=[1,2,3,4,5,6,7,8,9,10,11,12])
        ax2.set_xticks(np.log10(period_age(np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.])+0.5)),minor=True)
        ax2.set_xlim(1.98,3.)
        plt.gca().tick_params(axis="x",direction="in")
        plt.gca().tick_params(axis="x",direction="in",which='minor')
    else:
        plt.gca().xaxis.set_ticks_position('both')

## Plotting results from fits
def spline_marginalization(knots, y, x, Ns=100):
    '''
    Marginalize over the spline parameters
    
    Args:
        knots (array): knots of the spline
        y (array): values of the spline at the knots for a set of samples
        x (array): x values at which to evaluate the spline
        Ns (int): number of samples to use for the marginalization
    
    Returns:
        array: 16th, 50th, and 84th percentile of the marginalized spline
    '''
    if x is None:
        x = np.linspace(knots[0], knots[-1], 1000)
    return x, np.nanpercentile([InterpolatedUnivariateSpline(knots, m)(x) 
                                for m in y[np.random.randint(0,len(y),Ns)]], [16, 50, 84], axis=0)
                             
def plot_spline(mc, fld, xrr = None, color=sns.color_palette('colorblind')[0], one_over=False,
                display_knots=True, with_bracket=True, ls='dashed', scaling=1., label=None):
    '''
    Plot the marginalized spline
    
    Args:
        mc (numpyro chain): chain from the fit
        fld (str): name of the field in the chain
        xrr (array): x values at which to evaluate the spline
        color (str): color of the spline
    '''
    xrr, spline = spline_marginalization(mc.aux_parameters['aux_knots'], 
                                         np.array(mc.samples()[fld]),
                                         xrr, Ns=100)
    l,=plt.plot(xrr, np.exp((1-2*one_over)*spline[1])*scaling, color=color, ls=ls, label=label)
    if with_bracket:
        plt.fill_between(xrr, 
                         np.exp((1-2*one_over)*spline[0])*scaling, 
                         np.exp((1-2*one_over)*spline[2])*scaling, color=color, alpha=0.3)

    if display_knots:
        plt.plot(mc.aux_parameters['aux_knots'],
                 scaling*np.exp((1-2*one_over)*np.nanmedian(np.array(mc.samples()[fld]),axis=0)),
                 'x',c='gray',mew=1,mec='k');

def plot_errorbar_list(mc,k,scaling=1., one_over=False, ps='o',label=None, fade=None, nudge=0.):
    '''
    Plot errorbars for a list of chains
    
    Args:
        mc (list): list of chains
        k (str): name of the field in the chain
        scaling (float): scaling factor for the y-axis
        one_over (bool): whether to plot 1/variable
        ps (str): marker style
        label (str): label for the plot
        fade (float): log10 of the period below which to fade the points
        nudge (float): nudge the points by this amount in log10(period)
    '''
    for mm, pp in mc:
        a=1.
        if fade is not None:
            if np.log10(.5*(pp[0]+pp[1]))<fade:
                a=0.5
        plt.errorbar(np.log10(.5*(pp[0]+pp[1]))+nudge, np.exp((1-2*one_over)*np.nanmedian(mm.samples()[k]))*scaling,
                    xerr=np.array([[np.log10(.5*(pp[0]+pp[1]))-np.log10(pp[0]),
                                    np.log10(pp[1])-np.log10(.5*(pp[0]+pp[1]))]]).T,
                    yerr = np.array([[np.exp((1-2*one_over)*np.nanmedian(mm.samples()[k]))*scaling-np.exp((1-2*one_over)*np.nanpercentile(mm.samples()[k],16,axis=0))*scaling,
                                        np.exp((1-2*one_over)*np.nanpercentile(mm.samples()[k],84,axis=0))*scaling-np.exp((1-2*one_over)*np.nanmedian(mm.samples()[k]))*scaling]]).T,
                                    color=sns.color_palette()[0],mew=1.5,mec='k',ms=7,fmt=ps,label=label,alpha=a);
        label=None