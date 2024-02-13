#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: data.py
# Description: Utilities for loading the Mira sample and crossmatching with external catalogs

import sys
import numpy as np
import pandas as pd
import sqlutilpy
from wsdb_query import *

sys.path.append('../../mira/mira-date/')
from pm_transform import ml_mb
from wsdb_query import *


def load_mira_sample(lowb = False, 
                     sig_clip=3., 
                     reliable=True, 
                     high_amp=True, 
                     convert_pm=True, 
                     per_cut=[100.,1000.], 
                     scale_proper_motion_errors=False):
    """
    Load the Mira sample

    Args:
        lowb (bool): If True, only load the Miras with |b|<0.4
        sig_clip (float): Sigma clipping threshold for proper motions
        reliable (bool): If True, only load the reliably identified Miras
        high_amp (bool): If True, only load the Miras with amplitude>0.4
        convert_pm (bool): If True, convert proper motions from mas/yr to km/s/kpc
        per_cut (list): Period cut for the Miras
        scale_proper_motion_errors (bool): If True, scale the proper motion errors by UWE and an additional factor of 1.1
    
    Returns:
        pandas.DataFrame: Mira sample
    """
    
    data = pd.read_csv('final_nsd_mira_full_cols.csv')

    updated_crossmatch = pd.DataFrame(sqlutil.local_join(
                """
                    select * from mytable as m
                    left join lateral (select *, q3c_dist(m.ra_,m.dec_,s.ra,s.dec) from leigh_smith.virac2_rc2_all as s
                    where s.duplicate=0 and q3c_join(m.ra_, m.dec_,s.ra,s.dec,1./3600) 
                    order by q3c_dist(m.ra_,m.dec_,s.ra,s.dec) asc limit 1)
                    as tt on  true  order by xid """,
                'mytable', (data['ra'].values, data['dec'].values, 
                            np.arange(len(data))), ('ra_', 'dec_', 'xid'),**wsdb_kwargs))
    
    fields_to_replace = ['ra','dec','ra_error','dec_error','pmra','pmdec','pmra_error','pmdec_error','pmra_pmdec_corr',
                         'parallax','parallax_error','parallax_pmra_corr','parallax_pmdec_corr'
                         ]
    for field in fields_to_replace:
        data[field + '_old'] = data[field].values
        data[field] = updated_crossmatch[field]

    for field in ['pml','pmb','epml','epmb']:
        data[field + '_old'] = data[field].values
    
    extra_fields = ['chisq','uwe']
    for field in extra_fields:
        data[field] = updated_crossmatch[field]

    new_fields = ['sourceid', 'phot_z_mean_mag', 'phot_z_std_mag', 'phot_z_n_epochs',
       'z_n_obs', 'z_n_det', 'z_n_amb', 'phot_y_mean_mag', 'phot_y_std_mag',
       'phot_y_n_epochs', 'y_n_obs', 'y_n_det', 'y_n_amb', 'phot_j_mean_mag',
       'phot_j_std_mag', 'phot_j_n_epochs', 'j_n_obs', 'j_n_det', 'j_n_amb',
       'phot_h_mean_mag', 'phot_h_std_mag', 'phot_h_n_epochs', 'h_n_obs',
       'h_n_det', 'h_n_amb', 'phot_ks_mean_mag', 'phot_ks_std_mag',
       'phot_ks_n_epochs', 'ks_n_obs', 'ks_n_det', 'ks_n_amb', 'q3c_dist']
    for field in new_fields:
        data[field + '_rc2all'] = updated_crossmatch[field]

    data = ml_mb(data)
    
    data.to_csv('final_nsd_mira_full_cols_updated_proper_motions.csv')

    data['l_rad'], data['b_rad'] = np.deg2rad(data['l']), np.deg2rad(data['b'])

    if convert_pm:
        data['pml'] *= 4.74047 # mas/yr to km/s/kpc
        data['pmb'] *= 4.74047 # mas/yr to km/s/kpc
        data['epml'] *= 4.74047 # mas/yr to km/s/kpc
        data['epmb'] *= 4.74047 # mas/yr to km/s/kpc

    if scale_proper_motion_errors:
        data['epml'] *= data['uwe']
        data['epmb'] *= data['uwe']
        data['epml'] *= 1.1
        data['epmb'] *= 1.1

    fltr = [True]*len(data)
    print(np.count_nonzero(fltr))
    if reliable:
        fltr &= (data['unreliable']==0)
        print(np.count_nonzero(fltr), np.count_nonzero(~(data['unreliable']==0)))

    if high_amp:
        fltr &= (data['amplitude']>0.4)
        print(np.count_nonzero(fltr), np.count_nonzero(~(data['amplitude']>0.4)))

    if sig_clip is not None:
        for compt in ['l','b']:
            fltr_PM = (data['pm%s'%compt]==data['pm%s'%compt])
            fltr_PM &= (np.abs(data['pm%s'%compt]-np.nanmedian(data['pm%s'%compt]))<np.nanstd(data['pm%s'%compt].values)*sig_clip)
            fltr &= fltr_PM
        print(np.count_nonzero(fltr), np.count_nonzero(~fltr_PM))

    if lowb:
        fltr &= (np.abs(data['b'])<0.4)        
        print(np.count_nonzero(fltr), np.count_nonzero(~(np.abs(data['b'])<0.4)))

    if per_cut is not None:
        fltr &= (data['period']>per_cut[0])&(data['period']<per_cut[1])   
        print(np.count_nonzero(fltr), np.count_nonzero(~((data['period']>per_cut[0])&(data['period']<per_cut[1]))))
    
    data['joint_rv'] = data['maser_SiO_rv'].values.copy()
    data.loc[data['joint_rv']!=data['joint_rv'], 'joint_rv'] = data['maser_OH_rv'].values[data['joint_rv']!=data['joint_rv']]

    return data[fltr].reset_index(drop=True)


def cm_glimpse(data, radeccols=['ra', 'dec'], cols=None, cmradius=1., include_single_epoch=False):
    '''
    Crossmatch a table of sources with the GLIMPSE catalogue.

    Parameters:
        data (pandas.DataFrame): table of sources
        radeccols (list): column names for RA and Dec
        cols (list): columns to include in the output
        cmradius (float): crossmatch radius in arcsec
        include_single_epoch (bool): include single epoch photometry

    Returns:
        pandas.DataFrame: crossmatched table
    '''
    ra = data[radeccols[0]].values
    dec = data[radeccols[1]].values
    select_cols = "*"
    if cols is not None:
        select_cols=",".join(["tt."+c for c in cols])
    gc3 = pd.DataFrame(
        sqlutilpy.local_join(
            """
                select %s, q3c_dist(m.ra_,m.dec_,tt.ra,tt.dec) as dist from mytable as m
                left join lateral (select * from glimpse.catalog3 as s
                where q3c_join(m.ra_, m.dec_,s.ra,s.dec,%0.5f/3600)  
                order by q3c_dist(m.ra_,m.dec_,s.ra,s.dec) asc limit 1)
                as tt on  true  order by xid """%(select_cols,cmradius),
            'mytable', (ra, dec, np.arange(len(dec))), ('ra_', 'dec_', 'xid'),**wsdb_kwargs))

    gc2 = pd.DataFrame(
        sqlutilpy.local_join(
            """
                select %s, q3c_dist(m.ra_,m.dec_,tt.ra,tt.dec) as dist from mytable as m
                left join lateral (select * from glimpse.catalog2 as s
                where q3c_join(m.ra_, m.dec_,s.ra,s.dec,%0.5f/3600)  
                order by q3c_dist(m.ra_,m.dec_,s.ra,s.dec) asc limit 1)
                as tt on  true  order by xid """%(select_cols,cmradius),
            'mytable', (ra, dec, np.arange(len(dec))), ('ra_', 'dec_', 'xid'),**wsdb_kwargs))
    fltr = (gc3['mag3_6'] != gc3['mag3_6'])
    gc3[fltr] = gc2[fltr]

    if include_single_epoch:

        gc4 = pd.DataFrame(
            sqlutilpy.local_join(
                """
            select %s, q3c_dist(m.ra_,m.dec_,tt.ra,tt.dec) as dist from mytable as m
            left join lateral (select * from jason_sanders.glimpse2_epoch1_catalog as s
            where q3c_join(m.ra_, m.dec_,s.ra,s.dec,%0.5f/3600)  
            order by q3c_dist(m.ra_,m.dec_,s.ra,s.dec) asc limit 1)
            as tt on  true  order by xid """%(select_cols,cmradius),
                'mytable', (ra, dec, np.arange(len(dec))), ('ra_', 'dec_', 'xid'),**wsdb_kwargs))

        for fl in ['3_6','4_5','5_8','8_0']:
            ## Do it band-by-band in case some bands available and others aren't
            fltr = (gc3['mag'+fl] != gc3['mag'+fl])&(gc4['mag'+fl] == gc4['mag'+fl])
            for c in ['mag%s','d%sm','f%s','df%s','rms_f%s','sky_%s','sn_%s','dens_%s','m%s','n%s','sqf_%s']:
                gc3.loc[fltr,c%fl] = gc4[c%fl].values[fltr]

    return gc3

def cm_ramirez(data, radeccols=['ra', 'de'], radius=1.):
    '''
    Crossmatch a table of sources with the Ramirez+2008 catalogue.
    
    Parameters:
        data (pandas.DataFrame): table of sources
        radeccols (list): column names for RA and Dec
        radius (float): crossmatch radius in arcsec
    
    Returns:
        pandas.DataFrame: crossmatched table
    '''
    ra = data[radeccols[0]].values
    dec = data[radeccols[1]].values
    gnm = pd.DataFrame(
        sqlutil.local_join(
            """
                select *, q3c_dist(m.ra_,m.dec_,tt.ra,tt.dec) as dist from mytable as m
                left join lateral (select * from jason_sanders.spitzer_irac_gc as s
                where q3c_join(m.ra_, m.dec_,s.ra,s.dec,%0.3f/3600)  
                order by q3c_dist(m.ra_,m.dec_,s.ra,s.dec) asc limit 1)
                as tt on  true  order by xid """ % radius,
            'mytable', (ra, dec, np.arange(len(dec))), ('ra_', 'dec_', 'xid'),
            **wsdb_kwargs
        ))
    return gnm


def add_spitzer_phot(tbl,radeccols=['ra', 'dec']):
    '''
    Add Spitzer photometry to a table of sources (using both Glimpse and Ramirez+2008)
    
    Parameters:
        tbl (pandas.DataFrame): table of sources
        radeccols (list): column names for RA and Dec
        
    Returns:
        pandas.DataFrame: table with Spitzer photometry added
    '''
    glimpse = cm_glimpse(tbl, radeccols=radeccols, include_single_epoch=True)
    for c in ['mag3_6', 'mag4_5', 'mag5_8', 'mag8_0']:
        if c[3:]+'_glimpse' in tbl.columns:
            del tbl[c[3:]+'_glimpse']
        tbl[c] = glimpse[c]
        tbl.loc[(tbl[c]>90.)|(tbl[c]==0.),c]=np.nan
        tbl = tbl.rename(columns={c:c[3:]+'_glimpse'})
    for c in ['d3_6m', 'd4_5m', 'd5_8m', 'd8_0m']:
        if 'e'+c[1:-1]+'_glimpse' in tbl.columns:
            del tbl['e'+c[1:-1]+'_glimpse']
        tbl[c] = glimpse[c]
        tbl.loc[(tbl[c]>90.)|(tbl[c]==0.),c]=np.nan
        tbl = tbl.rename(columns={c:'e'+c[1:-1]+'_glimpse'})

    ramirez = cm_ramirez(tbl, radeccols=radeccols)
    for c in ['_3_6mag', '_4_5mag', '_5_8mag', '_8_0mag']:
        tbl[c[1:-3]+'_ramirez'] = ramirez[c]
        tbl['e'+c[1:-3]+'_ramirez'] = ramirez['e'+c]
        tbl.loc[tbl[c[1:-3]+'_ramirez']==0.,'e'+c[1:-3]+'_ramirez']=np.nan
        tbl.loc[tbl[c[1:-3]+'_ramirez']==0.,c[1:-3]+'_ramirez']=np.nan

    for c in ['3_6', '4_5', '5_8', '8_0']:
        tbl[c]=tbl[c+'_ramirez']
        tbl['e'+c]=tbl['e'+c+'_ramirez']
        tbl.loc[tbl[c]!=tbl[c],'e'+c]=tbl['e'+c+'_glimpse'][tbl[c]!=tbl[c]]
        tbl.loc[tbl[c]!=tbl[c],c]=tbl[c+'_glimpse'][tbl[c]!=tbl[c]]

    return tbl

if __name__=="__main__":
    data = load_mira_sample()

