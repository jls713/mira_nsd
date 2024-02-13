from astropy.table import Table
import astropy.units as u
import pandas as pd
import cdspyreadme

t = Table.read('https://www.homepages.ucl.ac.uk/~ucapjls/data/mira_vvv.fits')
t2 = Table.from_pandas(pd.read_csv('final_nsd_mira_full_cols_updated_proper_motions.csv'))

t2.rename_column('uwe','astfit_uwe')
t2.rename_column('chisq', 'astfit_chisq')

t.meta['comments']="""=====================================================================
Mira variables in the Nuclear Stellar Disc
=====================================================================

A catalogue of Mira variable candidates extracted from the VIRAC2
reduction of the VVV survey (see Sanders et al. 2022 for details).

VIRAC2 provides photometry and astrometry (proper motions) as used in
the analysis of Sanders et al. (2024). For consistency with the 
corresponding publications, the provided photometry corresponds to an 
intermediate version of VIRAC2 that was used to discover the Mira 
variable candidates whilst the astrometry is from the version of 
VIRAC2 that will be public. 
=====================================================================
""".split('\n')

final_column_units = {'ra':u.deg, 'dec':u.deg, 'ra_error':u.mas, 'dec_error':u.mas,
                      'pmra':u.mas/u.yr, 
                      'pmdec':u.mas/u.yr, 
                      'pmra_error':u.mas/u.yr, 
                      'pmdec_error':u.mas/u.yr, 
                      'pmra_pmdec_corr':u.dimensionless_unscaled,
                      'parallax':u.mas,
                      'parallax_error':u.mas,
                      'parallax_pmra_corr':u.dimensionless_unscaled,
                      'parallax_pmdec_corr':u.dimensionless_unscaled,
                      'astfit_chisq':u.dimensionless_unscaled,
                      'astfit_uwe':u.dimensionless_unscaled}

final_column_descr={
        'ra':'VIRAC-2 right ascension at 2014.0 reference epoch (deg)',
        'dec':'VIRAC-2 declination at 2014.0 reference epoch (deg)',
        'ra_error':'VIRAC-2 right ascension * cos(dec) uncertainty (mas)',
        'dec_error':'VIRAC-2 declination uncertainty (mas)',
        'parallax':'VIRAC-2 parallax (mas)',
        'parallax_error':'VIRAC-2 parallax uncertainty (mas)',
        'pmra':'VIRAC-2 right ascension proper motion with cos(dec) factor ($\mu_\alpha*=\mu_\alpha\cos\delta$) (mas/yr)',
        'pmra_error':'Uncertainty in VIRAC-2 right ascension proper motion with cos (dec) factor (mas/yr)',
        'pmdec':'VIRAC-2 declination proper motion (mas/yr)',
        'pmdec_error':'Uncertainty in VIRAC-2 decliation proper motion (mas/yr)',
        'parallax_pmra_corr':'VIRAC-2 correlation coefficient between parallax and pmra',
        'parallax_pmdec_corr':'VIRAC-2 correlation coefficient between parallax and pmdec',
        'pmra_pmdec_corr':'VIRAC-2 correlation coefficient between pmra and pmdec',
        'astfit_chisq':'VIRAC-2 astrometric fit chi-squared',
        'astfit_uwe':'VIRAC-2 astrometric fit unit weight error (chisq per degree of freedom)'}

fields_to_replace = ['ra','dec','ra_error','dec_error','pmra','pmdec','pmra_error','pmdec_error','pmra_pmdec_corr',
                     'parallax','parallax_error','parallax_pmra_corr','parallax_pmdec_corr',
                     'astfit_chisq','astfit_uwe']
for f in fields_to_replace:
    t[f]=t2[f]*final_column_units[f]
    t[f].description=final_column_descr[f]
t['Ks_nepochs'].description='Number of VIRAC-2 Ks epochs used in the light curve processing for classification as a Mira variable'
t['unreliable'].description='Binary flag if Mira variable classification is unreliable (=1) based on a visual classification of the light curve'
t.write('mira_vizier_table.fits', overwrite=True)

tablemaker = cdspyreadme.CDSTablesMaker()
tablemaker.catalogue="Mira variable candidates in the Milky Way NSD region"
tablemaker.title = "Bar formation epoch from NSD Mira variables"
tablemaker.abstract = """
A key event in the history of the Milky Way is the formation of the bar. This event affects the subsequent structural and dynamical evolution of the entire Galaxy. When the bar formed, gas was likely rapidly funnelled to the centre of the Galaxy settling in a star-forming nuclear disc. The Milky Way bar formation can then be dated by considering the age distribution of the oldest stars in the formed nuclear stellar disc. In this highly obscured and crowded region, reliable age tracers are limited, but bright, high-amplitude Mira variables make useful age indicators as they follow a period--age relation.
We fit dynamical models to the proper motions of a sample of Mira variables in the Milky Way's nuclear stellar disc region. Weak evidence for inside-out growth and both radial and vertical dynamical heating with time of the nuclear stellar disc is presented suggesting the nuclear stellar disc is dynamically well-mixed. Furthermore, for Mira variables around a $\sim350$ day period, there is a clear transition from nuclear stellar disc-dominated kinematics to background bar-bulge-dominated kinematics.
Using a Mira variable period--age relation calibrated in the solar neighbourhood, this suggests the nuclear stellar disc formed in a significant burst in star formation $(8\pm 1)\,\mathrm{Gyr}$ ago, although the data are also weakly consistent with a more gradual formation of the nuclear stellar disc at even earlier epochs.
This implies a relatively early formation time for the Milky Way bar ($\gtrsim8\,\mathrm{Gyr}$), which has implications for the growth and state of the young Milky Way and its subsequent history.
"""
tablemaker.more_description="""A catalogue of Mira variable candidates extracted from the VIRAC2
reduction of the VVV survey (see Sanders et al. 2022 for details).

VIRAC2 provides photometry and astrometry (proper motions) as used in
the analysis of Sanders et al. (2024). For consistency with the 
corresponding publications, the provided photometry corresponds to an 
intermediate version of VIRAC2 that was used to discover the Mira 
variable candidates whilst the astrometry is from the version of 
VIRAC2 that will be public. """
tablemaker.keywords = """Galaxy: evolution -- Galaxy: formation --  Galaxy: nucleus --  Galaxy: kinematics and dynamics -- stars: variables: general --  stars: AGB"""
tablemaker.date = 2024
tablemaker.author="Jason L. Sanders"
tablemaker.authors="Daisuke Kawata, Noriyuki Matsunaga, Mattia C. Sormani, Leigh C. Smith, Dante Minniti$ and Ortwin Gerhard"
tablemaker.addTable(t, name='mira_vizier_table_output')
tablemaker.writeCDSTables()
with open("vizier_ReadMe", "w") as fd:
    tablemaker.makeReadMe(out=fd)
