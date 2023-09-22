#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: sormani_potential.py
# Description: functions to generate the Sormani+2022 dynamical models and potential -- some code from Mattia Sormani

import numpy as np
import agama

def sormani_DF(init_potential,
               mass     = 0.097,
               Rdisk    = 0.075,
               Hdisk    = 0.025,
               sigmar0  = 75.0,
               Rsigmar  = 1.0,
               sigmamin = 2.0,
               Jmin     = 10.0):

    # introduce DF for the NSD component
    dfNSD = agama.DistributionFunction(potential=init_potential, type='QuasiIsothermal',
                                       mass=mass, Rdisk=Rdisk, Hdisk=Hdisk, 
                                       sigmar0=sigmar0, Rsigmar=Rsigmar, 
                                       sigmamin=sigmamin, Jmin=Jmin)
    
    return dfNSD

class sormani_DF_cls(object):
    def __init__(self, potential):
        self.potential = potential
    def __call__(self, Rdisk, Hdisk, sigmar0):
        return sormani_DF(self.potential, Rdisk=Rdisk, Hdisk=Hdisk, sigmar0=sigmar0)

def sormani_DF_exp(mass     = 0.097,
                   Rdisk    = 0.075,
                   Hdisk    = 0.025,
                   sigmar0  = 75.0,
                   vO       = 140.0):
    
    # introduce DF for the NSD component
    
    dfNSD = agama.DistributionFunction(type='Exponential',
                                       mass=mass, Jphi0=Rdisk*vO, Jz0=Hdisk*vO, 
                                       Jr0=sigmar0*Rdisk)
    
    return dfNSD

def sormani_DF_plus_background(init_potential,
               mass     = 0.097,
               Rdisk    = 0.075,
               Hdisk    = 0.025,
               sigmar0  = 75.0,
               Rsigmar  = 1.0,
               sigmamin = 2.0,
               Jmin     = 10.0,
               mass_bar = 1.,
               J0 = 20.,
               slopeIn = 1.,
               slopeOut = 4.):
    
    # introduce DF for the NSD component
    dfNSD = sormani_DF(init_potential=init_potential, mass=mass, Rdisk=Rdisk, Hdisk=Hdisk, sigmar0=sigmar0, Rsigmar=Rsigmar)
    
    dfBar = agama.DistributionFunction(type='DoublePowerLaw',
                                       mass=mass_bar, J0=J0, slopeIn=slopeIn, slopeOut=slopeOut)
    
    return agama.DistributionFunction(dfNSD, dfBar)

def sormani_DF_exp_plus_background(mass     = 0.097,
               Rdisk    = 0.075,
               Hdisk    = 0.025,
               sigmar0  = 75.0,
               Rsigmar  = 1.0,
               sigmamin = 2.0,
               Jmin     = 10.0,
               mass_bar = 1.,
               J0 = 20.,
               slopeIn = 1.,
               slopeOut = 4.):
    
    # introduce DF for the NSD component
    dfNSD = sormani_DF_exp(mass=mass, Rdisk=Rdisk, Hdisk=Hdisk, sigmar0=sigmar0)
    
    dfBar = agama.DistributionFunction(type='DoublePowerLaw',
                                       mass=mass_bar, J0=J0, slopeIn=slopeIn, slopeOut=slopeOut)
    
    return agama.DistributionFunction(dfNSD, dfBar)


class binney_new_df(object):
    def __init__(self,mass=0.097, Rdisk=0.075, Hdisk=0.025, sigmar0=75., vO=140.):
        self.mass, self.Rdisk, self.Hdisk, self.sigmar0, self.vO = mass, Rdisk, Hdisk, sigmar0, vO
        self.Jphi0=self.Rdisk*self.vO
        self.Jz0=self.Hdisk*self.vO
        self.Jr0=self.sigmar0*self.Rdisk
        self.Jd0, self.Jv0 = 0., 0.
    def __call__(self, x):
        if x.ndim==1:
            if x[2]<0.:
                return 0.
        X = np.atleast_2d(x)
        Jv = X[:,2]+self.Jv0
        Jd = X[:,2]+self.Jd0
        pR = 1.
        pz = 1.
        sclR = np.power(Jv/self.Jphi0,pR)/self.Jr0
        sclz = np.power(Jv/self.Jphi0,pz)/self.Jz0
        fTotal = self.mass/(2.*np.pi)**3 * X[:,2]/self.Jphi0**2 *sclR*sclz*np.exp(-sclR*X[:,0]-sclz*X[:,1]-Jd/self.Jphi0)
        if x.ndim==1:
            return fTotal[0]
        fTotal[X[:,2]<0.]=0.
        return fTotal

def generate_model(save_output=False):
    
    agama.setUnits(length=1, velocity=1, mass=1e10)   # 1 kpc, 1 km/s, 1e10 Msun

    # initialize the SelfConsistentModel object (only the potential expansion parameters)
    model = agama.SelfConsistentModel(
        RminCyl        = 0.005,
        RmaxCyl        = 1.0,
        sizeRadialCyl  = 25,
        zminCyl        = 0.005,
        zmaxCyl        = 1.0,
        sizeVerticalCyl= 25)

    # construct a two component model: NSD + NSC
    # NSD -> generated self-consistently
    # NSC -> kept fixed as an external potential

    # NSC best-fitting model from Chatzopoulos et al. 2015 (see Equation 28 here: https://arxiv.org/pdf/2007.06577.pdf)
    density_NSC_init = agama.Density(type='Dehnen',mass=6.1e-3,gamma=0.71,scaleRadius=5.9e-3,axisRatioZ=0.73)

    # NSD model 3 from Sormani et al. 2020 (see Equation 24 here: https://arxiv.org/pdf/2007.06577.pdf)
    d1 = agama.Density(type='Spheroid',DensityNorm=0.9*222.885,gamma=0,beta=0,axisRatioZ=0.37,outerCutoffRadius=0.0050617,cutoffStrength=0.7194)
    d2 = agama.Density(type='Spheroid',DensityNorm=0.9*169.975,gamma=0,beta=0,axisRatioZ=0.37,outerCutoffRadius=0.0246,cutoffStrength=0.7933)
    density_NSD_init = agama.Density(d1,d2)

    # add both NSC and NSD components as static density profiles for the moment:
    # assign both of them to a single CylSpline potential solver by setting disklike=True
    # and thus avoid creating an additional Multipole potential, which is typically used
    # for spheroidal components - this reduces computational cost by roughly a half
    model.components.append(agama.Component(density=density_NSC_init, disklike=True))
    model.components.append(agama.Component(density=density_NSD_init, disklike=True))

    # compute the initial guess for the potential
    model.iterate()
    # plotVcirc(model, 0)

    if save_output:
        model.potential.export('sormani_2021_initial_potential.agama')
    
    # replace the static density of the NSD by a DF-based component
    DF = sormani_DF(model.potential)
    model.components[1] = agama.Component(df=DF, disklike=True,
        RminCyl=0.005, RmaxCyl=0.75, sizeRadialCyl=20, zminCyl=0.005, zmaxCyl=0.25, sizeVerticalCyl=15)

    # iterate to make NSD DF & potential self-consistent
    for iteration in range(1,6):
        print('Starting iteration #%d' % iteration)
        model.iterate()

    if save_output:
        model.potential.export('sormani_2021_potential.agama')
    
    return model.potential, DF


def generate_bar_potential():
    #!/usr/bin/python
    '''
    This script defines an analytic approximation for the barred Milky Way model from Portail et al.(2017)
    and constructs a corresponding CylSpline potential, which can be used to integrate orbits, etc.
    The density is represented by four components: an X-shaped inner bar, two instances of long bars,
    and an axisymmetric disk. In addition, there is a 'central mass concentration' (a triaxial disk)
    and a flattened axisymmetric dark halo, which is represented by a separate Multipole potential.
    This potential model is a good fit for the central region of the Galaxy (within ~5kpc),
    but is not very realistic further out.
    The left panel shows the circular-velocity curve (in the axisymmetrized potential),
    and the right panel shows examples of a few orbits in this potential.

    Reference: Sormani et al.(submitted)

    Authors: Mattia Sormani, Eugene Vasiliev
    '''

    # Nearly identical to the built-in Disk density profile, but with a slightly different
    # vertical profile containing an additional parameter 'verticalSersicIndex'
    def makeDisk(**params):
        surfaceDensity      = params['surfaceDensity']
        scaleRadius         = params['scaleRadius']
        scaleHeight         = params['scaleHeight']
        innerCutoffRadius   = params['innerCutoffRadius']
        sersicIndex         = params['sersicIndex']
        verticalSersicIndex = params['verticalSersicIndex']
        def density(xyz):
            R = (xyz[:,0]**2 + xyz[:,1]**2)**0.5
            return (surfaceDensity / (4*scaleHeight) *
                np.exp( - (R/scaleRadius)**sersicIndex - innerCutoffRadius/(R+1e-100)) /
                np.cosh( (abs(xyz[:,2]) / scaleHeight)**verticalSersicIndex ) )
        return agama.Density(density)

    # Modification of equation 9 of Coleman et al. 2020 (https://arxiv.org/abs/1911.04714)
    def makeXBar(**params):
        densityNorm = params['densityNorm']
        x0   = params['x0']
        y0   = params['y0']
        z0   = params['z0']
        xc   = params['xc']
        yc   = params['yc']
        c    = params['c']
        alpha= params['alpha']
        cpar = params['cpar']
        cperp= params['cperp']
        m    = params['m']
        n    = params['n']
        outerCutoffRadius = params['outerCutoffRadius']
        def density(xyz):
            r  = np.sum(xyz**2, axis=1)**0.5
            a  = ( ( (abs(xyz[:,0]) / x0)**cperp + (abs(xyz[:,1]) / y0)**cperp )**(cpar/cperp) +
                (abs(xyz[:,2]) / z0)**cpar )**(1/cpar)
            ap = ( ((xyz[:,0] + c * xyz[:,2]) / xc)**2 + (xyz[:,1] / yc)**2 )**(0.5)
            am = ( ((xyz[:,0] - c * xyz[:,2]) / xc)**2 + (xyz[:,1] / yc)**2 )**(0.5)
            return (densityNorm / np.cosh(a**m) * np.exp( -(r/outerCutoffRadius)**2) *
                (1 + alpha * (np.exp(-ap**n) + np.exp(-am**n) ) ) )
        return agama.Density(density)

    # Modification of equation 9 of Wegg et al. 2015 (https://arxiv.org/pdf/1504.01401.pdf)
    def makeLongBar(**params):
        densityNorm = params['densityNorm']
        x0   = params['x0']
        y0   = params['y0']
        cpar = params['cpar']
        cperp= params['cperp']
        scaleHeight = params['scaleHeight']
        innerCutoffRadius   = params['innerCutoffRadius']
        outerCutoffRadius   = params['outerCutoffRadius']
        innerCutoffStrength = params['innerCutoffStrength']
        outerCutoffStrength = params['outerCutoffStrength']
        def density(xyz):
            R = (xyz[:,0]**2 + xyz[:,1]**2)**0.5
            a = ( (abs(xyz[:,0]) / x0)**cperp + (abs(xyz[:,1]) / y0)**cperp )**(1/cperp)
            return densityNorm / np.cosh(xyz[:,2] / scaleHeight)**2 * np.exp(-a**cpar
                -(R/outerCutoffRadius)**outerCutoffStrength - (innerCutoffRadius/R)**innerCutoffStrength)
        return agama.Density(density)

    # additional central mass concentration as described in sec.7.3 of Portail et al.(2017)
    def makeCMC(mass, scaleRadius, scaleHeight, axisRatioY):
        norm = mass / (4 * np.pi * scaleRadius**2 * scaleHeight * axisRatioY)
        return agama.Density(lambda xyz:
            norm * np.exp(-(xyz[:,0]**2 + (xyz[:,1]/axisRatioY)**2)**0.5 / scaleRadius
                - abs(xyz[:,2]) / scaleHeight) )

    # create the total density profile with 4 component from the provided array of unnamed parameters
    def makeDensityModel(params):
        ind=0
        densityDisk = makeDisk(
            surfaceDensity=params[ind+0],
            scaleRadius=params[ind+1],
            innerCutoffRadius=params[ind+2],
            scaleHeight=params[ind+3],
            sersicIndex=params[ind+4],
            verticalSersicIndex=params[ind+5])
        ind+=6
        densityXBar = makeXBar(
            densityNorm=params[ind+0],
            x0=params[ind+1],
            y0=params[ind+2],
            z0=params[ind+3],
            cpar=params[ind+4],
            cperp=params[ind+5],
            m=params[ind+6],
            outerCutoffRadius=params[ind+7],
            alpha=params[ind+8],
            c=params[ind+9],
            n=params[ind+10],
            xc=params[ind+11],
            yc=params[ind+12])
        ind+=13
        densityLongBar1 = makeLongBar(
            densityNorm=params[ind+0],
            x0=params[ind+1],
            y0=params[ind+2],
            scaleHeight=params[ind+3],
            cperp=params[ind+4],
            cpar=params[ind+5],
            outerCutoffRadius=params[ind+6],
            innerCutoffRadius=params[ind+7],
            outerCutoffStrength=params[ind+8],
            innerCutoffStrength=params[ind+9] )
        ind+=10
        densityLongBar2 = makeLongBar(
            densityNorm=params[ind+0],
            x0=params[ind+1],
            y0=params[ind+2],
            scaleHeight=params[ind+3],
            cperp=params[ind+4],
            cpar=params[ind+5],
            outerCutoffRadius=params[ind+6],
            innerCutoffRadius=params[ind+7],
            outerCutoffStrength=params[ind+8],
            innerCutoffStrength=params[ind+9] )
        ind+=10
        assert len(params)==ind, 'invalid number of parameters'
        return agama.Density(densityDisk, densityXBar, densityLongBar1, densityLongBar2)


    # create the potential of the entire model:
    # 4-component stellar density as defined above, plus central mass concentration, plus dark halo
    def makePotentialModel(params, useCMC=True):
        # combined 4 components and the CMC represented by a single triaxial CylSpline potential
        mmax = 12  # order of azimuthal Fourier expansion (higher order means better accuracy,
        # but values greater than 12 *significantly* slow down the computation!)
        pot_bary = agama.Potential(type='CylSpline',
            density=agama.Density(makeDensityModel(params), makeCMC(useCMC*0.2e10, 0.25, 0.05, 0.5)),
            symmetry='t', mmax=mmax, gridsizeR=25, gridsizez=25, Rmin=0.1, Rmax=40, zmin=0.05, zmax=20)
        # flattened axisymmetric dark halo with the Einasto profile
        pot_dark = agama.Potential(type='Multipole',
            density='Spheroid', axisratioz=0.8, gamma=0, beta=0,
            outerCutoffRadius=1.84, cutoffStrength=0.74, densityNorm=0.0263e10,
            gridsizer=26, rmin=0.01, rmax=1000, lmax=8)
        return agama.Potential(pot_bary, pot_dark)

    params = np.array(
    # disk
    [ 1.03063359e+09, 4.75409497e+00, 4.68804907e+00, 1.51100601e-01,
        1.53608780e+00, 7.15915848e-01 ] +
    # short/thick bar
    [ 3.16273226e+09, 4.90209137e-01, 3.92017253e-01, 2.29482096e-01,
        1.99110223e+00, 2.23179266e+00, 8.73227940e-01, 4.36983774e+00,
        6.25670015e-01, 1.34152138e+00, 1.94025114e+00, 7.50504078e-01,
        4.68875471e-01] +
    # long bar 1
    [ 4.95381575e+08, 5.36363324e+00, 9.58522229e-01, 6.10542494e-01,
        9.69645220e-01, 3.05125124e+00, 3.19043585e+00, 5.58255674e-01,
        1.67310332e+01, 3.19575493e+00] +
    # long bar 2
    [ 1.74304936e+13, 4.77961423e-01, 2.66853061e-01, 2.51516920e-01,
        1.87882599e+00, 9.80136710e-01, 2.20415408e+00, 7.60708626e+00,
    -2.72907665e+01, 1.62966434e+00]
    )

    agama.setUnits(length=1, mass=1, velocity=1)  # 1 kpc, 1 Msun, 1 km/s
    den = makeDensityModel(params)
    pot = makePotentialModel(params)
    pot_noCMC = makePotentialModel(params, useCMC=False)
    pot.export('portail_2017.agama')
    pot_noCMC.export('portail_2017_noCNC.agama')
    print('Created MW potential: total mass in stars=%.3g Msun, halo=%.3g Msun' %
        (pot[0].totalMass(), pot[1].totalMass()))
    # create an axisymmetrized version of the potential for plotting the true circular-velocity curve
    pot_axi = agama.Potential(
        agama.Potential(type='CylSpline', potential=pot[0],
            mmax=0, gridsizeR=25, gridsizez=25, Rmin=0.1, Rmax=40, zmin=0.05, zmax=20),
        pot[1])
    pot_axi_noCMC = agama.Potential(
        agama.Potential(type='CylSpline', potential=pot_noCMC[0],
            mmax=0, gridsizeR=25, gridsizez=25, Rmin=0.1, Rmax=40, zmin=0.05, zmax=20),
        pot[1])
    
    pot_axi_noCMC.export('portail_2017_noCNC_axi.agama')
    pot_axi.export('portail_2017_axi.agama')

    return pot_axi, pot_axi_noCMC


if __name__=="__main__":
    generate_model(True)
    generate_bar_potential()