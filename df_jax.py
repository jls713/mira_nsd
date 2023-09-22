#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: df_jax.py
# Description: DFs for NSD

import jax.numpy as jnp
from jax_cosmo.jax_utils import container
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from jax.scipy.special import erfc
from jax import jit
from functools import partial

'''binney DF for NSD'''
class binney_df_jax(container):
    '''binney DF for NSD parameterized by P: units are in kpc, km/s, Msun'''
    def __init__(self, Rdisk, Hdisk, deltaR, 
                 mass=1., vO=140., Jv0=0., Jd0=0., **kwargs):
        super(binney_df_jax_spline, self).__init__(Rdisk, Hdisk, deltaR,
                                                   mass=mass, vO=vO, Jv0=Jv0, Jd0=Jd0, **kwargs)

    def __call__(self, x):
        '''binney DF for NSD
        
        Args:
            x (jnp.array): 3D action coordinates (Jr, Jz, Jphi)

        Returns:
            jnp.array: DF value
        '''

        Rdisk, Hdisk, deltaR = self.params

        Jphi0 = Rdisk*self.config['vO']
        Jz0 =  Hdisk*self.config['vO']
        Jr0 = deltaR*self.config['vO']

        X = jnp.atleast_2d(x)
        
        # define auxiliary variables
        Jv = X[:,2]+self.config['Jv0']
        Jd = X[:,2]+self.config['Jd0']

        # define scaling factors
        pR = 1.
        pz = 1.

        sclR = jnp.power(Jv / Jphi0, pR) / Jr0
        sclz = jnp.power(Jv / Jphi0, pz) / Jz0

        fTotal = self.config['mass']/(2.*jnp.pi)**3 * X[:,2]/Jphi0**2*sclR*sclz*jnp.exp(-sclR*X[:,0]-sclz*X[:,1]-Jd/Jphi0)
        
        # add tapering for negative angular momentum
        fTotal=jnp.where(X[:,2]<0., fTotal*jnp.exp(Jv * X[:,2] / Jr0**2), fTotal)
        
        if x.ndim==1:
            fTotal = fTotal[0]

        return fTotal

'''binney DF for NSD using jax.cosmo.interpolate spline interpolation for Rdisk, Hdisk and sigmar0 
as a function of auxiliary variable P'''
class binney_df_jax_spline(container):
    '''binney DF for NSD parameterized by P: units are in kpc, km/s, Msun'''
    def __init__(self, ln_Rdisk_coeffs, ln_Hdisk_coeffs, ln_deltaR_coeffs, 
                 aux_knots = jnp.linspace(2.,3.,50),
                 mass=1., vO=140., ln_Jv0=None, ln_Jd0=None, **kwargs):
        super(binney_df_jax_spline, self).__init__(ln_Rdisk_coeffs, ln_Hdisk_coeffs, ln_deltaR_coeffs,
                                                   ln_Jv0, ln_Jd0,  
                                                   aux_knots,
                                                   vO=vO, mass=mass,
                                                   **kwargs)

    def __call__(self, x, aux, log=True, normalization=False):
        '''binney DF for NSD
        
        Args:
            x (jnp.array): 3D action coordinates (Jr, Jz, Jphi)
            aux (jnp.array): auxiliary variable P
            log (bool): if True, return log of DF
            normalization (bool): if True, return normalization factor

        Returns:
            jnp.array: DF value
        '''

        ln_Rdisk_coeffs, ln_Hdisk_coeffs, ln_deltaR_coeffs, ln_Jv0, ln_Jd0, aux_knots =  self.params

        Jp0_int = InterpolatedUnivariateSpline(aux_knots, ln_Rdisk_coeffs, k=3)
        Jz0_int = InterpolatedUnivariateSpline(aux_knots, ln_Hdisk_coeffs, k=3)
        Jr0_int = InterpolatedUnivariateSpline(aux_knots, ln_deltaR_coeffs, k=3)

        X = jnp.atleast_2d(x)

        # define scaling factors
        pR, pz = 1., 1.
        Jr0 = jnp.exp(Jr0_int(aux))*self.config['vO']
        Jz0 = jnp.exp(Jz0_int(aux))*self.config['vO']
        Jp0 = jnp.exp(Jp0_int(aux))*self.config['vO']

        # define auxiliary variables
        if ln_Jv0 is not None:
            Jv0_int = InterpolatedUnivariateSpline(aux_knots, ln_Jv0, k=3)
            Jv0 = jnp.exp(Jv0_int(aux))*self.config['vO']
        else:
            Jv0 = 0.
        if ln_Jd0 is not None:
            Jd0_int = InterpolatedUnivariateSpline(aux_knots, ln_Jd0, k=3)
            Jd0 = jnp.exp(Jd0_int(aux))*self.config['vO']
        else:
            Jd0 = 0.

        Jv = jnp.abs(X[...,2])+Jv0
        Jd = jnp.abs(X[...,2])+Jd0

        if pR!=1.:
            sclR = jnp.power(Jv/Jp0,pR)/Jr0
        else:
            sclR = Jv/(Jp0*Jr0)
        if pz!=1.:
            sclz = jnp.power(Jv/Jp0,pz)/Jz0
        else:
            sclz = Jv/(Jp0*Jz0)

        if normalization:
            norm = (2*Jp0*(2*Jp0*(Jd0+Jp0)+Jr0**2)
                    -jnp.exp((Jr0**2+Jp0*Jv0)**2/(4*Jp0**2*Jr0**2))
                    *Jr0*(-2*Jd0*Jp0+Jr0**2+Jp0*Jv0)
                    *jnp.sqrt(jnp.pi)*erfc((Jr0**2+Jp0*Jv0)/(2*Jp0*Jr0)))/\
                        (4*Jp0**3)*jnp.exp(-Jd0/Jp0)
        else:
            norm = 1.

        if log:
            fTotal = jnp.log(norm)+jnp.log(self.config['mass']/(2.*jnp.pi)**3 * Jd/Jp0**2 *sclR*sclz)\
                        -sclR*X[...,0]-sclz*X[...,1]-Jd/Jp0
            fTotal=jnp.where(X[...,2]<0., fTotal+Jv * X[...,2] / Jr0**2, fTotal)
        else:
            fTotal = norm*self.config['mass']/(2.*jnp.pi)**3 * Jd/Jp0**2*sclR*sclz*jnp.exp(
                        -sclR*X[...,0]-sclz*X[...,1]-Jd/Jp0)
            # add tapering for negative angular momentum
            fTotal=jnp.where(X[...,2]<0., fTotal*jnp.exp(Jv * X[...,2] / Jr0**2), fTotal)
        
        if x.ndim==1:
            fTotal = fTotal[0]

        return fTotal
    

'''quasi-isothermal DF with spline interpolation for Rdisk, Hdisk, sigmar0 and Rsigmar as a function of auxiliary variable P'''
class quasiisothermal_df_jax_spline(container):
    '''quasi-isothermal DF parameterized by P: units are in kpc, km/s, Msun'''
    
    def __init__(self, ln_Rdisk, ln_Hdisk, ln_sigmar0, ln_Rsigmar,
                 aux_knots = jnp.linspace(2.,3.,50), sigmamin=2.,
                 mass=1., **kwargs):
        super(quasiisothermal_df_jax_spline, self).__init__(ln_Rdisk, ln_Hdisk,
                                                            ln_sigmar0, ln_Rsigmar,
                                                            aux_knots,
                                                            sigmamin=sigmamin,
                                                            mass=mass,
                                                            **kwargs)

    @partial(jit, static_argnums=(0,3,4))
    def __call__(self, x, aux, log=True, normalization=False):
        '''quasi-isothermal DF
        
        Args:
            x (jnp.array): 3D action + 4 frequency coordinates (Jr, Jz, Jphi, 
            kappa, nu, Omega, Rc)
            aux (jnp.array): auxiliary variable P
            log (bool): if True, return log of DF
            normalization (bool): if True, return normalization factor

        Returns:
            jnp.array: DF value
        '''

        ln_Rdisk, ln_Hdisk, ln_sigmar0, ln_Rsigmar, aux_knots =  self.params

        X = jnp.atleast_2d(x)

        # Check if Rdisk is a scalar
        if ln_Rdisk.ndim==0:
            Rd = jnp.exp(ln_Rdisk)
            Hz = jnp.exp(ln_Hdisk)
            s0 = jnp.exp(ln_sigmar0)
            Rs = jnp.exp(ln_Rsigmar)
        else:
            lnRd_int = InterpolatedUnivariateSpline(aux_knots, ln_Rdisk, k=3)
            lnHz_int = InterpolatedUnivariateSpline(aux_knots, ln_Hdisk, k=3)
            lns0_int = InterpolatedUnivariateSpline(aux_knots, ln_sigmar0, k=3)
            lnRs_int = InterpolatedUnivariateSpline(aux_knots, ln_Rsigmar, k=3)
            Rd = jnp.exp(lnRd_int(aux))
            Hz = jnp.exp(lnHz_int(aux))
            s0 = jnp.exp(lns0_int(aux))
            Rs = jnp.exp(lnRs_int(aux))

        sig2R = s0**2*jnp.exp(-2*(X[...,-1]-Rd)/Rs) + self.config['sigmamin']**2
        sig2z = 2*Hz**2*X[...,4]**2 + self.config['sigmamin']**2
        sclR = X[...,3]/sig2R
        sclz = X[...,4]/sig2z

        fTotal = jnp.log(self.config['mass']/(4.*jnp.pi**3) * X[...,5] * sclR * sclz / (X[...,3]*Rd) **2)+\
                    -sclR*X[...,0]-sclz*X[...,1]-X[...,-1]/Rd
        fTotal=jnp.where(X[...,2]<0., fTotal+2*X[...,5]*X[...,2]/sig2R, fTotal)

        if not log:
            fTotal = jnp.exp(fTotal)

        if x.ndim==1:
            fTotal = fTotal[0]

        return fTotal
        