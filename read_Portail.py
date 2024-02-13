# Code to read Portail M2M bar model
# From Mattia Sormani

import numpy as np

def rotate(x,y,theta):
  xprime = x*np.cos(theta) - y*np.sin(theta)
  yprime = x*np.sin(theta) + y*np.cos(theta)
  return xprime, yprime

###############
# class to manage Portail bar
################

class Portail(object): 
    """
    Portail M2M bar model
    """
    def __init__(self): 

        ###############
        # read snapshot
        ################
        print("reading Portail bar...\r")

        # units
        kpc2iu = 0.793300000000000
        iu2kms = 360.245600000000      
        iu2SM  = 38047664072.3735      
        
        self.x,self.y,self.z,self.vx,self.vy,self.vz,self.m,self.flag = np.genfromtxt(
            '/home/jls/work/nsd/Gerhard/data/M85MW_37.5_0060000').T 

        self.x = self.x/kpc2iu
        self.y = self.y/kpc2iu
        self.z = self.z/kpc2iu

        self.vx = self.vx*(iu2kms/100.)
        self.vy = self.vy*(iu2kms/100.)
        self.vz = self.vz*(iu2kms/100.)

        self.m   = self.m*(iu2SM/1e10) # convert mass to units of 1e10Mun

        self.Cdm    = (np.abs(self.flag-0)<0.1)
        self.Cstars = (np.abs(self.flag-1)<0.1)
        self.Cnsd   = (np.abs(self.flag-2)<0.1)

        self.Omegap = 37.5

        print("finished")

    ##############################
    # cut useless stuff
    ############################## 

    def cut_stuff(self, cut_radius=15.):
        """
        removes particles outside of cut_radius kpc
        """
        x_cm = (self.x*self.m).sum()/self.m.sum()
        y_cm = (self.y*self.m).sum()/self.m.sum()
        z_cm = (self.z*self.m).sum()/self.m.sum()

        COND = (np.sqrt((self.x-x_cm)**2 + (self.y-y_cm)**2 + (self.z-z_cm)**2)<cut_radius) & (self.Cstars)

        self.m    = self.m[COND]  
        self.x    = self.x[COND]  
        self.y    = self.y[COND]  
        self.z    = self.z[COND]  
        self.vx   = self.vx[COND] 
        self.vy   = self.vy[COND] 
        self.vz   = self.vz[COND]
        self.flag = self.flag[COND]

        self.Cdm    = self.Cdm[COND]
        self.Cstars = self.Cstars[COND]
        self.Cnsd   = self.Cnsd[COND]

    ##############################
    # increase number of points 8x by symmetrising
    ############################## 

    def symmetrise(self):
        """
        symmetrises the bar by reflecting it in the principal planes
        """
        # reflect x
        self.x = np.hstack((self.x,-self.x))
        self.y = np.hstack((self.y,self.y))
        self.z = np.hstack((self.z,self.z))

        self.vx = np.hstack((+self.vx,self.vx))
        self.vy = np.hstack((+self.vy,-self.vy))
        self.vz = np.hstack((+self.vz,self.vz))

        self.m      = np.hstack((self.m,self.m))
        self.flag   = np.hstack((self.flag,self.flag))
        self.Cdm    = np.hstack((self.Cdm,self.Cdm))
        self.Cstars = np.hstack((self.Cstars,self.Cstars))
        self.Cnsd   = np.hstack((self.Cnsd,self.Cnsd))

        # reflect y
        self.x = np.hstack((self.x,self.x))
        self.y = np.hstack((self.y,-self.y))
        self.z = np.hstack((self.z,self.z))

        self.vx = np.hstack((+self.vx,-self.vx))
        self.vy = np.hstack((+self.vy,self.vy))
        self.vz = np.hstack((+self.vz,self.vz))

        self.m      = np.hstack((self.m,self.m))
        self.flag   = np.hstack((self.flag,self.flag))
        self.Cdm    = np.hstack((self.Cdm,self.Cdm))
        self.Cstars = np.hstack((self.Cstars,self.Cstars))
        self.Cnsd   = np.hstack((self.Cnsd,self.Cnsd))

        # # reflect z
        self.x = np.hstack((self.x,self.x))
        self.y = np.hstack((self.y,self.y))
        self.z = np.hstack((self.z,-self.z))

        self.vx = np.hstack((+self.vx,self.vx))
        self.vy = np.hstack((+self.vy,self.vy))
        self.vz = np.hstack((+self.vz,-self.vz))

        self.m      = np.hstack((self.m,self.m)) / 8.
        self.flag   = np.hstack((self.flag,self.flag))
        self.Cdm    = np.hstack((self.Cdm,self.Cdm))
        self.Cstars = np.hstack((self.Cstars,self.Cstars))
        self.Cnsd   = np.hstack((self.Cnsd,self.Cnsd))

    ###############
    # realign 
    ################

    def realign(self):
        """
        Rotate the bar to align with chosen angle
        """
        phibar = 25.0 # HERE IS THE ANGLE TO CHANGE BAR ORIENTATION
        theta_rot = np.radians(90-phibar) 
        self.x,self.y   = rotate(self.x,self.y,theta_rot)
        self.vx,self.vy = rotate(self.vx,self.vy,theta_rot)