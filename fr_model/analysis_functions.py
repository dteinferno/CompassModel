#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:49:37 2019

@author: kris
"""

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from fr_class import frmodel
import pickle
from scipy.optimize import curve_fit

def f(x, a):
    return a*x

def testvels(vels = np.arange(0,2.55, 0.025), time = 'late'):
    if time == 'early':
        storedat = 20
        tstop = 2500
        endpoint = 50
        startpoint=0
    else:
        storedat = 100
        tstop = 12000
        endpoint = 40
        startpoint=6
        endpoint = 28
    meanvels = []
    for vel in vels:
        cx = frmodel()
        cx.add_event([2000, 'vel', vel])
        cx.run_model(tstop = tstop, dt = 0.1, plot=['dirs'], storedat=storedat)
        meanvel = np.mean(cx.vels['EPG'][int(2100*10/storedat):])
        meanvels.append(meanvel)
    meanvels = np.array(meanvels)
    
    velsfit = vels[startpoint:endpoint]
    
    plt.plot(vels, meanvels, 'k-')
    if time == 'early':
        slope, pcov = curve_fit(f, velsfit, meanvels[:endpoint])
        s = slope[0]
        residuals = meanvels[:endpoint]- s*velsfit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((meanvels[:endpoint]-np.mean(meanvels[:endpoint]))**2)
        r2 = 1 - (ss_res / ss_tot)
        plt.plot(velsfit, s*velsfit, 'b--')
        plt.legend(['data', 'fit (a='+str(np.round(s, 1))+'  r='+str(np.round(r2, 2))+')'])
    else:
        s, i, r, p, stderr = linregress(velsfit, meanvels[startpoint:endpoint])
        plt.plot(velsfit, s*velsfit+i, 'b--')
        plt.legend(['data', 'fit (a='+str(np.round(s, 1))+'  r='+str(np.round(r**2, 2))+')'])
        
    plt.savefig('figures/testvels_'+time+'.png')
    plt.show()
    return vels, meanvels
    
#vels, meanvels = testvels(time = 'early')
#pickle.dump([vels, meanvels], open('pickled/linfit_early.pickled', 'wb'))


def sequential_shibire(Type = 'D7', inhib = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]):
    cx = frmodel()
    t = 0
    ts = [t]
    for i in range(1, len(inhib)):
        t+=2000
        cx.add_event([t, 'shi', Type, inhib[i]/inhib[i-1]])
        ts.append(t)
    cx.run_model(tstop = t+2000, storedat = 100)
    cx.plotints(fname = 'figures/D7_sequential_ints.png')
    cx.plotmags(fname = 'figures/D7_sequential_mags.png')
    
    for i in range(len(inhib)):
        name = ''.join(str(inhib[i]).split('.'))
        cx.plotdist(t='EPG', time = ts[i]+1500,
                    fname = 'figures/D7_inhib_'+name+'.png')
    
    
cx1 = sequential_shibire()
        
    
    
    
    


