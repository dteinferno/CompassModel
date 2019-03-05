#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:16:17 2019

@author: kris
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.stats.circstats as circ


N = 8
tauEPG = 10
tauPEN = 10
tauPEG = 10
tauD7 = 10

EPG = np.zeros((N,1))
PENl = np.zeros((N,1))
PENr = np.zeros((N,1))
PEG = np.zeros((N,1))
D7 = np.zeros((N,1))

EPG[5] = 10

[w_PENl_EPG,
w_PENr_EPG,
w_PEG_EPG,
w_EPG_EPG, #allow for local recurrent interactions!
w_EPG_PENl,
w_EPG_PENr,
w_EPG_PEG,
w_EPG_D7,
w_D7_PENl,
w_D7_PENr,
w_D7_PEG] = [np.zeros((N,N)) for i in range(11)]

for i in range(8):
    for j in range(8):
        if i == j:
            w_EPG_PENl[i,j] = 1.6
            w_EPG_PENr[i,j] = 1.6
            w_EPG_PEG[i,j] = 2.7
            w_EPG_D7[i,j] = 1
            w_PEG_EPG[i,j] = 2.7
            w_EPG_EPG[i,j] = 0.3
            
        else:
            w_D7_PENl[i,j] = 1
            w_D7_PENr[i,j] = 1
            w_D7_PEG[i,j] = 1.5
            
            if j-i == 1 or j-i == -7: w_PENl_EPG[i,j] = 1.6
            if i-j == 1 or i-j == -7: w_PENr_EPG[i,j] = 1.6
            
def transfer(vector):
    
    #newvec = [1/(1+np.exp(-(val-3))) for val in vector]
    newvec = [np.tanh(val) for val in vector]
    
    return np.reshape(newvec, (len(vector), 1))



def run_model(dt = 0.1, tstop = 10, D7_inh = [500, 0.5]):
    
    ts = np.arange(0, tstop, dt)
    Nt = len(ts)
    
    global w_D7_PENl
    global w_D7_PENr
    global w_D7_PEG
    
    EPGs = np.zeros((N, Nt))
    PEGs = np.zeros((N, Nt))
    D7s = np.zeros((N, Nt))
    PENls = np.zeros((N, Nt))
    PENrs = np.zeros((N, Nt))
    
    EPG = np.zeros((N,1))
    PENl = np.zeros((N,1))
    PENr = np.zeros((N,1))
    PEG = np.zeros((N,1))
    D7 = np.zeros((N,1))

    #EPG[:,0] = [0.3, 0.3, 0.4, 0.5, 0.4, 0.3, 0.3, 0.3] #stabilize
    EPG[:,0] = [0, 0, 0.37, 1, 0.37, 0, 0, 0]
    
    vr = 0
    vl = 0
    
    for i in range(len(ts)-1):
        EPGs[:, i] = EPG[:,0]
        PEGs[:, i] = PEG[:,0]
        D7s[:, i] = D7[:,0]
        PENls[:, i] = PENl[:,0]
        PENrs[:, i] = PENr[:,0]
        
        t = ts[i]
        
        
        #if t == 500: vr = 1
        if t == 900: vr = 0
        #if t == 900: vl = 1
        #if t ==1150: vl = 0
        #if t == 600: vl = 0.3
        #if t == 800: vl = 0
        
        if len(D7_inh) > 0:
            if t == D7_inh[0]:
                w_D7_PENl *= D7_inh[1]
                w_D7_PENr *= D7_inh[1]
                w_D7_PEG *= D7_inh[1]
        
        
        EPG = np.maximum(EPG + dt/tauEPG * ( -EPG + 
                transfer(w_PENl_EPG.dot(PENl) + w_PENr_EPG.dot(PENr) + 
                         w_PEG_EPG.dot(PEG) + w_EPG_EPG.dot(EPG) )), 0)
    
        PEG = np.maximum(PEG + dt/tauPEG * ( -PEG + 
                transfer(w_EPG_PEG.dot(EPG) - w_D7_PEG.dot(D7) )),0)
        
        D7 = np.maximum(D7 + dt/tauD7 * ( -D7 + 
                              transfer(w_EPG_D7.dot(EPG))),0)
        
        PENl = np.maximum(PENl + dt/tauPEN * ( -PENl + 
                    transfer(w_EPG_PENl.dot(EPG) - w_D7_PENl.dot(D7) + vl)),0)
        
        PENr = np.maximum(PENr + dt/tauPEN * ( -PENr + 
                    transfer(w_EPG_PENr.dot(EPG) - w_D7_PENr.dot(D7) + vr)),0)
        
    EPGs[:, i+1] = EPG[:,0]
    PEGs[:, i+1] = PEG[:,0]
    D7s[:, i+1] = D7[:,0]
    PENls[:, i+1] = PENl[:,0]
    PENrs[:, i+1] = PENr[:,0]
    return ts, [EPGs, PENls, PENrs, PEGs, D7s]

def getvec(array):
    array = array / np.mean(array)
    N = len(array)
    xs = [array[i]*np.cos(2*np.pi/N*i) for i in range(N)]
    ys = [array[i]*np.sin(2*np.pi/N*i) for i in range(N)]
    x, y = np.mean(xs), np.mean(ys)
    
    r = np.sqrt(x**2+y**2)
    
    if x == 0:
        if y > 0: ang =  np.pi/2
        else: ang = -np.pi/2
        
    elif x > 0:
        if y > 0: ang = np.arctan(y/x)
        else: ang = 2*np.pi+np.arctan(y/x)
    else: ang = np.pi + np.arctan(y/x)
    
    return r, ang
    
def getdirs(array):
    return [getvec(array[:,i])[1] for i in range(array.shape[1])]
def getmags(array):
    return [getvec(array[:,i])[0] for i in range(array.shape[1])]


#def getvar(array):
#    return circ.circvar(array, 0)
    
def plotdirs(ts, array):
    dirs = getdirs(arrs[0])
    plt.plot(ts, dirs)
    plt.ylim(0, 2*np.pi)
    plt.xlim(ts[0], ts[-1])
    plt.yticks([0, np.pi, 2*np.pi], ['0', 'pi', '2pi'])
    plt.title('direction')
    plt.show()
    
def plotmags(ts, array):
    dirs = getmags(arrs[0])
    plt.plot(ts, dirs)
    plt.ylim(0, 1)
    plt.xlim(ts[0], ts[-1])
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
    plt.title('magnitude')
    plt.show()
    
def plotints(ts, array):
    ints = np.sum(arrs[0], 0)
    plt.plot(ts, ints)
    plt.ylim(0, np.amax(ints))
    plt.xlim(ts[0], ts[-1])
    #plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
    plt.title('intensity')
    plt.show()
    
def plotdist(array):
    plt.bar(range(len(array)), array, width=1)
    plt.show()


ts, arrs = run_model(tstop=1500)
plotdirs(ts, arrs[0], fname = 'stable.png')
plotmags(ts, arrs[0], fname = 'D7_interrupt.png')
plotints(ts, arrs[0])

#EPG:

#dEPG = dt/tauEPG*(-EPG + transfer(w_PENl_EPG.dot(PENl) + w_PENr_EPG.dot(PENr) + w_PEG_EPG.dot(PEG) ))

#dPENl = dt/tauPEN*(-PENl + transfer(w_EPG_PENl.dot(EPG) + w_D7_PENl.dot(D7) + vl))
#dPENr = dt/tauPEN*(-PENr + transfer(w_EPG_PENr.dot(EPG) + w_D7_PENr.dot(D7) + vr))

#dPEG = dt/tauPEG*(-PEG + transfer(w_EPG_PEG.dot(EPG) + w_D7_PEG.dot(D7) ))

#dD7 = dt/tauD7*(-D7 + transfer(w_EPG_D7.dot(EPG)))

