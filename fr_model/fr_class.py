#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:16:17 2019

@author: kris
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.stats.circstats as circ
import scipy.io as sio
from utils import *


class frmodel():
    '''class for implementing CX firing rate models. Includes methods
    for simulating shibire experiments by inhibiting neuronal output'''

    def __init__(self, init_states = {'EPG':[0, 0, 0.38, 0.95, 0.85, 0.29, 0, 0]}):
        '''
        initialize model activities as init_states. Specifies activities for 8 tile neurons
        '''
        self.types = ['EPG', 'PEG', 'D7', 'PENl', 'PENr']
        
        N = 8
        self.N = 8
        self.tau = {'EPG': 80,
                    'PEN': 65,
                    'PEG': 70,
                    'D7': 70} #time constants in ms
        #herve tau = 80 (EPG), 65 (PEN)
        #tau = RmCm = (20800 Ohm cm2)(0.79 uF cm-2) = 16.42ms
        
        #construct weight matrices
        ws = {'EPG':{'EPG':{}, 'PENl':{}, 'PENr':{}, 'D7':{}, 'PEG':{}},
                   'PENl':{'EPG':{}},
                   'PENr':{'EPG':{}},
                   'PEG':{'EPG':{}},
                   'D7':{'PENl':{}, 'PENr':{}, 'PEG':{}}}
        for pre, dic in ws.items():
            for post in dic.keys():
                ws[pre][post] = np.zeros((N,N)) #initialize
                
        for i in range(8):
            for j in range(8):
                if i == j:
                    #same-tile activation. Label PEN neurons accordin to the
                    #EB tile of the EPG neuron innervating them
                    ws['EPG']['PENl'][i,j] = 1.60
                    ws['EPG']['PENr'][i,j] = 1.60
                    ws['EPG']['PEG'][i,j] = 2.7
                    ws['EPG']['D7'][i,j] = 0.80
                    ws['PEG']['EPG'][i,j] = 2.7
                    ws['EPG']['EPG'][i,j] = 0.40
                else:
                    ws['D7']['PENl'][i,j] = 1 #D7 gives global inhibition
                    ws['D7']['PENr'][i,j] = 1
                    ws['D7']['PEG'][i,j] = 1.5
                    #PEN1s innervate adjacent EPG
                    if j-i == 1 or j-i == -(N-1): ws['PENl']['EPG'][i,j] = 1.55
                    if i-j == 1 or i-j == -(N-1): ws['PENr']['EPG'][i,j] = 1.55      
        self.ws = ws
        
        self.state = {'EPG': np.zeros((N,1)),
                'PENl': np.zeros((N,1)),
                'PENr': np.zeros((N,1)),
                'PEG': np.zeros((N,1)),
                'D7': np.zeros((N,1))}
        
        #add initial states from input
        for key in init_states.keys(): self.state[key][:,0] = init_states[key]
        
        self.vr = 0
        self.vl = 0 #initialize velocities at 0
        self.plottimes = []
        
        self.events = {} #angular velocity
            
    def add_event(self, event, plot=False):
        t = event[0]
        if t in self.events.keys():
            self.events[t].append(event)
        else:
            self.events[t] = [event]
        if plot: self.plottimes.append(t)
            
    def add_vrots(self, ts, vrots):
        for i in range(len(ts)):
            self.add_event([ts[i], 'vel', vrots[i]/195])#*0.9/200])
            
    def process_event(self, events):
        '''
        [t, 'vel', velocity] +ve is right, -ve is left
        [t, 'vl', velocity]
        [t, 'shi', neuron, multiplicative_factor]
        [t, 'opto', neuron, indices, newactivity]
        '''
        
        for event in events:
            if event[1] == 'vel': 
                vel = event[2]
                if vel == 0: self.vr, self.vl = 0, 0
                elif vel > 0: self.vr, self.vl = vel, 0
                else: self.vr, self.vl = 0, np.abs(vel)
            elif event[1] == 'shi':
                if event[2] == 'PEN1':
                    for PEN in ['PENl', 'PENr']:
                        for output in self.ws[PEN].keys():
                            self.ws[PEN][output] *= event[3]
                else:
                    for output in self.ws[event[2]].keys():
                        self.ws[event[2]][output] *= event[3]
            elif event[1] == 'opto':
                self.state[event[2]][event[3]] = event[4]
            elif event[1] == 'end': print('simulation has finished')
        
        
    def transfer(self, vector):
        '''apply transfer function to a vector of inputs
        use tanh transfer function'''
        #newvec = [1/(1+np.exp(-(val-3))) for val in vector]
        newvec = [np.tanh(val) for val in vector]
        return np.reshape(newvec, (len(vector), 1))

    def timestep(self, dt):
        '''update firing rates for all neurons'''
        
        self.state['PENr'] = np.maximum(
                            self.state['PENr'] + 
                            dt/self.tau['PEN'] * ( 
                            -self.state['PENr'] + 
                            self.transfer(
                            self.ws['EPG']['PENr'].dot(self.state['EPG']) - 
                            self.ws['D7']['PENr'].dot(self.state['D7']) + 
                            self.vr
                            )),0)
        
        self.state['PENl'] = np.maximum(
                            self.state['PENl'] + 
                            dt/self.tau['PEN'] * ( 
                            -self.state['PENl'] + 
                            self.transfer(
                            self.ws['EPG']['PENl'].dot(self.state['EPG']) - 
                            self.ws['D7']['PENl'].dot(self.state['D7']) + 
                            self.vl
                            )),0) #get angular velocity input
        
        self.state['D7'] = np.maximum(
                            self.state['D7'] + 
                            dt/self.tau['D7'] * ( 
                            -self.state['D7'] + 
                            self.transfer(
                            self.ws['EPG']['D7'].dot(self.state['EPG'])
                            )),0) #only EPG input
        
        self.state['PEG'] = np.maximum(
                            self.state['PEG'] + 
                            dt/self.tau['PEG'] * ( 
                            -self.state['PEG'] + 
                            self.transfer(
                            self.ws['EPG']['PEG'].dot(self.state['EPG']) - 
                            self.ws['D7']['PEG'].dot(self.state['D7'])
                            )),0) #D7 inhibitory
        
        self.state['EPG'] = np.maximum(
                            self.state['EPG'] + 
                             dt/self.tau['EPG'] * (
                            -self.state['EPG'] + 
                            self.transfer(
                            self.ws['PENl']['EPG'].dot(self.state['PENl']) + 
                            self.ws['PENr']['EPG'].dot(self.state['PENr']) + 
                            self.ws['PEG']['EPG'].dot(self.state['PEG']) + 
                            self.ws['EPG']['EPG'].dot(self.state['EPG'])
                            )),0) 
        
        self.n += 1 #number of timesteps taken
        

    def run_to_next_event(self, dt, tstart, tstop):
        '''solve diff equations from tstart until tstop when next event occurs'''
        ts = np.arange(tstart, tstop, dt)
        for i in range(len(ts)):
            self.timestep(dt) #update firing rates
            if self.n % self.storedat == 0: #occasionally store data
                self.ind += 1
                if self.EPGs.shape[1] > self.ind:
                    self.EPGs[:, self.ind] = self.state['EPG'][:,0] #store results
                    self.PEGs[:, self.ind] = self.state['PEG'][:,0]
                    self.D7s[:, self.ind] = self.state['D7'][:,0]
                    self.PENls[:, self.ind] = self.state['PENl'][:,0]
                    self.PENrs[:, self.ind] = self.state['PENr'][:,0]
            
        
    def run_model(self, dt = 0.1, tstop = 1000, storedat = 200,
                  plot = ['dirs', 'mags', 'ints', 'vels'], plotevents = True):
        '''runs the model that has been built'''
        
        self.dt = dt
        self.ts = np.arange(0, tstop, dt) #timesteps
        Nt = len(self.ts)
        N = self.N #number of neurons
        self.add_event([tstop, 'end']) #add finishing the simulation as an event
        self.storedat = storedat
        
        self.plotts = np.arange(0, tstop, dt*storedat)
        #if Nt % storedat == 0: ndat = int(Nt/storedat)
        #ndat = int(Nt/storedat)+1 #datapoints to be stored
        ndat = len(self.plotts)
        
        self.EPGs = np.zeros((N, ndat))
        self.PEGs = np.zeros((N, ndat))
        self.D7s = np.zeros((N, ndat))
        self.PENls = np.zeros((N, ndat))
        self.PENrs = np.zeros((N, ndat))
        self.EPGs[:, 0] = self.state['EPG'][:,0] #store initial state
        self.PEGs[:, 0] = self.state['PEG'][:,0]
        self.D7s[:, 0] = self.state['D7'][:,0]
        self.PENls[:, 0] = self.state['PENl'][:,0]
        self.PENrs[:, 0] = self.state['PENr'][:,0]
        
        self.eventtimes = np.sort(list(self.events.keys())) #times where stuff happens
        
        self.n, self.ind = 0, 0 #no updates carried out
        
        prevtime = dt #have already store initial state
        for nexttime in self.eventtimes:
            #run simulation until next event
            self.run_to_next_event(dt, prevtime, nexttime)
            self.process_event(self.events[nexttime])
            prevtime = nexttime #update current time
        
        #store our calculated firing rates
        self.rates = {'EPG': self.EPGs,
                'PEG': self.PEGs,
                'D7': self.D7s,
                'PENl': self.PENls,
                'PENr': self.PENrs}
        self.combinePEN1() #get combined PEN intensities in EB
        self.storevecs() #calculate PVAs
        
        #plot some summary results for EPG neurons
        if 'dirs' in plot: self.plotdirs(plotevents=plotevents)
        if 'mags' in plot: self.plotmags(plotevents=plotevents)
        if 'vels' in plot: self.plotvels(plotevents=plotevents)
        if 'ints' in plot: self.plotints(plotevents=plotevents)
        
        
        return self.ts, self.rates
    
    def combinePEN1(self):
        p1 = np.roll(self.rates['PENl'], -1, 0)
        p2 = np.roll(self.rates['PENr'], 1, 0)
        rates = p1 + p2
        self.rates['PEN1'] = rates
        self.types.append('PEN1')
    
    def getvec(self, array):
        '''get polar coordinates of mean vector from array of N activities'''
        
        m = np.mean(array)
        if m == 0: return 0, 0 #no intensity
        array = array / m
        N = len(array)
        xs = [array[i]*np.cos(2*np.pi/N*i) for i in range(N)]
        ys = [array[i]*np.sin(2*np.pi/N*i) for i in range(N)]
        x, y = np.mean(xs), np.mean(ys) #average x and y coordinates
        
        r = np.sqrt(x**2+y**2) #magnitude
        if x == 0:
            if y > 0: ang =  np.pi/2
            else: ang = 3*np.pi/2
        elif x > 0:
            if y > 0: ang = np.arctan(y/x)
            else: ang = 2*np.pi+np.arctan(y/x)
        else: ang = np.pi + np.arctan(y/x)
        
        return r, ang-np.pi
    
    def get_vel(self, pos1, pos2):
        v = pos2 - pos1 #compute angular velocity of bump
        if np.abs(v) < np.pi: vel =  v #we have not jumped across
        elif v < 0: vel = v+2*np.pi #jumped across with +ve vel
        else: vel = v-2*np.pi #jumped across with -ve vel
        return vel/(self.dt*self.storedat) * 1000 * 360/(2*np.pi) #degrees per second
        
        
    def storevecs(self):
        '''get polar coordinates from vectors at all timepoints and store these'''
        self.rs = {}
        self.thetas = {}
        self.vels = {}
        for t in self.types: #go through all types of neurons
            arr = self.rates[t] #find rate data
            thetas = []
            rs = []
            vels = [] #angular velocity of bump
            a0 = [0, 0]
            for i in range(arr.shape[1]): #for each timepoint, calculate polar coords
                a = self.getvec(arr[:,i])
                rs.append(a[0])
                thetas.append(a[1])
                vels.append(self.get_vel(a0[1], a[1]))
                a0 = a
            vels[0] = 0
            self.rs[t] = rs #store data
            self.thetas[t] = thetas
            self.vels[t] = vels
        
    def plotdirs(self, t='EPG', plotevents=True, fname = 'none', truedirs = [], title='none',
                 cols=['g--', 'r--']):
        '''plot PVA directions for neuron type t over time'''
        if type(t) == str:
            plt.plot(self.plotts, get_cont(self.thetas[t]), 'k-')
        else:
            for i, newtype in enumerate(t):
                plt.plot(self.plotts, get_cont(self.thetas[newtype]), cols[i])
                
        if len(truedirs) > 0:
            plt.plot(truedirs[0], truedirs[1], truedirs[2])
            plt.legend(['simulated', 'real'])
        plt.ylim(-np.pi, np.pi)
        plt.xlim(self.plotts[0], self.plotts[-1])
        plt.yticks([-np.pi, 0, np.pi], ['-pi', '0', 'pi'])
        if title == 'none': plt.title('direction')
        else: plt.title(title)
        if plotevents: [plt.axvline(x=t, linestyle='--') for t in self.eventtimes]
        [plt.axvline(x=t, linestyle='--') for t in self.plottimes]
        if not fname == 'none': plt.savefig(fname)
        plt.show()
        
    def plotmags(self, t='EPG', plotevents=True, fname = 'none'):
        '''plot PVA magnitudes for neuron type t over time'''
        plt.plot(self.plotts, self.rs[t], 'k-')
        plt.ylim(0, 1)
        plt.xlim(self.plotts[0], self.plotts[-1])
        plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
        plt.title('magnitude')
        if plotevents: [plt.axvline(x=t, linestyle='--') for t in self.eventtimes]
        if not fname == 'none': plt.savefig(fname)
        plt.show()
        
    def plotints(self, t='EPG', plotevents=True, fname = 'none'):
        '''plot mean intensities for neuron type t over time'''
        ints = np.max(self.rates[t], 0)
        plt.plot(self.plotts, ints, 'k-')
        plt.ylim(0, np.amax(ints))
        plt.xlim(self.plotts[0], self.plotts[-1])
        #plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
        plt.title('intensity')
        if plotevents: [plt.axvline(x=t, linestyle='--') for t in self.eventtimes]
        if not fname == 'none': plt.savefig(fname)
        plt.show()
        
    def plotvels(self, t='EPG', plotevents=True, fname = 'none', truevels = [], title='none'):
        '''plot mean intensities for neuron type t over time
        if we're comparing with real data, we can add this to the plot as
        truevels = [times, vels]'''
        plt.plot(self.plotts, self.vels[t], 'k-')
        if len(truevels) > 0: plt.plot(truevels[0], truevels[1], truevels[2])
        plt.ylim(np.amin(self.vels[t]), np.amax(self.vels[t]))
        plt.xlim(self.plotts[0], self.plotts[-1])
        
        if len(truevels) > 0:
            plt.plot(truevels[0], truevels[1], 'g--')
            plt.ylim(np.amin(self.vels[t]), np.amax(np.append(self.vels[t], truevels[1])))
            plt.legend(['simulated', 'real'])
            
        #plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
        if title == 'none': plt.title('bump angular velocity')
        else: plt.title(title)
        if plotevents: [plt.axvline(x=t, linestyle='--') for t in self.eventtimes]
        if not fname == 'none': plt.savefig(fname)
        plt.show()
        
    def plotdist(self, t='EPG', time = 10, fname = 'none', cols = [[0,1,0,0.5],[1,0,0,0.5]]):
        '''plot activity distribution for neuron type t at time'''
        ind = np.argwhere(self.plotts == time)[0][0]
        if type(t) == str:
            plt.bar(range(self.N), self.rates[t][:,ind], width=1)
        else:
            for i, newtype in enumerate(t):
                plt.bar(range(self.N), self.rates[newtype][:,ind], width=1, color = cols[i])
        if not fname == 'none': plt.savefig(fname)
        plt.show()


def testmodel():
    cx = frmodel()
    cx.add_event([300, 'shi', 'D7', 0.5]) #inhibit
    cx.add_event([500, 'shi', 'D7', 2]) #back to normal
    cx.add_event([510, 'opto', 'EPG', 3, 2]) #zap back into bump
    cx.add_event([700, 'vel', 1]) #rotate right
    cx.run_model(tstop=1000)

#testmodel()    
#testvels()
    
cx = frmodel()
cx.add_event([1000, 'vel', 0.45])
cx.add_event([2000, 'vel', 0])
cx.add_event([3000, 'vel', -0.45])
cx.add_event([4000, 'vel', 0.45])
cx.add_event([5000, 'vel', 0])
#cx.run_model(tstop = 6000, dt = 0.1, storedat = 200)
#cx.plotdist(time=90, t='EPG', fname='EPG_bump.png')
#cx.plotdist(time=2500, t='EPG', fname='EPG_bump_D7_0.35.png')


'''
matdat= '/Users/kris/Documents/jayaraman/summer2018/\
Kris/Documents/imaging/Data_Dan/EB/PEN1_R_EPG_G_EB/cont.mat'

matdat = sio.loadmat(matdat)
matdat['alldata']
dat = matdat['alldata'][0,0][0,0][2] #shape 5,1 for 5 flies
fly = dat[0,3] #look at fly 4
fly = fly[0,0][1] #this contains the datya
trial = fly[0,2][0,0] #look at trial 3
'''

#EPG:

#dEPG = dt/tauEPG*(-EPG + transfer(w_PENl_EPG.dot(PENl) + w_PENr_EPG.dot(PENr) + w_PEG_EPG.dot(PEG) ))

#dPENl = dt/tauPEN*(-PENl + transfer(w_EPG_PENl.dot(EPG) + w_D7_PENl.dot(D7) + vl))
#dPENr = dt/tauPEN*(-PENr + transfer(w_EPG_PENr.dot(EPG) + w_D7_PENr.dot(D7) + vr))

#dPEG = dt/tauPEG*(-PEG + transfer(w_EPG_PEG.dot(EPG) + w_D7_PEG.dot(D7) ))

#dD7 = dt/tauD7*(-D7 + transfer(w_EPG_D7.dot(EPG)))

