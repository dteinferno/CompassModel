#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:24:03 2019

@author: kris
"""

import pandas
from fr_class import frmodel
import matplotlib.pyplot as plt
import numpy as np
from utils import *

def simulate_data(ext = '', events = []):

    colnames = ['time', 'vrot', 'heading', 'EPG', 'PEN1']
    data = pandas.read_csv('example_data/data.csv', names=colnames)
    data = data[data.loc[:, 'time'] >= 4]
    data = data[data.loc[:, 'time'] <= 30]
    
    head0 = data.loc[:,'heading'].iloc[0]
    ts = list(data.loc[:,'time'])
    vrots = list(data.loc[:,'vrot'])
    heads = [head0]
    for i, vrot in enumerate(vrots[:-1]):
        newhead = heads[i] + vrot*(ts[i+1]-ts[i])
        if newhead > np.pi: heads.append(newhead-2*np.pi)
        elif newhead < -np.pi: heads.append(newhead+2*np.pi)
        else: heads.append(newhead)
    
    #plt.plot(ts, get_cont(data.loc[:,'heading']), 'b--')
        
    ts = (np.array(ts) - ts[0])*1000+2000 #give 500 ms to equilibrate
    plt.plot(ts, get_cont(data.loc[:,'EPG']), 'g--')
    plt.plot(ts, get_cont(heads), 'b--')
    plt.xlim(0, max(ts))
    plt.legend(['bump', 'heading'])
    plt.title('experimental data')
    plt.savefig('figures/real_heading.png')
    plt.show()
    
    plt.plot(ts, get_cont(data.loc[:,'EPG']), 'g--')
    plt.plot(ts, get_cont(data.loc[:,'PEN1']), 'r--')
    plt.xlim(0, max(ts))
    plt.legend(['EPG', 'PEN1'])
    plt.title('experimental data')
    plt.savefig('figures/real_EPG_PEN1.png')
    plt.show()
    
    dinit = data.loc[:,'EPG'].iloc[0]+np.pi
    dinit = int(np.round(dinit/(2*np.pi)*8-0.5))
    
    EPGinit = [0, 0, 0, 0, 0, 0, 0, 0]
    EPGinit[dinit%7] = 0.95
    EPGinit[(dinit+1)%7] = 0.85
    EPGinit[(dinit-1)%7] = 0.40
    EPGinit[(dinit+2)%7] = 0.34
    
    vrots = np.array(vrots)*360/(2*np.pi)
    cx = frmodel(init_states = {'EPG':EPGinit})
    cx.add_vrots(ts, vrots)
    
    for event in events: #add additional events such as shibire inhibition
        cx.add_event(event, plot=True)
        
    
    cx.run_model(tstop = 28000, dt = 0.1, storedat = 200, plotevents=False, 
                 plot=['ints', 'mags'])
    
    cx.plotvels(truevels = [ts, vrots, 'g--'], plotevents = False,
                fname = 'figures/sim_vel'+ext+'.png')
    cx.plotdirs(truedirs = [ts, get_cont(data.loc[:,'EPG']), 'g--'], 
                            plotevents = False, fname = 'figures/sim_bump'+ext+'.png',
                            title = 'bump direction')
    cx.plotdirs(truedirs = [ts, get_cont(heads), 'b--'], plotevents = False,
                            fname = 'figures/sim_head'+ext+'.png', title='heading')
    cx.plotmags(plotevents = False, fname = 'figures/sim_mag'+ext+'.png')#, title='PVA magnitude')
    cx.plotints(plotevents = False, fname = 'figures/sim_int'+ext+'.png')#, title='maximum intensity')
    cx.plotdirs(t=['EPG', 'PEN1'], plotevents=False,
                fname = 'figures/sim_EPG_PEN1'+ext+'.png', title='EPG PEN1')

    return cx, data

cx, data = simulate_data()
cx, data = simulate_data(ext = '_D7_07', events = [[4500, 'shi', 'D7', 0.7]])
cx, data = simulate_data(ext = '_D7_04', events = [[4500, 'shi', 'D7', 0.4]])
cx, data = simulate_data(ext = '_PEN1_07', events = [[4500, 'shi', 'PEN1', 0.7]])
cx, data = simulate_data(ext = '_PEG_07', events = [[4500, 'shi', 'PEG', 0.7]])
cx, data = simulate_data(ext = '_EPG_07', events = [[4500, 'shi', 'EPG', 0.7]])
