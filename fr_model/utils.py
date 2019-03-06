#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 16:40:29 2019

@author: kris
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt

def get_cont(vec, thresh = 1.2*np.pi):
    vec = list(vec)
    for i in range(1,len(vec)):
        if abs(vec[i]-vec[i-1]) > thresh:
            vec[i-1] = np.nan
    return vec

def optimize_params(cx):
    
    print('ok')
    


