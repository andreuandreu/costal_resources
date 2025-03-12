# -*- coding: utf-8 -*-
"""model of resource allocation in desertic costal plain"""
from __future__ import division
import time
import datetime
import sys
import colorsys
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from random import randint
from collections import deque
from scipy.ndimage.interpolation import shift
import matplotlib.animation as animation
import matplotlib
from subprocess import call
import matplotlib.ticker as ticker

import pickle
import cmath

''' basic genertion and consumption of resources'''



class constants:

    land_0 = 4.5
    land_productivity = 0.3
    max_land = 9
    sea_productivity = 1
    consumption_rate = 1
    L_threshold = 0# 0.4





def time_steps():
    
    t = np.arange(1,10, 0.5)
    
    return t

def sea_production(t):
    sea_vector = np.random(0, 1, len(t))
    return sea_vector

def land_model_1exp():

    L = cnt.L_threshold
    L_array = []
    for i in range(10000):
        L += cnt.land_productivity/np.exp(L)
        L_array.append(L)
    plt.plot(L_array)


def resorurces_evol(t, c ):

    resources = []
    consumption = []
    production = []
    L = cnt.land_0
    
    for e in t:
        if L > cnt.L_threshold:
            R = L - c
            #p = cnt.land_productivity/np.exp(L+cnt.L_threshold) 
            p = cnt.land_productivity*(1-L/cnt.max_land)*L 
            #p = cnt.land_productivity*(1-L/cnt.max_land)*L
            L += p - c
            
            print('land')
        else:            
            land_margin = L - cnt.L_threshold
            #p = cnt.land_productivity/np.exp(L+cnt.L_threshold)
            p = cnt.land_productivity*(1-L/cnt.max_land)*L
            R = land_margin + cnt.sea_productivity*np.random.rand(1) - c
            L += p - land_margin
            resources.append(R)
            print('sea/land')
            if R < 0:
                print('resources exahsted!111, jumping to next cell')

        resources.append(np.array(R).flatten())
        consumption.append(np.array(c).flatten())
        production.append(np.array(p).flatten())

    return resources, consumption, production
    
def land_model_Log(t):

    L = cnt.land_productivity*np.log(t) + cnt.L_threshold
    
    plt.plot( L)

def plot_resources(series, names ):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("t")
    ax.set_ylabel("r")
 
    for s, n in zip(series, names):
        ax.plot(s, label = n)
    
    ax.legend()



cnt = constants()
t= time_steps()
#sea, land = resorurces(t)
resources, consumption, production = resorurces_evol(t, cnt.consumption_rate)

plot_resources([resources, consumption, production], ['resources','consumption','production'] )
#land_model_1exp()
#land_model_Log(t)
plt.show()













