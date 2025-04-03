# -*- coding: utf-8 -*-

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
from random import choice
from collections import deque
from scipy.ndimage.interpolation import shift

from subprocess import call

import agregated_sea_consumption_v9 as mc
import plots_functions as pf


import pickle
import cmath


''' version 0.9 of the code that explores under which parameter space the consumers need to consume algae in a costal desert'''
'''
the code is the same as version 0.1 sea-consumption-grid.py but changing one of the parameters of the grid, max_land_productivity instead of 
runs trough a set range of parameter spaces and computes the amount of sea resources consumed after X time steps
returns an hysotgram of searesource consumed when varying several combinations of parameters
start with land-production vs number of consumers.
'''

class limits:
    min_consumers = 2
    max_consumers = 12
    con_step = 1

    min_land_prod = 0.1
    max_land_prod = 0.33
    prod_step = 0.022

    high_land_min = 2
    high_land_max = 16
    Lhigh_step = 1.0

    min_tidal_deluge = 2
    max_tidal_deluge = 8
    tidal_deluge_step = 0.5

    high_sea_min = 4
    high_sea_max = 16
    high_sea_step = 1


def generate_grid(lim, which_par):

    consumers_parameter = np.arange(lim.min_consumers, lim.max_consumers, lim.con_step )

    if which_par == 'land_productivity':
        change_par = np.arange(lim.min_land_prod, lim.max_land_prod, lim.prod_step)
    elif which_par == 'high_land':
        change_par = np.arange(lim.high_land_min, lim.high_land_max, lim.Lhigh_step)
    elif which_par == 'tidal_deluge':
        change_par = np.arange(lim.min_tidal_deluge, lim.max_tidal_deluge, lim.tidal_deluge_step)
    elif which_par == 'high_sea':
        change_par = np.arange(lim.high_sea_min, lim.high_sea_max, lim.high_sea_step)
    else:
        print ('wrong parameter name, names shall be: ')
        print ('land_productivity, high_land, tidal_deluge, high_sea')
        print ('you wrote', which_par)
        print ('exiting')
        sys.exit()

    return consumers_parameter, change_par

def call_model( consumers_number,  chang_par, which_par):#land_productivity ):
    #sim = mc.constants()
    t= mc.time_steps(cnt)

    high_land, high_sea, land_vector, sea_vector = mc.create_n_inisialise_landscapes(cnt)
        
    cnt.n_consumers = consumers_number
    
    if which_par == 'land_productivity':
        cnt.land_productivity = chang_par
    elif which_par == 'high_land':
        cnt.high_land = chang_par
    elif which_par == 'tidal_deluge': 
        cnt.tidal_deluge = chang_par
    elif which_par == 'high_sea':    
        cnt.high_sea = chang_par

    burned_land, burned_sea, movements = mc.main(cnt)
    #consumers_array = np.random.randint(low =  0, high = cnt.length, size = consumers_number)#initial positions of the consumers
    #all_sea_consumption = mc.acumulated_sea_consumption(cnt, sea_consumption)
    #all_jumps = mc.acumulated_n_jumps(cnt, n_jumps_array)
    
    return burned_land, burned_sea, movements

def run_the_grid( consumers_parameter, change_par, which_par):# productivity_parameter

    sea_consumption_matrix  = np.zeros( (len(consumers_parameter), len(change_par)  ) ) 
    land_consumption_matrix  = np.zeros( (len(consumers_parameter), len(change_par)  ) ) 
    movements = np.zeros( cnt.time ) 
    for i, c in enumerate(consumers_parameter):
        
        for k, p in enumerate(change_par):
            
            l, s, m = call_model( c, p, which_par)
            print ('we are in', i, k, 'and we got ', l, s)
            sea_consumption_matrix[i,k] = s
            land_consumption_matrix[i,k] = l
    
    return sea_consumption_matrix, land_consumption_matrix, movements


start  = time.perf_counter() 

name = sys.argv[1]
which_par = sys.argv[2]

lim = limits()
cnt = mc.constants()
consumers_parameter, change_par  = generate_grid(lim, which_par)
matrix_sea_consumption, matrix_land_consumption, matrix_jumps = run_the_grid(consumers_parameter, change_par, which_par)
pf.plot_sea_resources_used(lim, matrix_sea_consumption, name, which_par)
pf.plot_land_resources_used(lim, matrix_land_consumption,  name, which_par)
#plot_jumps(lim, matrix_jumps, name+'_jumps')


print ('time to run all this stuff', str(datetime.timedelta(seconds = (time.perf_counter() - start))))
plt.show()