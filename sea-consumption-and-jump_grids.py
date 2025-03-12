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


''' version 0.1.1 of the code that explores under which parameter space the consumers need to consume algae in a costal desert'''
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

    #min_land_max = 2
    #max_land_max = 8
    #Lmax_step = 0.5

def generate_grid(lim):

    consumers_parameter = np.arange(lim.min_consumers, lim.max_consumers, lim.con_step )
    productivity_parameter = np.arange(lim.min_land_prod, lim.max_land_prod, lim.prod_step)
    #max_land_parameter = np.arange(lim.min_land_max, lim.max_land_max, lim.Lmax_step)

    return consumers_parameter, productivity_parameter# max_land_parameter

def call_model( consumers_number,  high_land):#land_productivity ):
    #sim = mc.constants()
    t= mc.time_steps(cnt)
    max_land, max_sea, land_vector, sea_vector = mc.create_n_inisialise_landscapes(cnt)
    cnt.n_consumers = consumers_number
    #cnt.high_land = high_land
    cnt.land_productivity = high_land

    burned_land, burned_sea, movements = mc.main(cnt)
    #consumers_array = np.random.randint(low =  0, high = cnt.length, size = consumers_number)#initial positions of the consumers
    #max_land = np.random.uniform(low =  1.9, high = high_land, size=cnt.length)#5 #maximum of land combustible resources per cell
    #sea_consumption, n_jumps_array = mc.resorurces_evol(cnt, t, Land, max_land, consumers_array) #land_productivity
    #all_sea_consumption = mc.acumulated_sea_consumption(cnt, sea_consumption)
    #all_jumps = mc.acumulated_n_jumps(cnt, n_jumps_array)
    
    return burned_land, burned_sea, movements

def run_the_grid( consumers_parameter, max_land_parameter):# productivity_parameter

    sea_consumption_matrix  = np.zeros( (len(consumers_parameter), len(productivity_parameter)  ) ) #len(max_land_parameter)
    land_consumption_matrix  = np.zeros( (len(consumers_parameter), len(productivity_parameter)  ) ) #len(max_land_parameter)
    movements = np.zeros( cnt.time ) #len(max_land_parameter)
    for i, c in enumerate(consumers_parameter):
        
        for k, p in enumerate(max_land_parameter):
            
            l, s, m = call_model( c, p)
            print ('we are in', i, k, 'and we got ', l, s)
            sea_consumption_matrix[i,k] = s
            land_consumption_matrix[i,k] = l
    
    return sea_consumption_matrix, land_consumption_matrix, movements


start  = time.perf_counter() 

name = sys.argv[1]

lim = limits()
cnt = mc.constants()
consumers_parameter, productivity_parameter  = generate_grid(lim)#max_land_parameter
matrix_sea_consumption, matrix_land_consumption, matrix_jumps = run_the_grid(consumers_parameter, productivity_parameter)#max_land_parameter
pf.plot_sea_resources_used(lim, matrix_sea_consumption, name)
pf.plot_land_resources_used(lim, matrix_land_consumption, name)
#plot_jumps(lim, matrix_jumps, name+'_jumps')


print ('time to run all this stuff', str(datetime.timedelta(seconds = (time.perf_counter() - start))))
plt.show()