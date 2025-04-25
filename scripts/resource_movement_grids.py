# -*- coding: utf-8 -*-

from __future__ import division
import time
import datetime
import sys
import numpy as np
import scipy

from random import randint
from random import choice
from collections import deque
from scipy.ndimage.interpolation import shift

from subprocess import call

import matplotlib.pyplot as plt

import config as cfg
import agregated_sea_consumption_v9 as mc
import plots_functions as pf



''' version 0.9 of the code that explores under which parameter space the consumers need to consume algae in a costal desert'''
'''
returns an hysotgram of searesource consumed when varying several combinations of parameters
start with land-production vs number of consumers.
'''


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

def settings_mobility_seaFuels(par, lim):

    consumers_parameter = np.arange(lim.min_consumers, int(par.length/2), lim.con_step )

    low_land_mediumNhigh_sea = [lim.scarce_land_prod, lim.medium_tidal_deluge, lim.high_tidal_deluge]

    return consumers_parameter, low_land_mediumNhigh_sea


def call_model(par, consumers_number,  chang_par, which_par):#land_productivity ):
    #sim = mc.constants()

    #high_land, high_sea, land_vector, sea_vector = mc.create_n_inisialise_landscapes(par)
        
    par.n_consumers = consumers_number
    
    if which_par == 'land_productivity':
        par.land_productivity = chang_par
    elif which_par == 'high_land':
        par.high_land = chang_par
    elif which_par == 'tidal_deluge': 
        par.tidal_deluge = chang_par
    elif which_par == 'high_sea':    
        par.high_sea = chang_par
    elif which_par == 'burners_number':    
        par.n_consumers = chang_par

    norm_burned_land, norm_burned_sea, norm_movements, norm_rad = mc.main(par)
    #consumers_array = np.random.randint(low =  0, high = par.length, size = consumers_number)#initial positions of the consumers
    #all_sea_consumption = mc.acumulated_sea_consumption(par, sea_consumption)
    #all_jumps = mc.acumulated_n_jumps(par, n_jumps_array)
    
    return norm_burned_land, norm_burned_sea, norm_movements, norm_rad

def run_the_grid(par, consumers_parameter, change_par, which_par):# productivity_parameter

    sea_consumption_matrix  = np.zeros( (len(consumers_parameter), len(change_par)  ) ) 
    land_consumption_matrix  = np.zeros( (len(consumers_parameter), len(change_par)  ) ) 
    movements = np.zeros((len(consumers_parameter), len(change_par)  )) 
    radius = np.zeros((len(consumers_parameter), len(change_par)  ))
    for i, c in enumerate(consumers_parameter):
        
        for k, p in enumerate(change_par):
            
            l, s, m, r = call_model( par, c, p, which_par)
            print ('we are in', i, k, 'and we got ', l, s)
            sea_consumption_matrix[i,k] = s
            land_consumption_matrix[i,k] = l
            movements[i,k] = m
            radius = r
    
    return sea_consumption_matrix, land_consumption_matrix, movements, radius
def run_two_values(par, consumers_parameter, values, which_par):# productivity_parameter

    sea_consumption_vectors  = np.zeros( (len(consumers_parameter), 2  ) ) 
    land_consumption_vectors  = np.zeros( (len(consumers_parameter), 2  ) ) 
    movements_vectors = np.zeros( (len(consumers_parameter), 2 ) ) 
    radius_vectors = np.zeros( (len(consumers_parameter), 2 ) )

    for i, c in enumerate(consumers_parameter):
        
        for k, v in enumerate(values[1:]):
            par.land_productivity = values[0]
            l, s, m, r = call_model( par, c, v, which_par)
            print ('we are in', i, k, c, v, 'and we got ', m)
            sea_consumption_vectors[i,k] = s
            land_consumption_vectors[i,k] = l
            movements_vectors[i,k] = m
            radius_vectors[i,k] = r
    
    return sea_consumption_vectors, land_consumption_vectors, movements_vectors, radius_vectors

#python scripts/resource_movement_grids.py name tidal_deluge
def main():
    
    start  = time.perf_counter() 

    name = sys.argv[1]
    which_par = sys.argv[2]

    lim = cfg.limits()
    par = cfg.par()
    
    consumers_parameter, change_par  = generate_grid(lim, which_par)
    matrix_sea_consumption, matrix_land_consumption, matrix_jumps, matrix_radius = run_the_grid(par, consumers_parameter, change_par, which_par)
    #pf.plot_sea_resources_used(par, lim, matrix_sea_consumption, name, which_par)
    #pf.plot_land_resources_used(par, lim, matrix_land_consumption,  name, which_par)
    #pf.plot_jumps_matrix(par, lim, matrix_jumps, name, which_par)

    consumers_parameter, low_land_mediumNhigh_sea = settings_mobility_seaFuels(par, lim)
    vectors_sea_consumption, vectors_land_consumption, vectors_jumps, vectors_radius =\
          run_two_values(par, consumers_parameter, low_land_mediumNhigh_sea, 'tidal_deluge')
    pf.plot_2jumps_vectors(par, lim, consumers_parameter, vectors_jumps, name)
    pf.plot_2radius_vectors(par, lim, consumers_parameter, vectors_radius, name)

    print ('time to run all this stuff', str(datetime.timedelta(seconds = (time.perf_counter() - start))))
    
    plt.show()

if __name__ == "__main__":  
    main()