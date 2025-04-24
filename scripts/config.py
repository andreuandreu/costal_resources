import numpy as np



class par:

    '''variables of the simulation'''
    high_land = 5#30 # maximum amount for the maximum capacity land cell, in burning rate units. 
    high_sea = 8#2*high_land#high_land*0.67#4.5 #initial condition on land combustible, in burning rate units. 
    
    land_productivity = 0.2# constant for the productivity of land, in burning rate units.  
    tidal_deluge = 0.4 #amount of recovery of sea fuel, multiplictive factor of intitial conditions 

    '''configuration of the simulation'''
    length = 44#length of the land area in cells
    time = 433 #max time 
    t_step_lenth = 0.5 # lenth of temporal steps

    burning_rate = 1  #burning rate, this has to be 1 as all the rest will be normalized to these units.
    
    radius = 4 # maximum number of cells that the consumer can move in one jump
    n_consumers = 8# number of consumers
    min_land = 0.01*burning_rate# minimum amount of land fuel always present but negligible

    positions = np.arange(length)
    burn_frames = 33 #do not use the first N frames to compute gloval burning of resources

    plots_dir = 'plots/'

'''limits of the simulation'''
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

    min_tidal_deluge = 0.2
    max_tidal_deluge = 0.45
    tidal_deluge_step = 0.025

    high_sea_min = 4
    high_sea_max = 16
    high_sea_step = 1.0

    scarce_land_prod = 0.15
    medium_tidal_deluge = 0.25
    high_tidal_deluge = 0.4


