import numpy as np



class par:

    '''variables of the simulation'''
    high_land = 5#30 # maximum amount for the maximum capacity land cell, in burning rate units. 
    high_sea = 5#initial condition on land combustible, in burning rate units. 
    
    land_productivity = 0.05# constant for the productivity of land, in burning rate units.  
    tidal_deluge = 0.33 #amount of recovery of sea fuel, multiplictive factor of intitial conditions 

    '''configuration of the simulation'''
    length = 44#length of the land area in cells
    time = 133#max time 
    runs = 5
    t_step_lenth = 0.5 # lenth of temporal steps

    burning_rate = 1  #burning rate, this has to be 1 as all the rest will be normalized to these units.
    
    radius = 2 # maximum number of cells that the consumer can move in one jump
    n_consumers = 8# number of consumers
    min_land = 0.1*burning_rate# minimum amount of land fuel always present but negligible

    positions = np.arange(length)
    burn_frames = 33 #do not use the first N frames to compute gloval burning of resources

    plots_dir = './plots_costal_resources/'
    data_dir = './data_costal_resources/'#'../data_matrices/'
    par_names = [ 'burners_number', 'tidal_deluge', 'high_sea', 'land_productivity', 'high_land']#
    

'''limits of the simulation'''
class limits:
    min_consumers = 2
    max_consumers = 22
    con_step = 2

    min_land_prod = 0.01
    max_land_prod = 0.22
    prod_step = 0.022

    high_land_min = 2
    high_land_max = 16
    Lhigh_step = 1.3

    min_tidal_deluge = 0.1
    max_tidal_deluge = 0.4
    tidal_deluge_step = 0.033

    high_sea_min = 4
    high_sea_max = 16
    high_sea_step = 1.2

    scarce_land_prod = 0.05
    medium_tidal_deluge = 0.15
    high_tidal_deluge = 0.4