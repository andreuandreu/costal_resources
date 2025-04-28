# -*- coding: utf-8 -*-
"""model of resource allocation in desertic costal plain"""
from __future__ import division
import numpy as np
import scipy
import random
from random import randint
from random import choice
from random import randrange

from scipy.ndimage.interpolation import shift

from subprocess import call

import config as cfg
import plots_functions as pf



''' version 0.9 of the code of costal resources'''
'''

Ninth instance of the code
What is new is the starting from scratch the method to make the consumers jump
and a new initalisation of the sea fuel and its regeneration

the Land resources are < burning "sea-burning-grid_max-land.py" is to
call this pacage.

it would require the imput parameters:
land_productivity
number_of_consumers

it would produce the output:
sum_of_sea_burning

'''


def create_n_inisialise_landscapes(par):

    '''create land vector'''
    max_land = np.random.uniform(low = par.min_land , high = par.high_land, size=par.length)#5 #maximum of land combustible resources per cell, in burning rate units. 
    max_sea = np.random.uniform(low =  0.0, high = par.high_sea, size=par.length)
    
    #land_vector= np.full(l, par.high_land )#uniform initial conditions
    land_vector= np.random.uniform(low =  0, high = 1, size=par.length ) * max_land  #random initial conditions
    sea_vector= np.random.uniform(low =  0, high = 1, size=par.length ) * max_sea  #random initial conditions    
    
    return max_land, max_sea, land_vector, sea_vector 


def time_steps(par):
    '''temporal array to set the simulation steps'''
    t = np.arange(1, par.time, par.t_step_lenth)
    
    return t


def fuels_evol(par, land_arr, sea_arr, max_land, max_sea):

    '''operates the vector of fuels, replenishes fuel levels naturally after each time step
    
    Parameters:
    par (class): contains model parameters 
    land_arr (array): level of land fuel in each cell.
    sea_arr (float): level of sea cell in each cell.

    Returns:
    updated values of fuels
    '''

    ###land fuel
    increase_land_fuel = par.land_productivity* (1 - land_arr/max_land) * land_arr 
    new_land_arr = land_arr + increase_land_fuel + par.min_land
    
    ###sea fuel
    new_sea_arr = sea_arr.copy()
    for i, s in enumerate(sea_arr):
        accumulated = s + par.tidal_deluge
        if accumulated <= max_sea[i] - par.tidal_deluge:
            new_sea_arr[i] = accumulated
        else:
            new_sea_arr[i] = max_sea[i]

    return  new_land_arr, new_sea_arr
    

def consume(burning, land, sea ):
    """
    returns new values for land and sea fuels for one cell depending on the initial values.

    Parameters:
    burning (float): level of burning per unit of time per burner
    land (float): level of land fuel in one cell.
    sea (float): level of sea cell in the same cell.

    Returns:
    updated values of fuels
    """

    if land >= burning:
        
        new_land = land - burning 

        #print('consume only land, {:2.2f}'.format(land), '{:.2f}'.format(new_land), burning)
        return new_land, sea
    
    elif land + sea > burning:
        new_land = par.min_land
        new_sea = sea - (burning - land + par.min_land)
        consumed = sea-new_sea + land - new_land
        #print('consume sea and land', '{:2.2f}'.format(land), '{:.2f}'.format(new_land), '{:.2f}'.format(sea), '{:.2f}'.format(new_sea), consumed )
        return new_land, new_sea
    
    else:
        new_land = par.min_land
        #print('consume leftover fuel', '{:2.2f}'.format(land), '{:2.2f}'.format(sea), 0 )
        return new_land, 0
    

def accumulated_burnings(par, land_fuel, sea_fuel):    
    '''computes the accumulated sea burning of all the consumers after a burndown period
    
    this is the mean sea burning by all the agents through all the time, after a burnout 
    period is removed'''

    land_matrix = np.matrix(np.array(land_fuel[par.burn_frames:]))
    all_land = np.matrix.sum(land_matrix)/(len(land_fuel[par.burn_frames:])*par.n_consumers)

    sea_matrix = np.matrix(np.array(sea_fuel[par.burn_frames:]))
    all_sea = np.matrix.sum(sea_matrix)/(len(sea_fuel[par.burn_frames:])*par.n_consumers)
    #print ('all the sea resources ', all_sea)

    return all_land, all_sea

def accumulated_n_jumps(par, positions):    
    '''computes the accumulated jumps per consumer after a burndown period
    
    this is the averaged by all the agents through all the time, after a burnout 
    period is removed'''
    
    pos_matrix = np.array(positions[par.burn_frames:])
    #print('jump matrix', pos_matrix)
    total_jumps = 0
  
    for i, l in enumerate(pos_matrix):
        if i < len(pos_matrix)-1:
            #time_jumps = np.sum(l != pos_matrix[i+1])
            
            time_jumps = len(l)-np.sum(l == pos_matrix[i+1])
            #print( '\nllllllll', np.array(l))
            #print( 'llllllll', pos_matrix[i+1])
            #print(len(l), len(pos_matrix[i+1]), 'sum', np.sum(l == pos_matrix[i+1]))
            #print('time jumps', time_jumps, total_jumps, '\n')
        total_jumps = total_jumps + time_jumps
        
    norm_jumps = total_jumps/(len(positions[par.burn_frames:])*par.n_consumers)
    #print ('\n all the jumps ', total_jumps, 'jump length', len(positions[par.burn_frames:]), norm_jumps, '\n')
   
    return norm_jumps

def accumulated_mean_R(par, radius):    
    '''computes the accumulated mean R per consumer after a burndown period
    
    this is the averaged by all the agents through all the time, after a burnout 
    period is removed'''

    all_R = np.sum(np.array(radius[par.burn_frames:]))
    #print ('all the radius', all_R, len(radius[par.burn_frames:]))
    #print ('all the radius', radius[par.burn_frames:])
    norm_R = all_R/(len(radius[par.burn_frames:])*par.n_consumers)#
    
    return norm_R


def maximum_R(par, radius):    

    max_R = np.max(np.array(radius[par.burn_frames:]))
    return max_R

def initialize_agents(n, s):
    """
    Randomly initializes the positions of n agents within the array indices.

    Parameters:
    n (int): Number of agents.
    s (int): Dimension of the array.

    Returns:
    list: List of agent positions.
    """
    return random.sample(range(s), n)


def shake_burners(par, burners, land_arr, sea_arr):

    """
    Determines the action to take among 2 options for the list of burners and updates the fuel arrays:
    - consume fuels
    - move and consume 

    Parameters:
    land_arr (numpy.ndarray): Array of values.
    sea_arr (numpy.ndarray): Array of values.
    burners (list): Current positions of agents.
    par (int): class containing parameters of the simulation

    Returns:
    list: Updated fuel levels.
    """

    new_land_arr = np.copy(land_arr)
    new_sea_arr = np.copy(sea_arr)
    radius = 0

    for i, pos in enumerate(burners):
        #print('burner', i, 'position', pos)
        R = par.radius
        if new_land_arr[pos] + new_sea_arr[pos] < par.burning_rate:
            burners, R = move_burner(i, pos, new_land_arr, new_sea_arr, burners, par.radius)
        #    radius = np.append(radius, R)    
        #else:
        radius = np.append(radius, R)

        new_land_arr[pos], new_sea_arr[pos] = consume(par.burning_rate, new_land_arr[pos], new_sea_arr[pos] )
    #print('radiussss', radius)
    return burners, new_land_arr, new_sea_arr, radius


def move_burner(i, burner_pos, land_arr, sea_arr, burners, R):
    """
    Moves one agent to the maximum value in a radius R that is not occupied.

    Parameters:
    array (numpy.ndarray): Array of values.
    agents (list): Current positions of agents.
    R (int): Radius within which agents can move.

    Returns:
    list: Updated agent positions.
    """

    new_positions = burners.copy()

    # Define the range to search within radius R
    start = max(0, burner_pos - R)
    end = min(len(land_arr), burner_pos + R + 1)

    # Find the maximum value in the range not occupied by other agents
    max_value = max(land_arr[start:end])
    best_position = burner_pos

    for pos in range(start, end):
        
        if pos not in new_positions and land_arr[pos] == max_value:
            #max_value = land_arr[pos]
            best_position = pos
            #print('best position', best_position, ' best choice ', land_arr[pos], max_value )
            #R = abs(pos - burner_pos)
            R = par.radius
            break
    
    if best_position == burner_pos and land_arr[pos] == max_value:
        #print('lost position', best_position, ' possible choices ', land_arr[start:end], land_arr[burner_pos], max_value )
        rand_position = random.choice(np.arange(start, end))
        while rand_position  in new_positions and land_arr[pos]  <= par.burning_rate/2:#max_value+ sea_arr[pos]
            max_value = max(land_arr[start:end])+par.min_land
            rand_position = random.choice(np.arange(start, end))
            start = max(0, start -1)
            end = min(len(land_arr), end + 1)
            
            #best_position = rand_position
        #print('rand position', rand_position, ' choices ', np.arange(start, end), 'original pos', burner_pos)
        best_position = rand_position
        #R = (end - start)/2
        R = abs(burner_pos - best_position)

               
    # Update the agent's position
    burners[i] = best_position

    return burners, R

def main(par):

    # Parameters
    np.set_printoptions(precision=3)
   
    # Initialize environment land and sea environments and agents
    max_land, max_sea, land_arr, sea_arr = create_n_inisialise_landscapes(par)
    burners = initialize_agents(par.n_consumers, par.length)

    # Simulate multiple time steps
    land_fuel_levels = land_arr
    sea_fuel_levels = sea_arr 

    burned_land_memo = []
    burned_sea_memo = []

    burned_land_mat = np.zeros(par.length)
    burned_sea_mat = np.zeros(par.length)
    positions = burners
    radius_memo = np.zeros(par.n_consumers+1)
    
    print('\n N HFG/Burners:' , par.n_consumers, ' \n')
   
    for t in range(par.time):
        #print(f"t{t}:Agent Positions: {burners}")
        #print(f"land fuel {land_arr}:'\n sea fuel: {sea_arr}\n")
        burners, new_land_arr, new_sea_arr, Rs = shake_burners(par, burners, land_arr, sea_arr)            
        
        burned_land = sum(land_arr - new_land_arr)
        burned_sea = sum(sea_arr - new_sea_arr)

        burned_land_arr = land_arr - new_land_arr
        burned_sea_arr = sea_arr - new_sea_arr

        burned_land_memo = np.append(burned_land_memo, burned_land)
        burned_sea_memo = np.append(burned_sea_memo, burned_sea)

        burned_land_mat = np.vstack([burned_land_mat, burned_land_arr])
        burned_sea_mat = np.vstack([burned_sea_mat, burned_sea_arr])
        
        land_arr, sea_arr = fuels_evol(par, new_land_arr, new_sea_arr, max_land, max_sea)
        #print(f"\nt{t + 1}: Agent Positions: {burners}")
        #print(f"land fuel: {land_arr}'\n sea fuel: {sea_arr}\n")

        land_fuel_levels = np.vstack([land_fuel_levels, land_arr])
        sea_fuel_levels = np.vstack([sea_fuel_levels, sea_arr])
        positions = np.vstack([positions, burners])
        
        #print('radiusmeme', Rs, radius_memo, len(radius_memo), len(Rs))
        radius_memo = np.vstack([radius_memo, Rs])
        #radius = np.vstack([radius, Rs])


    norm_burned_land, norm_burned_sea = accumulated_burnings(par, burned_land_memo, burned_sea_memo)
    norm_movements = accumulated_n_jumps(par, positions)
    norm_radius = accumulated_mean_R(par, radius_memo)
    max_R = maximum_R(par, radius_memo)
    #print( f" norm land fuel: {norm_burned_land}'\n norm sea fuel: {norm_burned_sea}\n" )
    #print( f" all land fuel: {sum(burned_land_memo)}'\n all sea fuel: {sum(burned_sea_memo)}\n" )
    #print( f" all sea fuel: {all_sea}'\n all sea fuel: {all_sea}\n" )
    
    #pf.plot_aggregated_resources(par, burned_sea_memo, burned_land_memo,  'nom')
    #pf.plot3_1cell_resources(par, sea_fuel_levels, land_fuel_levels,  'nom')
    #pf.vector_movie(par, land_fuel_levels, sea_fuel_levels, movements,  'nom')

    return norm_burned_land, norm_burned_sea, norm_movements, max_R

par = cfg.par()
if __name__ == "__main__":
    main(par)
