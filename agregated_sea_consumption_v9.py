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
from random import choice
from random import randrange
from collections import deque
from scipy.ndimage.interpolation import shift
import matplotlib.animation as animation
import matplotlib
from subprocess import call
import matplotlib.ticker as ticker

import plots_functions as pf



import pickle
import cmath


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

#
class constants:
    length = 44#length of the land area in cells
    time = 433 #max time 
    t_step_lenth = 0.5 # lenth of temporal steps
   
    burning_rate = 1  #burning rate, this has to be 1 as all the rest will be normalized to these units.
    high_land = 5#30 # maximum amount for the maximum capacity land cell, in burning rate units. 
    high_sea = 8#2*high_land#high_land*0.67#4.5 #initial condition on land combustible, in burning rate units. 
    min_land = 0.01*burning_rate# minimum amount of land fuel always present but negligible

    land_productivity = 0.2# constant for the productivity of land, in burning rate units.  
    tidal_deluge = 0.4 #amount of recovery of sea fuel, multiplictive factor of intitial conditions 

    radius = 4 # maximum number of cells that the consumer can move in one jump
    n_consumers = 8# number of consumers
    positions = np.arange(length)

    burn_frames = 33 #do not use the first N frames to compute gloval burning of resources


def create_n_inisialise_landscapes(cnt):

    '''create land vector'''
    max_land = np.random.uniform(low = cnt.min_land , high = cnt.high_land, size=cnt.length)#5 #maximum of land combustible resources per cell, in burning rate units. 
    max_sea = np.random.uniform(low =  0.0, high = cnt.high_sea, size=cnt.length)
    
    #land_vector= np.full(l, cnt.high_land )#uniform initial conditions
    land_vector= np.random.uniform(low =  0, high = 1, size=cnt.length ) * max_land  #random initial conditions
    sea_vector= np.random.uniform(low =  0, high = 1, size=cnt.length ) * max_sea  #random initial conditions    
    
    return max_land, max_sea, land_vector, sea_vector 


def time_steps(cnt):
    '''temporal array to set the simulation steps'''
    t = np.arange(1, cnt.time, cnt.t_step_lenth)
    
    return t


def fuels_evol(cnt, land_arr, sea_arr, max_land, max_sea):

    '''operates the vector of fuels, replenishes fuel levels naturally after each time step
    
    Parameters:
    cnt (class): contains model parameters and intitial conditions
    land_arr (array): level of land fuel in each cell.
    sea_arr (float): level of sea cell in each cell.

    Returns:
    updated values of fuels
    '''

    ###land fuel
    increase_land_fuel = cnt.land_productivity* (1 - land_arr/max_land) * land_arr 
    new_land_arr = land_arr + increase_land_fuel + cnt.min_land
    
    ###sea fuel
    new_sea_arr = sea_arr.copy()
    for i, s in enumerate(sea_arr):
        accumulated = s + cnt.tidal_deluge
        if accumulated <= max_sea[i] - cnt.tidal_deluge:
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
        new_land = land/10.
        new_sea = sea - (burning - land + land/10.)
        consumed = sea-new_sea + land - new_land
        #print('consume sea and land', '{:2.2f}'.format(land), '{:.2f}'.format(new_land), '{:.2f}'.format(sea), '{:.2f}'.format(new_sea), consumed )
        return new_land, new_sea
    
    else:
        #print('consume leftover fuel', '{:2.2f}'.format(land), '{:2.2f}'.format(sea), 0 )
        return land, 0
    


def acumulated_burnings(cnt, land_fuel, sea_fuel):    
    '''computes the acumulated sea burning of all the consumers after a burndown period
    
    this is the mean sea condumtion by all the agents through all the time, after a burnout 
    period is remobed'''
    land_matrix = np.matrix(np.array(land_fuel[cnt.burn_frames:]))
    all_land = np.matrix.sum(land_matrix)/len(land_fuel[cnt.burn_frames:])/cnt.n_consumers

    sea_matrix = np.matrix(np.array(sea_fuel[cnt.burn_frames:]))
    all_sea = np.matrix.sum(sea_matrix)/len(sea_fuel[cnt.burn_frames:])/cnt.n_consumers
    #print ('all the sea resources ', all_sea)

    return all_land, all_sea

def acumulated_n_jumps(cnt, n_jumps):    
    '''computes the acumulated jumps per consumer after a burndown period
    
    this is the averaged by all the agents through all the time, after a burnout 
    period is remobed'''

    jump_matrrix = np.matrix(np.array(n_jumps[cnt.burn_frames:]))
    print ('all the jumps resources ', np.matrix.sum(jump_matrrix), len(jump_matrrix[cnt.burn_frames:]))
    return np.matrix.sum(jump_matrrix)

    import numpy as np
import random



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


def shake_burners(burners, land_arr, sea_arr, cnt):

    """
    Determines the action to take among 2 options for the list of burners and updates the fuel arrays:
    - consume fuels
    - move and consume 

    Parameters:
    land_arr (numpy.ndarray): Array of values.
    sea_arr (numpy.ndarray): Array of values.
    burners (list): Current positions of agents.
    cnt (int): class containing parameters of the simulation

    Returns:
    list: Updated fuel levels.
    """

    new_land_arr = np.copy(land_arr)
    new_sea_arr = np.copy(sea_arr)

    for i, pos in enumerate(burners):

        if new_land_arr[pos] + new_sea_arr[pos] < cnt.burning_rate:
            burners = move_burner(i, pos, new_land_arr, burners, cnt.radius)
            
        new_land_arr[pos], new_sea_arr[pos] = consume(cnt.burning_rate, new_land_arr[pos], new_sea_arr[pos] )

    return burners, new_land_arr, new_sea_arr


def move_burner(i, burner_pos, land_arr, burners, R):
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
            #print ('position', pos, ' buner number ', i )
            #max_value = land_arr[pos]
            best_position = pos

    # Update the agent's position
    burners[i] = best_position

    return burners

def main(cnt):
    # Parameters
    np.set_printoptions(precision=3)

    # Initialize environment land and sea environments and agents
    max_land, max_sea, land_arr, sea_arr = create_n_inisialise_landscapes(cnt)
    burners = initialize_agents(cnt.n_consumers, cnt.length)

    # Simulate multiple time steps
    land_fuel_levels = land_arr
    sea_fuel_levels = sea_arr 

    burned_land_memo = []
    burned_sea_memo = []

    burned_land_mat = np.zeros(cnt.length)
    burned_sea_mat = np.zeros(cnt.length)
    movements = burners
    
    print('N BURNERS!!!' , cnt.n_consumers)
   
    for t in range(cnt.time):
        #print(f"t{t}:Agent Positions: {burners}")
        #print(f"land fuel {land_arr}:'\n sea fuel: {sea_arr}\n")
        burners, new_land_arr, new_sea_arr = shake_burners(burners, land_arr, sea_arr, cnt)            
        
        burned_land = sum(land_arr - new_land_arr)
        burned_sea = sum(sea_arr - new_sea_arr)

        burned_land_arr = land_arr - new_land_arr
        burned_sea_arr = sea_arr - new_sea_arr

        burned_land_memo = np.append(burned_land_memo, burned_land)
        burned_sea_memo = np.append(burned_sea_memo, burned_sea)

        burned_land_mat = np.vstack([burned_land_mat, burned_land_arr])
        burned_sea_mat = np.vstack([burned_sea_mat, burned_sea_arr])
        
        
        land_arr, sea_arr = fuels_evol(cnt, new_land_arr, new_sea_arr, max_land, max_sea)
        #print(f"\nt{t + 1}: Agent Positions: {burners}")
        #print(f"land fuel: {land_arr}'\n sea fuel: {sea_arr}\n")

        land_fuel_levels = np.vstack([land_fuel_levels, land_arr])
        sea_fuel_levels = np.vstack([sea_fuel_levels, sea_arr])
        movements = np.vstack([movements, burners])


    norm_burned_land, norm_burned_sea = acumulated_burnings(cnt, burned_land_memo, burned_sea_memo)

    #print( f" norm land fuel: {norm_burned_land}'\n norm sea fuel: {norm_burned_sea}\n" )
    #print( f" all land fuel: {sum(burned_land_memo)}'\n all sea fuel: {sum(burned_sea_memo)}\n" )
    #print( f" all sea fuel: {all_sea}'\n all sea fuel: {all_sea}\n" )
    #return sum(burned_land_memo), sum(burned_sea_memo)

    #pf.plot_resources(cnt, burned_sea_memo, burned_land_memo,  'nom')
    #pf.plot3_1cell_resources(cnt, sea_fuel_levels, land_fuel_levels,  'nom')
    #pf.vector_movie(cnt, land_fuel_levels, sea_fuel_levels, movements,  'nom')
    return norm_burned_land, norm_burned_sea, movements



if __name__ == "__main__":
    cnt = constants()
    main(cnt)
