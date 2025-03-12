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



import pickle
import cmath


''' version 0.7 of the code of costal resources'''
'''

Seventh instance of the code
what is new from version is teh removal of the land threshol to be substiuted
by a simpler jumping and sea consumption whrn the Land resources are < Consumption
"sea-consumption-grid_max-land.py" is to call this pacage.
streamlined version to compute the total of sea resources consumed given some input parameters
that where taken as a constant in the previous verions

it would require the imput parameters:
land_productivity
number_of_consumers

it would produce the output:
sum_of_sea_consumption

'''

#


class constants:
    length = 40 #length of the land area in cells
    time = 222 #max time 
    t_step_lenth = 0.5 # lenth of temporal steps
   
    #land_productivity = 0.2 #rate of increase of combustible in a land cell per unit of time
    consumption_rate = 1  #consumption rate, this has to be 1 as all the rest will be normalized to these units.
    #high_land = 30 # maximum amount for the maximum capacity land cell, in consumption rate units. 
    land_0 = 5#high_land*0.67#4.5 #initial condition on land combustible, in consumption rate units. 
    #max_land = np.random.uniform(low =  1.9, high = high_land, size=length)#5 #maximum of land combustible resources per cell, in consumption rate units. 
    land_productivity = 0.2# constant for the productivity of land, in consumption rate units.  
    
    sea_productivity = 0.9 #amount of maximum combustible avaliable on the sea in form of algae by unit of time, in consumption rate units. 
    
    radius = 4 # maximum number of cells that the consumer can move in one jump
    margin = 0.2  #margin of combustible abaliabbility from the cell with maximum combustible where the consumer can move into, in consumption rate units. 
    margin_margin = 0.1 #if two jumpers fall in the same cell, increase the margin of possible other cells where the jumper can go, in consumption rate units. 
    elegivility = 0.5 #determines the spread, sigmam of a gausaian distribution to select how likely a cell is to be chosen depending how many less resources it has than teh maximum
    #n_consumers = 18 # number of consumers
    positions = np.arange(length)

    burn_frames = 50 #do not use the first N frames to compute gloval consumption of resources


def land_vector(cnt, l):
    '''create land vector'''
    
    #land_vector= np.full(l, cnt.land_0 )#uniform initial conditions
    land_vector= np.random.uniform(low =  0, high = cnt.land_0, size=l ) #random initial conditions
    #print('t_o land', land_vector)
    return land_vector

    

def time_steps(cnt):
    '''temporal array to set the simulation steps'''
    t = np.arange(1, cnt.time, cnt.t_step_lenth)
    
    return t

def which_jumpers(cnt, aux, L, prev_positions):
    '''return the consumers that need to change position'''

    jumped_to = prev_positions
    other_jumpers = prev_positions.tolist()

    if len(aux) > 0:
        for p, i in zip(prev_positions[aux], aux):
            #print ('shit im doing', l, i )
            jumped_into = vector_jump(cnt, L, p, other_jumpers)
            #jumped_into = random_walk_jump(cnt, L, p , other_jumpers)
            other_jumpers.append(jumped_into)
            jumped_to[i] = jumped_into
        return np.array(jumped_to)
    else:
        return np.array(prev_positions)

def random_walk_jump(cnt, L, p , other_jumpers):
    '''jumping strategy, based on random walk, returns the cell with the most resources'''

    upordown = choice([-1,1]) 
    select = p + upordown
    counter = 0
   
    print('ssss1', select, p)

    try: 
        while (L[select] < L[p] + cnt.consumption_rate or select in other_jumpers) :  
            select= select + upordown
            counter = counter + 1
            print ('isssss thisssss less fuckkkkkkk>__+:?>', select, counter) 
        return select
    except:
        upordown = -upordown
        print('ssss2', select, p)
        while (L[select] < L[p] + cnt.consumption_rate or select in other_jumpers ):
            counter = counter + 1
            print ('isssss thisssss fuckkkkkkk>__+:?>', select, counter) 
            if counter > cnt.length/2.:
                print('is FUKEEED!')
                raise
            select= select + upordown
        return select
       
    # y_labels = y[::step_y]

def gaussian_jump(cnt, L, p , other_jumpers):
    '''jumping strategy, based on a probability to jump to a certain empty cell, depending on how many resources there are'''

    if p > cnt.radius and p<= cnt.length - cnt.radius:
        limits = [p-cnt.radius, p+cnt.radius+1]
    elif p <= cnt.radius:
        limits = [0, p+cnt.radius]
    else:
        limits = [p-cnt.radius, cnt.length-1]

    max_p = max(L[limits[0]:limits[1]]) 
     
    dist = distnp.random.normal(max_p, elegivility, 1)

    max_p = max(L[p-cnt.radius:p+cnt.radius+1])


def vector_jump(cnt, L, position , other_jumpers):
    '''jumping strategy, within an area defined by a distance r, returns the unocupied cell with the most resources'''
    
    #ind_jump = np.where(l > cnt.radius -1 and l + cnt.radius < cnt.length)
    #print(l, 'ind_jump', ind_jump[0], ind_jump)
        
    increase_margin = 0    
    if position > cnt.radius and position<= cnt.length - cnt.radius:#len(aux[0]) > 1:
        max_p = max(L[position-cnt.radius:position+cnt.radius+1])
        new_margin = cnt.margin  +  increase_margin
        aux = np.where(L[position-cnt.radius:position+cnt.radius+1] > max_p - max_p*new_margin  )
        if len(aux[0]) > 1:
            select = choice(aux[0])
        else:
            select = aux[0][0]

        chosen = position - cnt.radius + select 
        j = 0
        while (chosen in other_jumpers and j < 6):
            j = j+1
            #if j %10 ==1:
            #    print ('jjjjjM',j)
            #this keeps jumping until the selected cell is not concident with a cell chosen by another jumper
            new_margin = cnt.margin  +  increase_margin
            aux = np.where(L[position-cnt.radius:position+cnt.radius+1] > max_p - max_p*new_margin  )#or L = L[l]
            select = choice(aux[0])
            chosen = position - cnt.radius + select 
            increase_margin = increase_margin + cnt.margin          
    
        if j > 4:
            return randrange(cnt.radius,  cnt.length-cnt.radius)
        else:
            return chosen

    elif p <= cnt.radius:
        max_p = max(L[0:position+cnt.radius])
        new_margin = cnt.margin  +  increase_margin
        aux = np.where(L[0:position+cnt.radius] > max_p - max_p*new_margin)
        #aux = l + randint(0, cnt.radius)
        if len(aux[0]) > 1:
            select = choice(aux[0])
        else:
            select = aux[0][0]

        j = 0
        while (select in other_jumpers and j < 6):
            j = j+1
            #if j %10 ==1:
               # print ('jjjjjL',j)
            #this keeps jumping until the selected cell is not concident with a cell chosen by another jumper
            new_margin = cnt.margin  +  increase_margin
            aux = np.where(L[0:position+cnt.radius+1] > max_p - max_p*new_margin  )#or L = L[l]
            select = choice(aux[0])
            increase_margin = increase_margin + cnt.margin 
        if j > 4:
            return select + cnt.radius*2
        else:
            return select 

    else:
        
        max_p = max(L[position-cnt.radius:cnt.length-1])
        new_margin = cnt.margin  +  increase_margin
        aux = np.where(L[position-cnt.radius:cnt.length-1] > max_p - max_p*new_margin)
        
        print('out of othernessss!!!', aux, max_p)
        if len(aux[0]) > 1:
            select = choice(aux[0])
        else:
            select = aux[0][0]
        chosen = position - cnt.radius + select
        j = 0
        while (chosen in other_jumpers and j< 6):
            j = j+1
            #if j %10 ==1:
            #    print ('jjjjjH',j)
            #this keeps jumping until the selected cell is not concident with a cell chosen by another jumper
            new_margin = cnt.margin  +  increase_margin
            aux = np.where(L[p-cnt.radius:cnt.length-1] > max_p - max_p*new_margin  )#or L = L[p]
            select = choice(aux[0])
            chosen = position - cnt.radius + select 
            increase_margin = increase_margin + cnt.margin 
        
        if j > 4:
            return chosen - cnt.radius*2
        else:
            return chosen
        #print ('upper section', aux[0], 'ssss', select)



def resorurces_evol(cnt, t, L, max_land, consumers_array):

    '''operates the vector of resources'''
 
    #resouces = []
    #production = []
    #position = []
    #resources.append(L)
    #position.append(l)

    sea_consumption = [] 
    n_jumps_array = [] 
    
    for e in t:

        p = cnt.land_productivity* (1 - L/max_land) *L
        L = L + p
        s = np.full( len(consumers_array), 0.0) #be mindfull, it has to be float!!!! 0.0

        ###land consuption conumers
        ind_landc = np.where( L[consumers_array] > cnt.consumption_rate )
        L[consumers_array[ind_landc[0]]] =  L[consumers_array[ind_landc[0]]] \
            - cnt.consumption_rate

        ###sea and land consumption consumers
        mask = (L[consumers_array]  <= cnt.consumption_rate) # (L[consumers_array] - cnt.consumption_rate + cnt.sea_productivity >= 0 ) 
        ind_seac = np.where(mask)[0]
        
        land_margin = np.array(L[consumers_array[ind_seac]]) - cnt.consumption_rate
        L[consumers_array[ind_seac]] = cnt.consumption_rate
        s[ind_seac] = cnt.sea_productivity

        ###jumping consumers 
        aux =   np.where(L[consumers_array]   <= cnt.consumption_rate)  #&  (L[l] - cnt.L_threshold + cnt.sea_productivity < c ) 
        consumers_array = which_jumpers(cnt, aux[0], L, consumers_array)
        n_jumps = len(aux[0])

        ###store values
        #resources.append(L)
        #position.append(l)
        #production.append(p)
        sea_consumption.append(np.array(s/len(consumers_array)))#/(len(consumers_array)*(cnt.time))
        n_jumps_array.append(n_jumps/len(consumers_array))
    
    return  sea_consumption#, n_jumps_array
    


def acumulated_sea_consumption(cnt, sea_resources):    
    '''computes the acumulated sea consumption of all the consumers after a burndown period
    
    this is the mean sea condumtion by all the agents through all the time, after a burnout 
    period is remobed'''

    sea_matrrix = np.matrix(np.array(sea_resources[cnt.burn_frames:]))
    print ('all the sea resources ', np.matrix.sum(sea_matrrix), len(sea_resources[cnt.burn_frames:]))
    return np.matrix.sum(sea_matrrix)/len(sea_resources[cnt.burn_frames:])

def acumulated_n_jumps(cnt, n_jumps):    
    '''computes the acumulated jumps per consumer after a burndown period
    
    this is the averaged by all the agents through all the time, after a burnout 
    period is remobed'''

    jump_matrrix = np.matrix(np.array(n_jumps[cnt.burn_frames:]))
    print ('all the jumps resources ', np.matrix.sum(jump_matrrix), len(jump_matrrix[cnt.burn_frames:]))
    return np.matrix.sum(jump_matrrix)
    

