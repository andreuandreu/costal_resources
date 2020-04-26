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
from collections import deque
from scipy.ndimage.interpolation import shift
import matplotlib.animation as animation
import matplotlib
from subprocess import call
import matplotlib.ticker as ticker



import pickle
import cmath


''' version 0.5 of the code of costal resources'''
'''
Fift instance of the code
"sea-consumption-grid.py" is to call this pacage.
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
    length = 75 #length of the land area in cells
    time = 120 #max time 
    t_step_lenth = 0.5 # lenth of temporal steps
   
    land_productivity = 0.2 #rate of increase of combustible in a land cell per unit of time
    high_land = 30 # maximum amount for the maximum capacity land cell
    land_0 = high_land*0.67#4.5 #initial condition on land combustible.
    max_land = np.random.uniform(low =  1.9, high = high_land, size=length)#5 #maximum of land combustible resources per cell
    L_threshold = 0.4 #threshold below which the consumer does not consume more resources from that cell
    
    sea_productivity = 0.9 #amount of maximum combustible avaliable on the sea in form of algae by unit of time 
    
    consumption_rate = 1 #rate of consumtion, this has to be set to 1, everything is normalized by this
    radius = 7 # maximum number of cells that the consumer can move in one jump
    margin = 0.2  #margin of combustible abaliabbility from the cell with maximum combustible where the consumer can move into
    margin_margin = 0.1 #if two jumpers fall in the same cell, increase the margin of possible other cells where the jumper can go
    n_consumers = 18 # number of consumers
    walkers_array = np.random.randint(low =  0, high = length, size=n_consumers)#initial positions of the consumers
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
            other_jumpers.append(jumped_into)
            jumped_to[i] = jumped_into
        return np.array(jumped_to)
    else:
        return np.array(prev_positions)




def vector_jump(cnt, L, p , other_jumpers):
    '''jumping strategy, within an area defined by a distance r, returns the cell with the most resources'''
    
    #ind_jump = np.where(l > cnt.radius -1 and l + cnt.radius < cnt.length)
    #print(l, 'ind_jump', ind_jump[0], ind_jump)
        
    increase_margin = 0    
    if p > cnt.radius and p<= cnt.length - cnt.radius:#len(aux[0]) > 1:
        max_p = max(L[p-cnt.radius:p+cnt.radius+1])
        new_margin = cnt.margin  +  increase_margin
        aux = np.where(L[p-cnt.radius:p+cnt.radius+1] > max_p - max_p*new_margin  )
        if len(aux[0]) > 1:
            select = choice(aux[0])
        else:
            select = aux[0][0]

        chosen = p - cnt.radius + select 
        while (chosen in other_jumpers):
            #this keeps jumping until the selected cell is not concident with a cell chosen by another jumper
            new_margin = cnt.margin  +  increase_margin
            aux = np.where(L[p-cnt.radius:p+cnt.radius+1] > max_p - max_p*new_margin  )#or L = L[l]
            select = choice(aux[0])
            chosen = p - cnt.radius + select 
            increase_margin = increase_margin + cnt.margin_margin          
    
        return chosen 

    elif p <= cnt.radius:
        max_p = max(L[0:p+cnt.radius])
        new_margin = cnt.margin  +  increase_margin
        aux = np.where(L[0:p+cnt.radius] > max_p - max_p*new_margin)
        #aux = l + randint(0, cnt.radius)
        if len(aux[0]) > 1:
            select = choice(aux[0])
        else:
            select = aux[0][0]

        while (select in other_jumpers):
            #this keeps jumping until the selected cell is not concident with a cell chosen by another jumper
            new_margin = cnt.margin  +  increase_margin
            aux = np.where(L[0:p+cnt.radius+1] > max_p - max_p*new_margin  )#or L = L[l]
            select = choice(aux[0])
            increase_margin = increase_margin + cnt.margin_margin 
        return select 

    else:
        max_p = max(L[p-cnt.radius:cnt.length-1])
        new_margin = cnt.margin  +  increase_margin
        aux = np.where(L[p-cnt.radius:cnt.length-1] > max_p - max_p*new_margin)
        #aux = p + randint(0, cnt.radius)
        if len(aux[0]) > 1:
            select = choice(aux[0])
        else:
            select = aux[0][0]
        chosen = p - cnt.radius + select
        while (chosen in other_jumpers):
            #this keeps jumping until the selected cell is not concident with a cell chosen by another jumper
            new_margin = cnt.margin  +  increase_margin
            aux = np.where(L[p-cnt.radius:cnt.length-1] > max_p - max_p*new_margin  )#or L = L[p]
            select = choice(aux[0])
            chosen = p - cnt.radius + select 
            increase_margin = increase_margin + cnt.margin_margin 
        
        return chosen
        #print ('upper section', aux[0], 'ssss', select)



def resorurces_evol(cnt, t, L, c, l):

    '''operates the vector of resources'''
 
    #resouces = []
    #production = []
    #position = []
    #resources.append(L)
    #position.append(l)

    sea_consumption = [] 
    
    for e in t:
        print('time', e)

        p = cnt.land_productivity* (1 - L/cnt.max_land) *L
        L = L + p
        s = np.full( len(l), 0)

        ###land consuption conumers
        ind_landc = np.where( L[l]  - c > cnt.L_threshold )
        L[l[ind_landc[0]]] =  L[l[ind_landc[0]]] - c

        ###sea consumption consumers
        #ind_seac = np.where(  L[l[aux[0]]] - cnt.L_threshold + cnt.sea_productivity >= c  ) 
        ind_seac = (L[l]  - c <= cnt.L_threshold)  &  (L[l] - cnt.L_threshold + cnt.sea_productivity >= c ) 
        land_margin = L[l[ind_seac]] - cnt.L_threshold

        

        L[l[ind_seac]] = cnt.L_threshold

        s[ind_seac] = c  - land_margin

        ###jumping consumers 
        aux =   np.where(L[l]  - c <= cnt.L_threshold)  #&  (L[l] - cnt.L_threshold + cnt.sea_productivity < c ) 
        l = which_jumpers(cnt, aux[0], L, l)
        
        ###store values
        #resources.append(L)
        #position.append(l)
        #production.append(p)
        sea_consumption.append(np.array(s))

    
    return  sea_consumption
    


def acumulated_sea_consumption(cnt, sea_resources):    
    '''computes the acumulated sea consumption of all the consumers after a burndown period''''

    sea_matrrix = np.matrix(sea_resources[cnt.burn_frames:])
    print ('all the sea resources ', np.matrix.sum(sea_matrrix))
    return np.matrix.sum(sea_matrrix)
    

