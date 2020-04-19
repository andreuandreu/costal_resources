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


''' version 0.4 of the code of costal resources'''
'''
Forth instance of the code,
it adds several consumers to the vector. The jumping strategy is designed to avoid 
that two consumers fall in the same cell. If they do, the cells selected as possible
jump are expanded by decreasing the  percentage that a cell has to have when compared 
with the maximum of the cells within the radius, where the other consumer probably was.
'''

#

def land_vector(l):
    '''create land vector'''
    
    #land_vector= np.full(l, cnt.land_0 )#uniform initial conditions
    land_vector= np.random.uniform(low =  0, high = cnt.land_0, size=l ) #random initial conditions
    #print('t_o land', land_vector)
    return land_vector


class constants:
    length = 75 #length of the land area in cells
    time = 120 #time steps
   
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
    

def time_steps():
    
    t = np.arange(1,cnt.time, 0.5)
    
    return t

def which_jumpers(aux, L, prev_positions):
    '''return the consumers that need to change position'''
    jumped_to = prev_positions
    other_jumpers = prev_positions.tolist()

    if len(aux) > 0:
        for p, i in zip(prev_positions[aux], aux):
            #print ('shit im doing', l, i )
            jumped_into = vector_jump(L, p, other_jumpers)
            other_jumpers.append(jumped_into)
            jumped_to[i] = jumped_into
        return np.array(jumped_to)
    else:
        return np.array(prev_positions)




def vector_jump(L, p , other_jumpers):
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



def resorurces_evol(t, c, L, l):

    '''operates the vector of resourexit()ces'''
    resources = []
    sea_consumption = [] 
    production = []
    position = []
    resources.append(L)
    position.append(l)
    
    for e in t:
        print('time', e)

        p = cnt.land_productivity* (1 - L/cnt.max_land) *L
        L = L + p
        s = np.full( len(l), 0)

        # land consuption conumers
        ind_landc = np.where( L[l]  - c > cnt.L_threshold )
        L[l[ind_landc[0]]] =  L[l[ind_landc[0]]] - c

        #sea consumption consumers
        #ind_seac = np.where(  L[l[aux[0]]] - cnt.L_threshold + cnt.sea_productivity >= c  ) 
        ind_seac = (L[l]  - c <= cnt.L_threshold)  &  (L[l] - cnt.L_threshold + cnt.sea_productivity >= c ) 
        land_margin = L[l[ind_seac]] - cnt.L_threshold

        ufff = np.where(land_margin > cnt.consumption_rate)
        if ind_seac[0]:
            print ('ind sea', ind_seac) 
            print ('low resources sea', L[l[ind_seac]])
            print ('lllll', l[ind_seac])

            print ('land margin', land_margin, cnt.L_threshold)

        L[l[ind_seac]] = cnt.L_threshold

        s[ind_seac] = c  - land_margin

        #jumping consumers 
        aux =   np.where(L[l]  - c <= cnt.L_threshold)  #&  (L[l] - cnt.L_threshold + cnt.sea_productivity < c ) 
        l = which_jumpers(aux[0], L, l)
        position.append(l)
        #store values
        resources.append(L)
        
        sea_consumption.append(np.array(s))
        production.append(p)

    #for e in resources:
    #    print('RRRese', e, '\n')
    #for e in sea_consumption:
    #    print ('sea', e)
    #for e in position:
    #    print ('positt', e)

    return resources, sea_consumption, production, position
    

def plot_resources(resources, name):

    fig = plt.figure(name)
    ax = fig.add_subplot(111)
    ax.set_xlabel("t")
    ax.set_ylabel("Sea resources used")

    for r in np.array(resources).T:
        ax.plot(r)
    name_sea = '../plots_costal_resources/sea_resources_'+ name + '.eps'
    plt.savefig(name_sea,  bbox_inches = 'tight')


def plot_total_resources(resources, name):

    fig = plt.figure(name)
    ax = fig.add_subplot(111)
    ax.set_xlabel("t")
    ax.set_ylabel("total Sea resources used")

    plt.plot(sum(np.array(resources).T))
 
    name_sea = '../plots_costal_resources/total_sea_resources_'+ name + '.eps'
    plt.savefig(name_sea,  bbox_inches = 'tight')

    
    

def vector_movie(M, position, nom):
    fig, ax = plt.subplots()#111, 'matrix movie'
    A = np.rot90([M[0][::-1]])#
    print('pppppppp', position[0])
    ax.clear()
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=cnt.high_land)
    matrice = ax.matshow(A, cmap = cm.OrRd, norm = normalize)# origin="lower"
    jumper_plot_position = 1.2
    for s in   position[0]:
        ax.scatter(jumper_plot_position, s, marker = 'o', facecolors = 'k')#.plot(position[0])
    plt.colorbar(matrice)
    #plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    
    def update(i):
        #matrice.axes.clear()
        A = np.rot90([M[i][::-1]])
        #matrice.axes.clear()

        if i>0:
            #color = A[position[i-1][1], position[i-1][0]]
            #print( color, 'aaa', position[i-1], A )
            for s in position[i-1]:
                matrice.axes.scatter(jumper_plot_position, s,  marker = 'o', c = 'w', lw = 0, s=33)#
        matrice.set_array(A)
        print (i, 'iiii', len(position), len(M))
        for s in position[i]:
            matrice.axes.scatter(jumper_plot_position, s,  marker = 'o', facecolors = 'k', lw = 0, s=30)
        

    ani = animation.FuncAnimation(fig, update, frames=len(M), interval=80)#

    
    plt.show()
    name_gif = '../plots_costal_resources/matrix_land_' + nom+'.gif'
    ani.save(name_gif,  dpi = 80)#,writer = 'imagemagick')


    #ax01.set_title('$\\nu$ ' + str(v) + ' b ' + str(b))
    # set label names
    #ax.set_xlabel("m")
    #ax.set_ylabel("l")
    


cnt = constants()
t= time_steps()
Land = land_vector(cnt.length)
name = sys.argv[1]


resources, sea_consumption, production, position = resorurces_evol(t, cnt.consumption_rate, Land , cnt.walkers_array)
plot_resources(sea_consumption, name + '_individual')
plot_total_resources(sea_consumption, name+ '_total')

vector_movie(resources, position, name)

plt.show()













