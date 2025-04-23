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

'''
Third instance of the code, 
modifies the previous code to only deal with a vector of consumption instead of a matrix
simplifyies the sea/land divided consumption of the previous "matrix" version.
It also uses a different rutine of choosing where the the consumer will jump, 
previously was random, now it selects the cells within a radius r around itself that have up to
a 80% of the cell containing the maximum of resources, and jumps randomly to any of these cells. 
It also modifies the ploting to only show a vector and consumers moving next to the cell they consumed
'''
#



def land_vector(l):
    '''create land vector'''
    
    land_vector= np.full(l, cnt.land_0 )#cnt.max_land
    
    #print('t_o land', land_vector)
    return land_vector


class constants:

    land_0 = 1.1 #initial condition on land combustible.
    land_productivity = 0.2 #rate of increase of combustible in a land cell per unit of time
    max_land = 5 #maximum of land combustible resources per cell
    sea_productivity = 0.9 #amount of maximum combustible avaliable on the sea in form of algae by unit of time 
    consumption_rate = 1 #rate of consumtion, this has to be set to 1, everything is normalized by this
    L_threshold = 0.4 #threshold below which the consumer does not consume more resources from that cell
    time = 40 #time steps
    length = 30 #length of the land area in cells
    radius = 3 # maximum number of cells that the consumer can move in one jump
    margin = 0.2  #margin of combustible abaliabbility from the cell with maximum combustible where the consumer can move into


def time_steps():
    
    t = np.arange(1,cnt.time, 0.5)
    
    return t


def vector_jump(L, l):
    '''jumping strategy, within an area defined by a distance r, returns the cell with the most resources'''


    if  l > cnt.radius -1 and l + cnt.radius < cnt.length:
        max_l = max(L[l-cnt.radius:l+cnt.radius+1])
        aux = np.where(L[l-cnt.radius:l+cnt.radius+1] > max_l - max_l*cnt.margin  )#or L = L[l]
        if len(aux[0]) > 1:
            select = choice(aux[0])
            #upordown = choice([-1,1]) 
            #index =  aux[0] +l - cnt.radius #int()+upordown
            #select = l - cnt.radius + aux[0][index]
            
            #print('auxxxuxuxuux', aux[0])
            #print('len',len(aux[0]), 'lll', l, '-+', upordown,'in', index , 'select', select, 'position', select +l - cnt.radius )
        else:
            select = aux[0]
            print ('lll', l, 'aux', aux)
        return select +l - cnt.radius #l + rand

    elif l <= cnt.radius:
        max_l = max(L[0:l+cnt.radius])
        aux = np.where(L[0:l+cnt.radius] > max_l - max_l*cnt.margin)
        #aux = l + randint(0, cnt.radius)
        if len(aux[0]) > 1:
            upordown = choice([-1,1])
            print ('juuuumpa', int((len(aux[0])-1)/2)+upordown)
            select = aux[0][int((len(aux[0])-1)/2)+upordown]
        else:
            select = aux[0]
        return select 

    else:
        max_l = max(L[l-cnt.radius:cnt.length-1])
        aux = np.where(L[l-cnt.radius:cnt.length-1] > max_l - max_l*cnt.margin)
        #aux = l + randint(0, cnt.radius)
        if len(aux[0]) > 1:
            upordown = choice([-1,1])
            select = aux[0][int((len(aux[0])-1)/2)+upordown]
        else:
            select = aux[0]
        return select +l - cnt.radius



def resorurces_evol(t, c, L, l=0):

    '''operates the vector of resources'''
    resources = []
    sea_consumption = []
    production = []
    position = []
    resources.append(L)
    position.append(l)
    
    for e in t:

        p = cnt.land_productivity* (1 - L/cnt.max_land) *L
        L = L + p
        if L[l] - c > cnt.L_threshold:
           
            L[l] =  L[l] - c
            s = 0     
        
        else:  
            if  L[l] - cnt.L_threshold + cnt.sea_productivity > c+0.2:       
                
                land_margin = L[l] - cnt.L_threshold
                L[l] = cnt.L_threshold 

                s = c  - land_margin
                print('sea/land consumption')
                
                
            else:
                l = vector_jump(L, l)#randint(0,cnt.length-1)
                s = -1
                #print ('jump to another square', l)


        resources.append(L)
        position.append(l)
        sea_consumption.append(s)
        production.append(p)

    #for e in resources:
     #   print('eee', e, '\n', '\n')
    return resources, sea_consumption, production, position
    

def plot_resources(resources, name):

    fig = plt.figure(name)
    ax = fig.add_subplot(111)
    ax.set_xlabel("t")
    ax.set_ylabel("Sea resources used")

    aux = np.where(np.array(resources) < 0)
    for i in aux[0]:
        resources[i] = 0
    ax.plot(resources)
    name_sea = '../plots_costal_resources/sea_resources_'+ name + '.eps'
    plt.savefig(name_sea,  bbox_inches = 'tight')

def plot_matrix(matrix, name):

    
    fig = plt.figure(name)
    ax = fig.gca() #fig.add_subplot(111)
    M = ax.matshow(matrix.T, interpolation='nearest', cmap=cm.OrRd)#,
    
    plt.colorbar(M, ax=ax)
    
    

def vector_movie(M, position, nom):
    fig, ax = plt.subplots()#111, 'matrix movie'
    A = np.rot90([M[0][::-1]])#
    print('pppppppp', position[0])
    ax.clear()
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=cnt.max_land)
    matrice = ax.matshow(A, cmap = cm.OrRd, norm = normalize)# origin="lower"

    ax.scatter(0.7,position[0], marker = 'o', facecolors = 'k')#.plot(position[0])
    plt.colorbar(matrice)
    #plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'



    
    def update(i):
        #matrice.axes.clear()
        A = np.rot90([M[i][::-1]])
        #matrice.axes.clear()

        if i>0:
            #color = A[position[i-1][1], position[i-1][0]]
            #print( color, 'aaa', position[i-1], A )
            matrice.axes.scatter(0.7, position[i-1],  marker = 'o', c = 'w', lw = 0, s=33)#
        matrice.set_array(A)
        print (i, 'iiii', len(position), len(M))
        matrice.axes.scatter(0.7,position[i],  marker = 'o', facecolors = 'k', lw = 0, s=30)
    
    ax.axis('off')

    ani = animation.FuncAnimation(fig, update, frames=len(M), interval=230)#

    
    plt.show()
    name_gif = 'matrix_land_' + nom+'.gif'
    ani.save(name_gif,  dpi = 80, writer = 'imagemagick')


    #ax01.set_title('$\\nu$ ' + str(v) + ' b ' + str(b))
    # set label names
    #ax.set_xlabel("m")
    #ax.set_ylabel("l")
    


cnt = constants()
t= time_steps()
Land = land_vector(cnt.length)
name = sys.argv[1]
#wtf(Land)
#example_matrxani()
resources, sea_consumption, production, position = resorurces_evol(t, cnt.consumption_rate, Land , int(cnt.length/2-0.5))
plot_resources(sea_consumption, 'sea_consumption')
#plot_matrix(resources[0], 'land_resources')

vector_movie(resources, position, name)

plt.show()













