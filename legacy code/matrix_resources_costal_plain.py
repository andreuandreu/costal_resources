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
from collections import deque
from scipy.ndimage.interpolation import shift
import matplotlib.animation as animation
import matplotlib
from subprocess import call
import matplotlib.ticker as ticker

import config as cfg

import pickle
import cmath


'''
extends the previous consumption and production of resources to a NxM matrix 
and plots the result in a dianamic movie showing the matrix as cells
and plots a graph of all the sea resources consumed.
'''
#>python matrix_resources_costal_plain.py name_plot


def land_grid(m,l, maxL=5):
    '''create grid'''
    
    land_matrix = np.full((m, l), maxL)#np.random.rand(6).reshape(m,l)*var.land_0
    
    print('t_o land', land_matrix)
    return land_matrix

def time_steps(par):
    
    t = np.arange(1, par.time, 0.5)
    
    return t

def jump(L):

    '''jumping strategy, returns the cell with the most resources'''

    aux = np.where(L == max(L))

    return aux[0][0], aux[0][1]

def area_jump( par, L, m, l, d):
    '''jumping strategy, within an area defined by a distance d, returns the cell with the most resources'''

    low_lim = m-d
    if low_lim <  0:
        low_lim = 0

    high_lim = m+d
    if high_lim >  par.length-1:
        high_lim = par.length-1
        
  
    area = L[low_lim:high_lim][:]
    aux = np.where(area == max(area))

    return aux[0][0], aux[0][1]



def resorurces_evol(var, par, t, L, m=0, l=0):

    '''operates the matrix of resources'''
    resources = []
    sea_consumption = []
    production = []
    position = []
    resources.append(L)
    position.append([m,l])

    c = par.consumption_rate
    
    for e in t:

        p = var.land_productivity* (1 - L/var.max_land) *L
        L = L + p

        if L[m][l] - c > par.L_threshold:
           
            L[m][l] =  L[m][l] - c
            s = 0
            print('land consumption')        
        
        else:  
            if m == 0 and L[m][l] - par.L_threshold + var.sea_productivity > c+0.2:       
                
                land_margin = L[m][l] - par.L_threshold
                L[m][l] = par.L_threshold 

                s = c  - land_margin
                print('sea/land consumption')
                
                
            else:
                m = randint(0,par.width-1)
                l = randint(0,par.length-1)
                s = -1
                print ('jump to another square', m, l)

        resources.append(L)
        position.append([m,l])
        sea_consumption.append(s)
        production.append(p)


    for e in resources:
        print('eee', e, '\n', '\n')
    return resources, sea_consumption, production, position
    

def plot_resources(resources, name):

    fig = plt.figure(name)
    ax = fig.add_subplot(111)
    ax.set_xlabel("t")
    ax.set_ylabel("Sea resources used")

    ax.plot(resources)
    name_sea = '../plots_costal_resources/sea_resources_'+ name + '.eps'
    plt.savefig(name_sea,  bbox_inches = 'tight')

def plot_matrix(matrix, name):

    
    fig = plt.figure(name)
    ax = fig.gca() #fig.add_subplot(111)
    M = ax.matshow(matrix.T, interpolation='nearest', cmap=cm.OrRd)#,
    
    plt.colorbar(M, ax=ax)
    

def wtf(M):

    resources = []
    resources.append(M)
    
    for i in range(3):

        p = 2*M
        M = M +p #np.random.rand(6).reshape(2,3)
        
        print('rmatrix', M )
        print('resuurces', resources)
        resources.append(M)
 
    return resources

def matrix_movie(par, var, M, position, nom):
    fig, ax = plt.subplots()#111, 'matrix movie'
    A = M[0].T

    print("Initial values of Landscape vector ", A)
    ax.clear()
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=var.max_land)
    matrice = ax.matshow(A, cmap = cm.OrRd, norm = normalize)
    ax.scatter(position[0][0], position[0][1], marker = 'o', facecolors = 'k')#.plot(position[0])
    plt.colorbar(matrice)
    #plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

    
    def update(i):
        #matrice.axes.clear()
        A = M[i].T
        #matrice.axes.clear()

        if i>0:
            #color = A[position[i-1][1], position[i-1][0]]
            #print( color, 'aaa', position[i-1], A )
            matrice.axes.scatter(position[i-1][0], position[i-1][1], marker = 'o', c = 'w', lw = 0,  s=33)#
        matrice.set_array(A)
        print (i, 'iiii', len(position), len(M))
        matrice.axes.scatter(position[i][0], position[i][1], marker = 'o', facecolors = 'k', lw = 0, s=30)
        

    ani = animation.FuncAnimation(fig, update, frames=len(M), interval=630)#

    
    plt.show()
    name_gif = '../' + par.plots_dir + 'matrix_land_' + nom+'.gif'
    ani.save(name_gif,  dpi = 80)#,writer = 'imagemagick')


    #ax01.set_title('$\\nu$ ' + str(v) + ' b ' + str(b))
    # set label names
    #ax.set_xlabel("m")
    #ax.set_ylabel("l")
    
def main():

    var = cfg.var()
    par = cfg.par()
    t= time_steps(par)
    Land = land_grid(var.width, var.length, var.max_land)
    name = sys.argv[1]
    #wtf(Land)
    #example_matrxani()
    resources, sea_consumption, production, position = resorurces_evol(var, par, t, Land , 0, 0)
    plot_resources(sea_consumption, 'sea_consumption')
    #plot_matrix(resources[0], 'land_resources')

    matrix_movie(var, resources, position, name)

    plt.show()

if __name__ == "__main__":
    main()














