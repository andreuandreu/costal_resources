# -*- coding: utf-8 -*-

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
import scripts.agregated_sea_consumption_v9 as mc



import pickle
import cmath


''' version 0.1 of the code that explores under which parameter space the consumers need to consume algae in a costal desert'''
'''
the code runs trought a set range of parameter spaces and computes the amount of sea resources consumed after X time steps
returns an hisotgram of searesource consumed when varing several combinations of parameters
start with land-production vs number of consumers.
'''
#python sea-consumption-grid.py name_plot

class limits:
    min_consumers = 2
    max_consumers = 44
    con_step = 1

    min_land_prod = 0.1
    max_land_prod = 0.3
    prod_step = 0.03

def generate_grid(lim):

    consumers_parameter = np.arange(lim.min_consumers, lim.max_consumers, lim.con_step )
    productivity_parameter = np.arange(lim.min_land_prod, lim.max_land_prod, lim.prod_step)

    return consumers_parameter, productivity_parameter

def call_model( consumers_number, land_productivity ):
    sim = mc.constants()
    t= mc.time_steps(cnt)
    Land = mc.land_vector(cnt, sim.length)

    consumers_array = np.random.randint(low =  0, high = cnt.length, size = consumers_number)#initial positions of the consumers
 
    sea_consumption = mc.resorurces_evol(cnt, t, Land, land_productivity, consumers_array)
    all_sea_consumption = mc.acumulated_sea_consumption(cnt, sea_consumption)
    
    return all_sea_consumption

def run_the_grid( consumers_parameter, productivity_parameter):

    sea_consumption_matrix  = np.zeros( (len(consumers_parameter), len(productivity_parameter) ) )

    for i, c in enumerate(consumers_parameter):
        
        for j, p in enumerate(productivity_parameter):
            print ('we are in', i, j)
            s = call_model(c, p)
            sea_consumption_matrix[i,j] = s
    
    return sea_consumption_matrix

def plot_sea_resources_used(lim, M, nom):
    fig, ax = plt.subplots()#111, 'matrix movie'

    ax.set_ylabel("n consumers")
    ax.set_xlabel("land productivity")



    max_matrice = max(map(max, M))
    print ('max', max_matrice)
    print ('final M', '\n', M)
 
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=max_matrice)
    #matrice = ax.matshow(M, cmap = cm.Blues, norm = normalize, extent = [lim.min_land_prod, lim.max_land_prod, lim.min_consumers, lim.max_consumers ])
    matrice = ax.imshow(M, cmap = cm.Blues, norm = normalize, interpolation = 'none')#extent = [lim.min_land_prod, lim.max_land_prod, lim.max_consumers, lim.min_consumers ]

    x =  np.arange(lim.min_land_prod, lim.max_land_prod, lim.prod_step)
    nx = x.shape[0]
    n_labels = len(x)-1
    step_x = int(nx / (n_labels - 1))
    x_positions = np.arange(0, nx, step_x )
    x_labels = x[::step_x]
    #ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:.1f}'.format(y))
    #ax.yaxis.set_major_formatter(ticks_y)
    plt.xticks(x_positions, x_labels)

    y =  np.arange(lim.min_consumers, lim.max_consumers, lim.con_step)
    ny = y.shape[0]
    n_ylabels = len(y)-1
    step_y = int(ny / (n_ylabels - 1))
    y_positions = np.arange(0, ny, step_y )
    y_labels = y[::step_y]
    #ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:.1f}'.format(y))
    #ax.yaxis.set_major_formatter(ticks_y)
    plt.yticks(y_positions, y_labels)
    
    plt.colorbar(matrice)
    #plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
       
    name_file = '../plots_costal_resources/histogram_sea_' + nom+'.png'
    plt.savefig(name_file,  bbox_inches = 'tight')
    plt.show()

start  = time.perf_counter() 

name = sys.argv[1]
lim = limits()
cnt = mc.constants()
consumers_parameter, productivity_parameter = generate_grid(lim)
matrix_sea_consumption = run_the_grid( consumers_parameter, productivity_parameter)
plot_sea_resources_used(lim, matrix_sea_consumption, name)

print ('time to run all this stuff', str(datetime.timedelta(seconds = (time.perf_counter() - start))))
