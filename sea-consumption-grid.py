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
import vector_multiple_consumers as mc



import pickle
import cmath


''' version 0.1 of the code that explores under which parameter space the consumers need to consume algae in a costal desert'''
'''
the code runs trought a set range of parameter spaces and computes the amount of sea resources consumed after X time steps
returns an hisotgram of searesource consumed when varing several combinations of parameters
start with land-production vs number of consumers.
'''


def sea_resources_used(M, nom):
    fig, ax = plt.subplots()#111, 'matrix movie'
 

    normalize = matplotlib.colors.Normalize(vmin=0, vmax=max_sea)
    matrice = ax.matshow(A, cmap = cm.OrRd, norm = normalize)# origin="lower"
    
    plt.colorbar(matrice)
    #plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    
    
    
    plt.show()
    name_file = '../plots_costal_resources/histogram_sea_' + nom+'.png'
    ani.save(name_file,  dpi = 80)#,writer = 'imagemagick')



