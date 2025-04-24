from __future__ import division

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.ndimage.interpolation import shift
import matplotlib.animation as animation
import matplotlib
from subprocess import call
import matplotlib.ticker as ticker

import matplotlib.cm as pltcm
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter

import config as cfg


def matrix_movie(par, M, position, nom):
    fig, ax = plt.subplots()#111, 'matrix movie'
    A = [M[0].T]
    print("\n Initial values landscape vector", A, '\n')
    ax.clear()
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=par.high_land)
    matrice = ax.matshow(A, cmap = cm.OrRd, norm = normalize)
    ax.scatter(position[0][0], position[0][1], marker = 'o', facecolors = 'k')#.plot(position[0])
    plt.colorbar(matrice)
    #plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

    
    def update(i):
        #matrice.axes.clear()
        A = [M[i].T]
        #matrice.axes.clear()

        if i>0:
            #color = A[position[i-1][1], position[i-1][0]]
            #print( color, 'aaa', position[i-1], A )
            matrice.axes.scatter(position[i-1][0], position[i-1][1], marker = 'o', c = 'w', lw = 0,  s=33)#
        matrice.set_array(A)
        print (i, 'iiii', len(position), len(M))
        matrice.axes.scatter(position[i][0], position[i][1], marker = 'o', facecolors = 'k', lw = 0, s=30)
        

    ani = animation.FuncAnimation(fig, update, frames=len(M), interval=630)#

    
    
    name_gif = 'matrix_land_' + nom+'.gif'
    #ani.save(name_gif,  dpi = 80)#,writer = 'ima
    plt.show()


def vector_movie(par, L, S, position, nom):
    #fig, ax = plt.subplots()#111, 'matrix movie'
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(211)

    A = np.rot90([L[0][::-1]])#
    B = np.rot90([S[0][::-1]])#

    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9) 

    print('\n initial position', position[0], '\n')
    ax1.clear()
    ax2.clear()
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    #

    normalizeL = matplotlib.colors.Normalize(vmin=0, vmax=par.high_land)
    normalizeS = matplotlib.colors.Normalize(vmin=0, vmax=par.high_sea)
    matriceL = ax1.matshow(A, cmap = pltcm.OrRd, norm = normalizeL)# #extent = [left,right, up, down ]
    matriceS = ax2.matshow(B, cmap = pltcm.Blues, norm = normalizeS)# #extent = [left,right, up, down ]
    
    #ax1.set_aspect(aspect=0.6)
    ax2.set_position([0.1,0.081, 0.81, 0.865])
    ax2.set_aspect(aspect=1.6)

    #cm = plt.get_cmap('gist_rainbow')
    #cNorm = colors.Normalize(vmin = 0, vmax = par.n_consumers -1)
    #scalarMap = pltcm.ScalarMappable(norm=cNorm, cmap = cm)
    #facecolors = ['k', 'r', 'b', 'g', 'o', 'y']
    #ax.set_color_cycle([cm(1.*i/par.n_consumers)  for i in range(par.n_consumers)])

    jumper_plot_position = -0.5

    colors = ['y', 'g', 'b', 'r', 'o', 'k']

    for i, s in   enumerate(position[0]):
        ax1.plot(jumper_plot_position, s, marker = 'o', c = colors[i], markersize=8, fillstyle = 'full', markeredgewidth = 0.0)#.plot(position[0])#, facecolors = scalarMap.to_rgba[i], color = '0.5'
    #plt.colorbar(matrice)
    #plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

    
    
    def update(i):
        
        A = np.rot90([L[i][::-1]])
        B = np.rot90([S[i][::-1]])
        #matriceS.axes.clear()

        
        if i>0:
            #color = A[position[i-1][1], position[i-1][0]]
            #print( color, 'aaa', position[i-1], A )
            for s in position[i-1]:
                matriceL.axes.plot(jumper_plot_position, s,  marker = 'o', c = 'w', alpha = 1.0, lw = 0, markersize=8.2, fillstyle = 'full',  markeredgewidth = 0.0)#
       
        
        matriceL.set_array(A)
        matriceS.set_array(B)
        print (i, 'iiii', len(position), len(L))
        for i, s in enumerate(position[i]):
            matriceL.axes.plot(jumper_plot_position, s,  marker = 'o', c = colors[i], lw = -0.1,  markersize=8, fillstyle = 'full',  markeredgewidth = 0.0)#, facecolors = scalarMap.to_rgba[i],, markeredgewidth = 0.0,  c = '0.5'
        
    ax1.axis('off')
    ax2.axis('off')
    
    ani = animation.FuncAnimation(fig, update, frames=len(L), interval=220)#

    
    #plt.show()
    name_gif = './' + par.plots_dir + 'matrix_land_' + nom+'.gif'
    ani.save(name_gif,  dpi = 80)#,writer = 'imagemagick')
    matriceL.axes.clear()


def plot_aggregated_resources(par, sea, land, name):

    fig = plt.figure(name)
    ax = fig.add_subplot(111)

    ax.set_xlabel("t")
    ax.set_ylabel("Fuels Burned")

    ax.plot(sea, alpha = 0.5, label='Seaweed')
    ax.plot(land, alpha = 0.5, label = 'Land fuel')

    t = np.linspace(0, par.time, len(sea))
    t_avg = []
    sea_avg = []
    land_avg = []
    rang = 20
    for ind in range(len(sea)-rang +1):
        sea_avg.append(np.mean(sea[ind:ind+rang]))
        land_avg.append(np.mean(land[ind:ind+rang]))
        t_avg.append(np.mean(t[ind:ind+rang]))

    ax.plot(t_avg, sea_avg, color="blue", linewidth=2.4)
    ax.plot(t_avg, land_avg, color="orange", linewidth=2.4)


    plt.legend(frameon = False)

    name = name_file(par, name)
    name_sea = './' + par.plots_dir + 'plot2_time_series/sea_resources_'+ name + '.eps'
    name_sea = './' + par.plots_dir + 'plot2_time_series/sea_resources_'+ name + '.png'
    plt.savefig(name_sea,  bbox_inches = 'tight')
    plt.show()

def plot_aggregated_movements(par, sea, land, name):

    fig = plt.figure(name)
    ax = fig.add_subplot(111)

    ax.set_xlabel("t")
    ax.set_ylabel("Number of movements")

    ax.plot(sea, alpha = 0.5, label='Seaweed')
    ax.plot(land, alpha = 0.5, label = 'Land fuel')

    t = np.linspace(0, par.time, len(sea))
    t_avg = []
    sea_avg = []
    land_avg = []
    rang = 20
    for ind in range(len(sea)-rang +1):
        sea_avg.append(np.mean(sea[ind:ind+rang]))
        land_avg.append(np.mean(land[ind:ind+rang]))
        t_avg.append(np.mean(t[ind:ind+rang]))

    ax.plot(t_avg, sea_avg, color="blue", linewidth=2.4)
    ax.plot(t_avg, land_avg, color="orange", linewidth=2.4)


    plt.legend(frameon = False)

    name = name_file(par, name)
    #name_sea = './' + par.plots_dir + 'plot_3time_series_jumps_'+ name + '.eps'
    #name_sea = './' + par.plots_dir + 'plot3_3time_series_jumps_'+ name + '.svg'
    name_sea = './' + par.plots_dir + 'plot_3time_series_jumps_'+ name + '.png'
    plt.savefig(name_sea,  bbox_inches = 'tight')
    plt.show()

def plot_1cell_resources(par, sea, land, name):

    fig = plt.figure(name)
    ax = fig.add_subplot(111)
    ax.set_xlabel("t")
    ax.set_ylabel("fuel in cell ")

    ax.plot(sea.T[c], alpha = 0.5, label= 'Seaweed')
    ax.plot(land.T[c], alpha = 0.5, label = 'Land fuel')

    t = np.linspace(0, par.time, len(sea))
   

    plt.legend(frameon = False)

    name = name_file(par, name)
    name_sea = './' + par.plots_dir + 'plot1_vector_movie/1cell_resources_'+ name + '.svg'
    #name_sea = './' + par.plots_dir + 'plot1_vector_movie/1cell_resources_'+ name + '.eps'
    #name_sea = './' + par.plots_dir + 'plot1_vector_movie/1cell_resources_'+ name + '.png'
    plt.savefig(name_sea,  bbox_inches = 'tight')
    plt.show()

def plot3_1cell_resources(par, sea, land, name):

    nrow = 3
    ncol = 1
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol)
    #ax.set_xlabel("t")
    #ax.set_ylabel("fuel in cell ")

    which_cells = [3, 8, 15]
    for ax, c in zip(axs, which_cells):
        ax.plot(sea.T[c], alpha = 0.5, label= 'Seaweed')
        ax.plot(land.T[c], alpha = 0.5, label = 'Land fuel')

    t = np.linspace(0, par.time, len(sea))
   

    plt.legend(frameon = False)

    name = name_file(par, name)
    name_sea = './' + par.plots_dir + 'plot_3-1cell_resources_'+ name + '.svg'
    #name_sea = './' + par.plots_dir + 'plot_3-1cell_resources_'+ name + '.eps'
    #name_sea = './' + par.plots_dir + 'plot_3-1cell_resources_'+ name + '.png'
    plt.savefig(name_sea,  bbox_inches = 'tight')
    plt.show()



def plot_sea_resources_used(par, lim, M, nom, which_par):
    
    fig, ax = plt.subplots()#111, 'matrix movie'

    ax.set_ylabel("HFG Number")

    if which_par == 'land_productivity':
        ax.set_xlabel("$L_{p}$")
        x =  np.arange(lim.min_land_prod, lim.max_land_prod, lim.prod_step)
    elif which_par == 'high_land':
        ax.set_xlabel("$L^{max}$")
        x =  np.arange(lim.high_land_min, lim.high_land_max, lim.Lhigh_step)
    elif which_par == 'tidal_deluge': 
        ax.set_xlabel("$S_{d}$")
        x =  np.arange(lim.min_tidal_deluge, lim.max_tidal_deluge, lim.tidal_deluge_step)
    elif which_par == 'high_sea':    
        ax.set_xlabel("$S^{max}$")
        x =  np.arange(lim.high_sea_min, lim.high_sea_max, lim.high_sea_step)
    
    max_matrice = max(map(max, M))
    min_matrice = min(map(min, M))

    print ('max', max_matrice)
    print ('final M Sea', '\n', M)
 
    normalize = matplotlib.colors.Normalize(vmin=min_matrice, vmax=max_matrice)
    #matrice = ax.matshow(M, cmap = cm.Blues, norm = normalize, extent = [lim.min_land_prod, lim.max_land_prod, lim.min_consumers, lim.max_consumers ])
    #matrice = ax.imshow(M, cmap = cm.Blues, norm = normalize, interpolation = 'none')#extent = [lim.min_land_prod, lim.max_land_prod, lim.max_consumers, lim.min_consumers ]
    
    cmap = cm.get_cmap('Blues', 5)    # 6 discrete colors
    matrice = ax.pcolormesh(M, cmap = cmap, norm = normalize)#extent = [lim.min_land_prod, lim.max_land_prod, lim.max_consumers, lim.min_consumers ]

    nx = x.shape[0]
    n_labels = len(x)-1
    step_x = int(nx / (n_labels - 1))
    x_positions = np.arange(0, nx, step_x )
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    x_labels = x[::step_x]#"{:10.4f}".format(x)
    for i,xx in enumerate(x):
        x_labels[i] = "{:.2f}".format(xx)

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
       
    string_name_file = './' + par.plots_dir + 'plot_seaGrid_histogram_sea_Lmax_' + nom+'.png'
    #string_name_file = './' + par.plots_dir + 'plot_seaGrid_histogram_sea_Lmax_' + nom+'.svg'
    plt.savefig(string_name_file,  bbox_inches = 'tight')


def plot_land_resources_used(par, lim, M, nom, which_par):
    fig, ax = plt.subplots()#111, 'matrix movie'

    ax.set_ylabel("HFG Number ")
    
    if which_par == 'land_productivity':
        ax.set_xlabel("$L_{p}$")
        x =  np.arange(lim.min_land_prod, lim.max_land_prod, lim.prod_step)
    elif which_par == 'high_land':
        ax.set_xlabel("$L^{max}$")
        x =  np.arange(lim.high_land_min, lim.high_land_max, lim.Lhigh_step)
    elif which_par == 'tidal_deluge': 
        ax.set_xlabel("$S_{d}$")
        x =  np.arange(lim.min_tidal_deluge, lim.max_tidal_deluge, lim.tidal_deluge_step)
    elif which_par == 'high_sea':    
        ax.set_xlabel("$S^{max}$")
        x =  np.arange(lim.high_sea_min, lim.high_sea_max, lim.high_sea_step)
    
    max_matrice = max(map(max, M))
    min_matrice = min(map(min, M))
    print ('max', max_matrice)
    print ('final M land', '\n', M)
 
    normalize = matplotlib.colors.Normalize(vmin=min_matrice, vmax=max_matrice)
    #matrice = ax.matshow(M, cmap = cm.Blues, norm = normalize, extent = [lim.min_land_prod, lim.max_land_prod, lim.min_consumers, lim.max_consumers ])
    #matrice = ax.imshow(M, cmap = cm.Blues, norm = normalize, interpolation = 'none')#extent = [lim.min_land_prod, lim.max_land_prod, lim.max_consumers, lim.min_consumers ]
    
    cmap = cm.get_cmap('OrRd', 5)    # 6 discrete colors
    matrice = ax.pcolormesh(M, cmap = cmap, norm = normalize)#extent = [lim.min_land_prod, lim.max_land_prod, lim.max_consumers, lim.min_consumers ]


    nx = x.shape[0]
    n_labels = len(x)-1
    step_x = int(nx / (n_labels - 1))
    x_positions = np.arange(0, nx, step_x )
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    x_labels = x[::step_x]#"{:10.4f}".format(x)
    for i,xx in enumerate(x):
        x_labels[i] = "{:.2f}".format(xx)

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
       
    string_name_file = './' + par.plots_dir + 'plot4_landGrid/histogram_land_Lmax_' + nom+'.png'
    plt.savefig(string_name_file,  bbox_inches = 'tight')

def plot_jumps_matrix(par, lim, M, nom):

    fig, ax = plt.subplots()#111, 'matrix movie'

    ax.set_ylabel("HFG Number ")
    #ax.set_xlabel("$L_{max}$")
    ax.set_xlabel("$L_{p}$")

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    max_matrice = max(map(max, M))
    print ('max', max_matrice)
    print ('final M jumps', '\n', M)
 
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=max_matrice)
   
    #cmap = cm.get_cmap('PuRd', 5) 
    cmap = cm.get_cmap('Greens', 5)    # 6 discrete colors
    matrice = ax.pcolormesh(M, cmap = cmap, norm = normalize)#extent = [lim.min_land_prod, lim.max_land_prod, lim.max_consumers, lim.min_consumers ]

    #x =  np.arange(lim.min_land_max, lim.max_land_max, lim.Lmax_step)
    x =  np.arange(lim.min_land_prod, lim.max_land_prod, lim.prod_step)
    nx = x.shape[0]
    n_labels = len(x)-1
    step_x = int(nx / (n_labels - 1))
    x_positions = np.arange(0, nx, step_x )
    x_labels = x[::step_x]
    for i,xx in enumerate(x):
        x_labels[i] = "{:.2f}".format(xx)
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
       
    string_name_file = './' + par.plots_dir + 'histogram_jump_Lmax_' + nom+'.png'
    plt.savefig(string_name_file,  bbox_inches = 'tight')


def plot_2jumps_vectors(par, burners_nums, jumpVectors, nom):

    fig, ax = plt.subplots()

    ax.set_ylabel("Average Movements")
    ax.set_xlabel("HFG Number")

    print( 'whaaaaat', burners_nums, jumpVectors.T[0])
    #plt.plot(par.n_consumers, MovVectors[0], color="orange", 'o', markersize=8)
    plt.plot(burners_nums, jumpVectors.T[0], ls = '--', color="magenta", alpha = 0.5, label = r'$S_d = 0.25$')
    #plt.plot(par.n_consumers, MovVectors[1], color="blue", '+', markersize=8)
    plt.plot(burners_nums, jumpVectors.T[1], ls = '-.', color="magenta", label = r'$S_d = 0.4$')
    plt.legend(frameon = False)
       
    string_name_file = './' + par.plots_dir + '2jump_vectors_' + nom+'.png'
    plt.savefig(string_name_file,  bbox_inches = 'tight')
    #string_name_file = './' + par.plots_dir + '2jump_vectors_' + nom+'.svg'
    #plt.savefig(string_name_file,  bbox_inches = 'tight')
    

def name_file(par, name):
    
    characteristics = 'burn'+ '{:2}'.format(par.n_consumers)+'_cells'+ '{:2}'.format(par.length)+\
                    '_t'+ '{:3}'.format(par.time)+'_hL'+'{:1}'.format(par.high_land)+'_hS'+\
                    '{:}'.format(par.high_sea)+'_Lp'+ '{:.1f}'.format(par.land_productivity)+\
                    '_tD' + '{:.1f}'.format(par.tidal_deluge)
    
    return name +'_'+ characteristics