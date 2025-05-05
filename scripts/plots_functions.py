from __future__ import division

import numpy as np
from subprocess import call
from scipy.ndimage.interpolation import shift

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.cm as pltcm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator



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

    ax.set_xlabel("time steps")
    ax.set_ylabel("fuels burned per agent & time step")

    ax.plot(sea/par.n_consumers, alpha = 0.5, label='Seaweed')
    ax.plot(land/par.n_consumers, alpha = 0.5, label = 'Land fuel')

    t = np.linspace(0, par.time, len(sea))
    t_avg = []
    sea_avg = []
    land_avg = []
    rang = 20
    for ind in range(len(sea)-rang +1):
        sea_avg.append(np.mean(sea[ind:ind+rang]))
        land_avg.append(np.mean(land[ind:ind+rang]))
        t_avg.append(np.mean(t[ind:ind+rang]))

    ax.plot(t_avg, np.array(sea_avg)/par.n_consumers, color="blue", linewidth=2.4)
    ax.plot(t_avg, np.array(land_avg)/par.n_consumers, color="orange", linewidth=2.4)


    plt.legend(frameon = False)

    name = name_file(par, name)
    name_sea = './' + par.plots_dir + 'plot2_time_series_'+ name + '.eps'
    name_sea = './' + par.plots_dir + 'plot2_time_series_'+ name + '.png'
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
    

def prerpare_plot(ax, lim, M, which_par):

    ax.set_ylabel("HFG Number $n_b$")

    if which_par == 'land_productivity':
        ax.set_xlabel("Land Productivity $L_{p}$")
        x =  np.arange(lim.min_land_prod, lim.max_land_prod, lim.prod_step)
    elif which_par == 'high_land':
        ax.set_xlabel("Maximum Land $L^{max}$")
        x =  np.arange(lim.high_land_min, lim.high_land_max, lim.Lhigh_step)
    elif which_par == 'tidal_deluge': 
        ax.set_xlabel("Tidal Deposition $S_{d}$")
        x =  np.arange(lim.min_tidal_deluge, lim.max_tidal_deluge, lim.tidal_deluge_step)
    elif which_par == 'high_sea':    
        ax.set_xlabel("Maximum Sea $S^{max}$")
        x =  np.arange(lim.high_sea_min, lim.high_sea_max, lim.high_sea_step)
    
    max_matrice = max(map(max, M))
    min_matrice = min(map(min, M))

    print ('max', max_matrice)
    print ('final M Sea', '\n', M)

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

    y =  np.arange(lim.min_consumers, lim.max_consumers, lim.con_step)
    ny = y.shape[0]
    n_ylabels = len(y)-1
    step_y = int(ny / (n_ylabels - 1))
    y_positions = np.arange(0, ny, step_y )
    y_labels = y[::step_y]
    #ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:.1f}'.format(y))
    #ax.yaxis.set_major_formatter(ticks_y)

    return max_matrice, min_matrice, x_positions, x_labels, y_positions, y_labels


def prerpare_TriPlot(ax, lim, M, which_par, num):

    fs = 16
    if num == 0 or num == 2:
        ax.set_ylabel("HFG Number $n_b$", fontsize=fs)

    if which_par == 'land_productivity' and (num == 0 or num == 2):
        ax.set_xlabel("Land Productivity $L_{p}$", fontsize=fs)
    elif which_par == 'high_land' and (num == 0 or num == 2):
        ax.set_xlabel("Maximum Land $L^{max}$" , fontsize=fs)
    elif which_par == 'tidal_deluge' and (num == 0 or num == 2): 
        ax.set_xlabel("Tidal Deposition $S_{d}$", fontsize=fs)
    elif which_par == 'high_sea' and (num == 0 or num == 2):    
        ax.set_xlabel("Maximum Sea $S^{max}$" , fontsize=fs)

    if which_par == 'land_productivity':
        x =  np.arange(lim.min_land_prod, lim.max_land_prod, lim.prod_step)
    elif which_par == 'high_land':
        x =  np.arange(lim.high_land_min, lim.high_land_max, lim.Lhigh_step)
    elif which_par == 'tidal_deluge': 
        x =  np.arange(lim.min_tidal_deluge, lim.max_tidal_deluge, lim.tidal_deluge_step)
    elif which_par == 'high_sea':    
        x =  np.arange(lim.high_sea_min, lim.high_sea_max, lim.high_sea_step)
    
    max_matrice = max(map(max, M))
    min_matrice = min(map(min, M))

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

    y =  np.arange(lim.min_consumers, lim.max_consumers, lim.con_step)
    ny = y.shape[0]
    n_ylabels = len(y)-1
    step_y = int(ny / (n_ylabels - 1))
    y_positions = np.arange(0, ny, step_y )
    y_labels = y[::step_y]
    #ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:.1f}'.format(y))
    #ax.yaxis.set_major_formatter(ticks_y)

    if num == 0 or num == 2:
        ax.set_xticks(x_positions, x_labels)
        ax.set_yticks(y_positions, y_labels)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    return max_matrice, min_matrice

def prepare_QuadPlot(axes, lim, M, which_par, num):
  

    if which_par == 'land_productivity':
        x =  np.arange(lim.min_land_prod, lim.max_land_prod, lim.prod_step)
    elif which_par == 'high_land':
        x =  np.arange(lim.high_land_min, lim.high_land_max, lim.Lhigh_step)
    elif which_par == 'tidal_deluge': 
        x =  np.arange(lim.min_tidal_deluge, lim.max_tidal_deluge, lim.tidal_deluge_step)
    elif which_par == 'high_sea':    
        x =  np.arange(lim.high_sea_min, lim.high_sea_max, lim.high_sea_step)
    
    max_matrice = max(map(max, M))
    min_matrice = min(map(min, M))

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

    y =  np.arange(lim.min_consumers, lim.max_consumers, lim.con_step)
    ny = y.shape[0]
    n_ylabels = len(y)-1
    step_y = int(ny / (n_ylabels - 1))
    y_positions = np.arange(0, ny, step_y )
    y_labels = y[::step_y]
    #ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:.1f}'.format(y))
    #ax.yaxis.set_major_formatter(ticks_y)
   
    if num == 0:
        ax = axes[0, 0]
        ax.set_yticks(y_positions, y_labels)
        ax.set_xticks([])
    elif num == 1:
        ax = axes[0, 1]
        ax.set_xticks([])
        ax.set_yticks([])
    elif num == 2:
        ax = axes[1, 0]
        ax.set_xticks(x_positions, x_labels)
        ax.set_yticks(y_positions, y_labels)
    elif num == 3:
        ax = axes[1, 1]
        ax.set_yticks([])
        ax.set_xticks(x_positions, x_labels)
    ax.tick_params(axis='both', which='major', labelsize=11)

    return ax, max_matrice, min_matrice


def plot_sea_resources_used(par, lim, M, nom, which_par):

    fig, ax = plt.subplots()#111, 'matrix movie'
    max_matrice, min_matrice, x_positions, x_labels, y_positions, y_labels = prerpare_plot(ax, lim, M, which_par)

    normalize = matplotlib.colors.Normalize(vmin=min_matrice, vmax=max_matrice)
    #matrice = ax.matshow(M, cmap = cm.Blues, norm = normalize, extent = [lim.min_land_prod, lim.max_land_prod, lim.min_consumers, lim.max_consumers ])
    #matrice = ax.imshow(M, cmap = cm.Blues, norm = normalize, interpolation = 'none')#extent = [lim.min_land_prod, lim.max_land_prod, lim.max_consumers, lim.min_consumers ]
    
    cmap = cm.get_cmap('Blues', 5)    # 6 discrete colors
    matrice = ax.pcolormesh(M, cmap = cmap, norm = normalize)#extent = [lim.min_land_prod, lim.max_land_prod, lim.max_consumers, lim.min_consumers ]

    plt.xticks(x_positions, x_labels)
    plt.yticks(y_positions, y_labels)
    
    plt.colorbar(matrice)

       
    string_name_file = './' + par.plots_dir + 'Grid_sea_Lmax_' + which_par + nom +'.png'
    #string_name_file = './' + par.plots_dir + 'Grid_sea_Lmax_' + which_par + nom +'.svg'
    plt.savefig(string_name_file,  bbox_inches = 'tight')


def plot_land_resources_used(par, lim, M, nom, which_par):

    fig, ax = plt.subplots()#111, 'matrix movie'
    ax, max_matrice, min_matrice, x_positions, x_labels, y_positions, y_labels = prerpare_plot(ax, lim, M, which_par)
 
    normalize = matplotlib.colors.Normalize(vmin=min_matrice, vmax=max_matrice)
    
    cmap = cm.get_cmap('OrRd', 5)    # 6 discrete colors
    matrice = ax.pcolormesh(M, cmap = cmap, norm = normalize)#extent = [lim.min_land_prod, lim.max_land_prod, lim.max_consumers, lim.min_consumers ]
    
    plt.xticks(x_positions, x_labels)
    plt.yticks(y_positions, y_labels)
    plt.colorbar(matrice)

       
    string_name_file = './' + par.plots_dir + 'Grid_land_Lmax_' + which_par + nom +'.png'
    plt.savefig(string_name_file,  bbox_inches = 'tight')

def plot_jumps_matrix(par, lim, M, nom, which_par):

    fig, ax = plt.subplots()#111, 'matrix movie'
    ax, max_matrice, min_matrice, x_positions, x_labels, y_positions, y_labels = prerpare_plot(ax, lim, M, which_par)
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
 
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=max_matrice)
   
    cmap = cm.get_cmap('PuRd', 5) 
    #cmap = cm.get_cmap('Greens', 5)    # 6 discrete colors
    matrice = ax.pcolormesh(M, cmap = cmap, norm = normalize)#extent = [lim.min_land_prod, lim.max_land_prod, lim.max_consumers, lim.min_consumers ]

    plt.xticks(x_positions, x_labels)
    plt.yticks(y_positions, y_labels)
    plt.colorbar(matrice)
       
    string_name_file = './' + par.plots_dir + 'Grid_jump_Lmax_' + which_par + nom +'.png'
    plt.savefig(string_name_file,  bbox_inches = 'tight')


def tri_plotSeaLandJumps(par, lim, Mats, nom, which_par):

    # Create the triangular layout
    num_matrices = len(Mats)-1
    fig, axes = plt.subplots(num_matrices - 1, num_matrices - 1, figsize=(12, 10), constrained_layout=True)
    

    cmaps = [cm.get_cmap('Blues', 5), cm.get_cmap('OrRd', 5), cm.get_cmap('PuRd', 5)]
    #which_par = ['land_productivity', 'high_land', 'tidal_deluge', 'high_sea']
    ii = 0

    
    # Loop through the triangular arrangement
    for i in range(num_matrices - 1):
        for j in range(i + 1, num_matrices):
            # Get the current subplot
            ax = axes[i, j - 1] if j - 1 < len(axes[i]) else None
            if ax:
                # Load data for the parameter pair
                #filename = f"{Mats[i]}_{Marts[j]}.npy"
                #X, Y, Z = load_data(filename)
               
                max_matrice, min_matrice = prerpare_TriPlot(ax, lim, Mats[ii], which_par, ii)
                normalize = matplotlib.colors.Normalize(vmin=min_matrice, vmax=max_matrice)
                mesh = ax.pcolormesh(Mats[ii], cmap = cmaps[ii], norm = normalize)
                fig.colorbar(mesh)
                #plt.xticks(x_positions, x_labels)
                ii += 1
            else:
                # Hide unused subplots
                axes[i, j - 1].axis('off')

    # Hide empty subplots in the lower triangle
    for i in range(1, num_matrices - 1):
        for j in range(i):
            axes[i, j].axis('off')

    # Adjust layout and show the plot
    fig.subplots_adjust(wspace=0, hspace=0)
    string_name_file = './' + par.plots_dir + 'Tri_Mat_' + which_par + nom +'.png'
    plt.savefig(string_name_file,  bbox_inches = 'tight')


def quad_plotSeaLandJumps(par, lim, Mats, nom, which_par):
    fs = 14
    # Create the triangular layout
    num_matrices = len(Mats) -1
    fig, axes = plt.subplots(num_matrices - 1, num_matrices - 1, figsize=(13, 10))#, constrained_layout=True
    #axall = fig.add_subplot(111, frameon=False)

    cmaps = [cm.get_cmap('Blues', 5), cm.get_cmap('YlGn', 5), cm.get_cmap('PuRd', 5) ,cm.get_cmap('YlOrBr', 5) ]
    #which_par = ['land_productivity', 'high_land', 'tidal_deluge', 'high_sea']

    
    # Loop through the triangular arrangement
    for i in range(num_matrices+1):

        ax, max_matrice, min_matrice = prepare_QuadPlot(axes, lim, Mats[i], which_par, i)
        normalize = matplotlib.colors.Normalize(vmin=min_matrice, vmax=max_matrice)
        mesh = ax.pcolormesh(Mats[i], cmap = cmaps[i], norm = normalize)
        if i == 0 or i == 2:
            fig.colorbar(mesh,location='left')#ax=[ax]
        else:
            fig.colorbar(mesh)

    fig.text(0.12, 0.5, r"HFG Number ($n_b$)", ha='center', va='center', fontsize=fs, rotation='vertical')
    
    v = 0.5
    h = 0.06
    if which_par == 'land_productivity':
        fig.text(v, h, "Land Productivity $L_{p}$", ha='center', va='center', fontsize=fs)
    elif which_par == 'high_land':
        fig.text(v, h, "Maximum Land $L^{max}$", ha='center', va='center',  fontsize=fs)
    elif which_par == 'tidal_deluge': 
        fig.text(v, h, "Tidal Deposition $S_{d}$", ha='center', va='center',  fontsize=fs)
    elif which_par == 'high_sea':    
        fig.text(v, h, "Maximum Sea $S^{max}$", ha='center', va='center',  fontsize=fs)
    
    fig.text(0.11, 0.88, "MF",  fontsize=fs+2, verticalalignment='top', horizontalalignment='left')
    fig.text(0.91, 0.88, "TF",  fontsize=fs+2, verticalalignment='top', horizontalalignment='right')
    fig.text(0.09, 0.11, "mov",  fontsize=fs+2, verticalalignment='bottom', horizontalalignment='left')
    fig.text(0.90, 0.11, "ran",  fontsize=fs+2, verticalalignment='bottom', horizontalalignment='right')

    # Adjust layout and show the plot
    fig.subplots_adjust(wspace=0, hspace=0)
    #axall.set_xlabel('common xlabel')
    #axall.set_ylabel('common ylabel')
    string_name_file = './' + par.plots_dir + 'Quad_' + which_par + nom +'.png'
    plt.savefig(string_name_file,  bbox_inches = 'tight')


def plot_2jumps_vectors(par, lim, burners_nums, jumpVectors, nom):

    #fig, ax = plt.subplots()
    ax = plt.figure().gca()

    ax.set_ylabel("Average Movements")
    ax.set_xlabel(r"HFG Number ($n_b$)")


    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   

    #plt.plot(burners_nums, jumpVectors.T[0], ls = '--', color="magenta", alpha = 0.5)
    plt.plot(burners_nums, jumpVectors.T[0], 'o', color="magenta", alpha = 0.5, markersize=8, label = r'Tidal deposition: $S_d = $'+ format(lim.medium_tidal_deluge, '.2f'))
    #plt.plot(burners_nums, jumpVectors.T[1], ls = '-.', color="magenta", label = r'$S_d = $'+ format(lim.high_tidal_deluge, '.2f'))
    plt.plot(burners_nums, jumpVectors.T[1], '*', color="magenta",  markersize=8, label = r'Tidal deposition: $S_d = $'+ format(lim.high_tidal_deluge, '.2f'))
    plt.legend(frameon = False)
       
    string_name_file = './' + par.plots_dir + '2jump_vectors_' + nom+'.png'
    plt.savefig(string_name_file,  bbox_inches = 'tight')
    #string_name_file = './' + par.plots_dir + '2jump_vectors_' + nom+'.svg'
    #plt.savefig(string_name_file,  bbox_inches = 'tight')

def plot_2radius_vectors(par, lim, burners_nums, jumpVectors, nom):

    #fig, ax = plt.subplots()
    ax = plt.figure().gca()

    ax.set_ylabel("Maximum Range")
    ax.set_xlabel(r"HFG Number ($n_b$)")


    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   

    #plt.plot(burners_nums, jumpVectors.T[0], ls = '--', color="magenta", alpha = 0.5)
    plt.plot(burners_nums, jumpVectors.T[0], 'o', color="brown", alpha = 0.5, markersize=8, label = r'Tidal deposition: $S_d = $'+ format(lim.medium_tidal_deluge, '.2f'))
    #plt.plot(burners_nums, jumpVectors.T[1], ls = '-.', color="magenta", label = r'$S_d = $'+ format(lim.high_tidal_deluge, '.2f'))
    plt.plot(burners_nums, jumpVectors.T[1], '*', color="brown",  markersize=8, label = r'Tidal deposition: $S_d = $'+ format(lim.high_tidal_deluge, '.2f'))
    plt.legend(frameon = False)
       
    string_name_file = './' + par.plots_dir + '2range_vectors_' + nom+'.png'
    plt.savefig(string_name_file,  bbox_inches = 'tight')
    #string_name_file = './' + par.plots_dir + '2jump_vectors_' + nom+'.svg'
    #plt.savefig(string_name_file,  bbox_inches = 'tight')


def plot_envelope_2jumps_vectors(par, lim, mean_values, min_values, max_values , nom):
    
# Create a figure and axes
    fs = 14
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Plot the ±1 sigma envelope and the mean
    x = np.arange(lim.min_consumers, lim.max_consumers, lim.con_step)
    lower_bound = np.ones(len(x)) * par.radius
    
    ax.fill_between(x, min_values[0], max_values[0] , color="magenta", alpha = 0.1, linewidth=0.0)
    ax.plot(x, mean_values[0], lw = 2.0, ls = '--', color="magenta",label = r'Medium Tidal Deposition: $S_d = $'+ format(lim.medium_tidal_deluge, '.2f'))

    ax.fill_between(x, min_values[1], max_values[1] , color="magenta", alpha = 0.3, linewidth=0.0)
    ax.plot(x, mean_values[1], lw = 2.5,ls = ':', color="magenta",label = r'High Tidal Deposition: $S_d = $'+ format(lim.high_tidal_deluge, '.2f'))

    # Add labels, legend, and title
    ax.set_ylabel("Average Movements Distribution", fontsize=fs)
    ax.set_xlabel(r"HFG Number ($n_b$)", fontsize=fs)
    #ax.set_title('Mean and ±1 Sigma Envelope of Arrays')
    ax.legend(frameon=False, fontsize=fs)

    string_name_file = './' + par.plots_dir + '2jumps_vectors_dist' + nom+'.png'
    plt.savefig(string_name_file,  bbox_inches = 'tight')
    string_name_file = './' + par.plots_dir + '2jump_vectors_dist' + nom+'.svg'
    plt.savefig(string_name_file,  bbox_inches = 'tight')



def plot_envelope_2radius_vectors(par, lim, mean_values, min_values, max_values , nom):
    
# Create a figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.tick_params(axis='both', which='major', labelsize=12)
    fs = 14
    # Plot the ±1 sigma envelope and the mean
    x = np.arange(lim.min_consumers, lim.max_consumers, lim.con_step)
    lower_bound = np.ones(len(x)) * par.radius
    
    ax.fill_between(x, min_values[0]/par.length, max_values[0]/par.length , color="brown", alpha = 0.1, linewidth=0.0)
    ax.plot(x, mean_values[0]/par.length, lw = 2.5, ls = '--', color="brown")

    ax.fill_between(x, min_values[1]/par.length, max_values[1]/par.length , color="brown", alpha = 0.3, linewidth=0.0)
    ax.plot(x, mean_values[1]/par.length, lw = 2.0, ls = ':', color="brown")

    # Add labels, legend, and title
    ax.set_ylabel("Maximum Range Distribution/landscape size", fontsize=fs)
    ax.set_xlabel(r"HFG Number ($n_b$)", fontsize=fs)
    #ax.set_title('Mean and ±1 Sigma Envelope of Arrays')

    string_name_file = './' + par.plots_dir + '2range_vectors_dist' + nom+'.png'
    plt.savefig(string_name_file,  bbox_inches = 'tight')
    string_name_file = './' + par.plots_dir + '2jump_vectors_dist' + nom+'.svg'
    plt.savefig(string_name_file,  bbox_inches = 'tight')

def name_file(par, name):
    
    characteristics = 'burn'+ '{:2}'.format(par.n_consumers)+'_cells'+ '{:2}'.format(par.length)+\
                    '_t'+ '{:3}'.format(par.time)+'_hL'+'{:1}'.format(par.high_land)+'_hS'+\
                    '{:}'.format(par.high_sea)+'_Lp'+ '{:.1f}'.format(par.land_productivity)+\
                    '_tD' + '{:.1f}'.format(par.tidal_deluge)
    
    return name +'_'+ characteristics