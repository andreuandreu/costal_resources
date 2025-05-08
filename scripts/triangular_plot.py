import matplotlib.pyplot as plt
import numpy as np
import os

import matplotlib
import matplotlib.cm as cm

import resource_movement_grids as rmg
import itertools

def prepare_gridPlot(ax, lim, which_pars, i, j, ii):
    
    fs = 12

    if which_pars[0] == 'land_productivity':
        y =  np.arange(lim.min_land_prod, lim.max_land_prod, lim.prod_step)
        
        ny = y.shape[0]
        n_ylabels = len(y)-1
        step_y = int(ny / (n_ylabels - 1))
        y_positions = np.arange(0, ny, step_y )
        float_formatter = "{:.2f}".format
        np.set_printoptions(formatter={'float_kind':float_formatter})
        y_labels = y[::step_y]
        for i,yy in enumerate(y):
            y_labels[i] = "{:.2f}".format(yy)

        ax.set_ylabel("Land Productivity $L_{p}$", rotation=270, labelpad=15, fontsize=fs)

    elif which_pars[0] == 'high_land':
        y =  np.arange(lim.high_land_min, lim.high_land_max, lim.Lhigh_step)

        ny = y.shape[0]
        n_ylabels = len(y)-1
        step_y = int(ny / (n_ylabels - 1))
        y_positions = np.arange(0, ny, step_y )
        y_labels = y[::step_y]
        ax.set_ylabel("Highest Land $L^{h}$", rotation=270, labelpad=15, fontsize=fs)

    elif which_pars[0] == 'tidal_deluge': 
        y =  np.arange(lim.min_tidal_deluge, lim.max_tidal_deluge, lim.tidal_deluge_step)

        ny = y.shape[0]
        n_ylabels = len(y)-1
        step_y = int(ny / (n_ylabels - 1))
        y_positions = np.arange(0, ny, step_y )
        float_formatter = "{:.2f}".format
        np.set_printoptions(formatter={'float_kind':float_formatter})
        y_labels = y[::step_y]
        for i,yy in enumerate(y):
            y_labels[i] = "{:.2f}".format(yy)

        ax.set_ylabel("Tidal Deposition $S_{d}$", rotation=270, labelpad=15, fontsize=fs)

    elif which_pars[0] == 'high_sea':    
        y =  np.arange(lim.high_sea_min, lim.high_sea_max, lim.high_sea_step)

        ny = y.shape[0]
        n_ylabels = len(y)-1
        step_y = int(ny / (n_ylabels - 1))
        y_positions = np.arange(0, ny, step_y )
        float_formatter = "{:1f}".format
        np.set_printoptions(formatter={'float_kind':float_formatter})
        y_labels = y[::step_y]
        for i,yy in enumerate(y):
            y_labels[i] = "{:1f}".format(yy)
        ax.set_ylabel("Highest Sea $S^{h}$", rotation=270, labelpad=15, fontsize=fs)

    elif which_pars[0] == 'burners_number':
        y =  np.arange(lim.min_consumers, lim.max_consumers, lim.con_step)

        ny = y.shape[0]
        n_ylabels = len(y)-1
        step_y = int(ny / (n_ylabels - 1))
        y_positions = np.arange(0, ny, step_y )
        y_labels = y[::step_y]
        ax.set_ylabel("HFG Number $n_b$", rotation=270, labelpad=15, fontsize=fs)

    else:
        print('Error: unknown parameter')
        return
    
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    if which_pars[1] == 'land_productivity':
        x =  np.arange(lim.min_land_prod, lim.max_land_prod, lim.prod_step)
    elif which_pars[1] == 'high_land':
        x =  np.arange(lim.high_land_min, lim.high_land_max, lim.Lhigh_step)
    elif which_pars[1] == 'tidal_deluge': 
        x =  np.arange(lim.min_tidal_deluge, lim.max_tidal_deluge, lim.tidal_deluge_step)
    elif which_pars[1] == 'high_sea':    
        x =  np.arange(lim.high_sea_min, lim.high_sea_max, lim.high_sea_step)
    elif which_pars[1] == 'burners_number':
        x =  np.arange(lim.min_consumers, lim.max_consumers, lim.con_step)
    else:
        print('Error: unknown parameter')
        return
        

    nx = x.shape[0]
    n_labels = len(x)-1
    step_x = int(nx / (n_labels - 1))
    x_positions = np.arange(0, nx, step_x )
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    x_labels = x[::step_x]#"{:10.4f}".format(x)
    for i,xx in enumerate(x):
        x_labels[i] = "{:.2f}".format(xx)

    ax.xaxis.set_label_position("top")
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()


    ax.set_yticks(y_positions, y_labels)
    ax.set_xticks(x_positions, x_labels, rotation=90, ha='left')
    

    if i < 10 :
        ax.set_yticks([])


    if ii > 3:#
        ax.set_xticks([])
    else:
        if which_pars[1] == "land_productivity":
            ax.set_xlabel("Land Productivity $L_{p}$", fontsize=fs)
        
        elif which_pars[1] == 'high_land':
            ax.set_xlabel("Highest Land $L^{h}$", fontsize=fs)
            
        elif which_pars[1] == 'tidal_deluge':
            ax.set_xlabel("Tidal Deposition $S_{d}$",  fontsize=fs)
           
        elif which_pars[1] == 'high_sea':    
            ax.set_xlabel("Highest Sea $S^{h}$",  fontsize=fs)
            

    return ax


def prepare_minMaxLevels(par, lim, all_pairs, which_ind, nom):

    save_name = rmg.name_files(par, lim, all_pairs[0])
    Mats = np.load(par.data_dir + nom + '_' + save_name, allow_pickle=True)
    M = Mats[which_ind]
    for i, a in enumerate(all_pairs):

        save_name = rmg.name_files(par, lim, a)
        Mats = np.load(par.data_dir + nom + '_' + save_name, allow_pickle=True)
        M = np.append(M, Mats[which_ind])
    
    #print ('M', M, len(M))
    max_matrice = max( M)
    min_matrice = min( M)

    return max_matrice, min_matrice



def triangle_plotSeaLandJumps(par, lim, nom, which_tria):

    fs = 14
    # Create the triangular layout
    all_pairs = list(itertools.combinations(par.par_names, 2))
    
    num_matrices = int(len(all_pairs)/2)
    fig, axes = plt.subplots(num_matrices - 1, num_matrices - 1, figsize=(12, 10))#, constrained_layout=True
    
    cmaps = { 'MF':cm.get_cmap('Blues', 5), 'TF':cm.get_cmap('YlGn', 5), 'mov':cm.get_cmap('PuRd', 5) ,'ran':cm.get_cmap('YlOrBr', 5) }
    which_ind = {'MF':0, 'TF':1, 'mov':2, 'ran':3}

    max_matrice, min_matrice = prepare_minMaxLevels(par, lim, all_pairs, which_ind[which_tria], nom)
    
    ii = 0

    # Loop through the triangular arrangement
    for i in range(num_matrices - 1):
        for j in range(i + 1, num_matrices):
            # Get the current subplot
            ax = axes[i, j - 1] if j - 1 < len(axes[i]) else None
            
            if ax:
                print('we are in pair index', ii)
                # Load data for the parameter pair
                print ('Loading data for',   all_pairs[ii])
                which_pars = all_pairs[ii]
                save_name = rmg.name_files(par, lim, which_pars)
                Mats = np.load(par.data_dir + nom + '_' + save_name, allow_pickle=True)
                
                ax = prepare_gridPlot(ax, lim, which_pars, i, j-1, ii)
                normalize = matplotlib.colors.Normalize(vmin=min_matrice, vmax=max_matrice)
                mesh = ax.pcolormesh(Mats[which_ind[which_tria]], cmap = cmaps[which_tria], norm = normalize)

            else:
                # Hide unused subplots
                axes[i, j - 1].axis('off')
            ii += 1

    # Hide empty subplots in the lower triangle
    for i in range(1, num_matrices - 1):
        for j in range(i):
            axes[i, j].axis('off')


    # Adjust layout and show the plot
    cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
    fig.colorbar(mesh, cax=cbar_ax)
    fig.subplots_adjust(wspace=0, hspace=0)


    string_name_file = './' + par.plots_dir + 'Triangle_plot_' + which_tria + nom +'.png'
    plt.savefig(string_name_file,   bbox_inches = 'tight')#,

