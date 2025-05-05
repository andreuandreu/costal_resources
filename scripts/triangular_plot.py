import matplotlib.pyplot as plt
import numpy as np
import os

import matplotlib
import matplotlib.cm as cm

import resource_movement_grids as rmg
import itertools

def prepare_gridPlot(ax, lim, M, which_par):

    max_matrice = max(map(max, M))
    min_matrice = min(map(min, M))
  
    '''
    if which_par == 'land_productivity':
        x =  np.arange(lim.min_land_prod, lim.max_land_prod, lim.prod_step)
    elif which_par == 'high_land':
        x =  np.arange(lim.high_land_min, lim.high_land_max, lim.Lhigh_step)
    elif which_par == 'tidal_deluge': 
        x =  np.arange(lim.min_tidal_deluge, lim.max_tidal_deluge, lim.tidal_deluge_step)
    elif which_par == 'high_sea':    
        x =  np.arange(lim.high_sea_min, lim.high_sea_max, lim.high_sea_step)

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
    '''

    ax.set_xticks([])
    ax.set_yticks([])
   
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

    # Create the triangular layout
    all_pairs = list(itertools.combinations(par.par_names, 2))
    
    num_matrices = int(len(all_pairs)/2)
    fig, axes = plt.subplots(num_matrices - 1, num_matrices - 1, figsize=(12, 10), constrained_layout=True)
    
    cmaps = { 'MF':cm.get_cmap('Blues', 5), 'TF':cm.get_cmap('YlGn', 5), 'mov':cm.get_cmap('PuRd', 5) ,'rad':cm.get_cmap('YlOrBr', 5) }
    which_ind = {'MF':0, 'TF':1, 'mov':2, 'rad':3}

    max_matrice, min_matrice = prepare_minMaxLevels(par, lim, all_pairs, which_ind[which_tria], nom)
    
    ii = 0

    # Loop through the triangular arrangement
    for i in range(num_matrices - 1):
        for j in range(i + 1, num_matrices):
            # Get the current subplot
            ax = axes[i, j - 1] if j - 1 < len(axes[i]) else None
            print('we are in pair index', ii)
            if ax:
                # Load data for the parameter pair
                print ('Loading data for',   all_pairs[ii])
                which_pars = all_pairs[ii]
                save_name = rmg.name_files(par, lim, which_pars)
                Mats = np.load(par.data_dir + nom + '_' + save_name, allow_pickle=True)
                
                ax = prepare_gridPlot(ax, lim, Mats[which_ind[which_tria]], which_pars)
                normalize = matplotlib.colors.Normalize(vmin=min_matrice, vmax=max_matrice)
                mesh = ax.pcolormesh(Mats[which_ind[which_tria]], cmap = cmaps[which_tria], norm = normalize)
                fig.colorbar(mesh)
                #plt.xticks(x_positions, x_labels)
            else:
                # Hide unused subplots
                axes[i, j - 1].axis('off')
            ii += 1

    # Hide empty subplots in the lower triangle
    for i in range(1, num_matrices - 1):
        for j in range(i):
            axes[i, j].axis('off')

    # Adjust layout and show the plot
    fig.subplots_adjust(wspace=0, hspace=0)
    string_name_file = './' + par.plots_dir + 'Triangle_plot_' + which_tria + nom +'.png'
    plt.savefig(string_name_file,  bbox_inches = 'tight')





'''


# Directory to save and load .npy files
data_dir = "./data_matrices"
os.makedirs(data_dir, exist_ok=True)

# Function to generate and save data as .npy files
def generate_and_save_data(param1, param2, filename):
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X * param1) + np.cos(Y * param2)
    np.save(os.path.join(data_dir, filename), Z)
    return X, Y, Z

# Function to load data from .npy files
def load_data(filename):
    Z = np.load(os.path.join(data_dir, filename))
    x = np.linspace(0, 10, Z.shape[1])
    y = np.linspace(0, 10, Z.shape[0])
    X, Y = np.meshgrid(x, y)
    return X, Y, Z

# Parameters
parameters = ['a', 'b', 'c', 'd', 'e']
num_params = len(parameters)

# Generate and save data for all parameter pairs
for i in range(num_params - 1):
    for j in range(i + 1, num_params):
        filename = f"{parameters[i]}_{parameters[j]}.npy"
        generate_and_save_data(i + 1, j + 1, filename)

# Create the triangular layout
fig, axes = plt.subplots(num_params - 1, num_params - 1, figsize=(12, 10), constrained_layout=True)

# Loop through the triangular arrangement
for i in range(num_params - 1):
    for j in range(i + 1, num_params):
        # Get the current subplot
        ax = axes[i, j - 1] if j - 1 < len(axes[i]) else None
        if ax:
            # Load data for the parameter pair
            filename = f"{parameters[i]}_{parameters[j]}.npy"
            X, Y, Z = load_data(filename)
            
            # Plot the data using pcolormesh
            mesh = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
            ax.set_xlabel(parameters[i])
            ax.set_ylabel(parameters[j])
            fig.colorbar(mesh, ax=ax)
        else:
            # Hide unused subplots
            axes[i, j - 1].axis('off')

# Hide empty subplots in the lower triangle
for i in range(1, num_params - 1):
    for j in range(i):
        axes[i, j].axis('off')

# Adjust layout and show the plot
plt.suptitle("Triangular Multi-Panel Plot with pcolormesh", fontsize=16)
plt.show()
'''