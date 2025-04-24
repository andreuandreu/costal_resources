import matplotlib.pyplot as plt
import numpy as np
import os

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