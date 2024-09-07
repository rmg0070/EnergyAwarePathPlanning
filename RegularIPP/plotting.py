import os
import numpy as np
import matplotlib.pyplot as plt
# 
# Define the paths to the directories
# C:\\Users\\LibraryUser\\Desktop\\IPP\\figs
mixture_path = 'C:\\Users\\LibraryUser\\Desktop\\IPP\\MixtureGP\\1\\variables'
regular_path = 'C:\\Users\\LibraryUser\\Desktop\\IPP\\regularGP\\1\\variables'
plot_path = 'C:\\Users\\LibraryUser\\Desktop\\IPP\\figs'

# Create the plots directory if it doesn't exist
os.makedirs(plot_path, exist_ok=True)

# List all files in the variables directories
# mixture_files = os.listdir(mixture_path)
# regular_files = os.listdir(regular_path)

# # Ensure that we're comparing the same variables by finding common files
# common_files = set(mixture_files).intersection(regular_files)

# # Iterate over each common variable file and plot the comparisons
# for file_name in common_files:
#     # Load the variables (assuming .npy format, change as needed)
#     mixture_data = np.load(os.path.join(mixture_path, file_name))
#     regular_data = np.load(os.path.join(regular_path, file_name))
    
#     # Plot the data
#     plt.figure(figsize=(10, 6))
#     plt.plot(mixture_data, label=f'MixtureGP {file_name}', marker='o')
#     plt.plot(regular_data, label=f'RegularGP {file_name}', marker='x')
#     plt.title(f'Comparison of {file_name} between MixtureGP and RegularGP')
#     plt.xlabel('Index')
#     plt.ylabel(file_name.split('.')[0].upper())  # Assuming the filename describes the variable
#     plt.legend()
#     plt.grid(True)
    
#     # Save the plot
#     plot_filename = os.path.join(plot_path, f'{file_name.split(".")[0]}_comparison.png')
#     plt.savefig(plot_filename)
#     plt.close()  # Close the figure to free up memory

#     print(f'Saved plot for {file_name} as {plot_filename}')



# List all files in the variables directories
mixture_files = os.listdir(mixture_path)
regular_files = os.listdir(regular_path)

# Ensure that we're comparing the same variables by finding common files
common_files = set(mixture_files).intersection(regular_files)
if 'iteration_array.npy' in common_files:
    iteration_array = np.load(os.path.join(mixture_path, 'iteration_array.npy'))
    common_files.remove('iteration_array.npy') 
# Assuming iteration_array is defined earlier in your script
# For example:
# iteration_array = np.arange(0, len(mixture_data))

# Define colors for the plots, assuming you have more than one robot or variable to plot
ROBOT_COLOR = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Extend this if needed

# Number of robots/variables
N = 3  # Adjust this according to your data

# Iterate over each common variable file and plot the comparisons
for file_name in common_files:
    # Load the variables (assuming .npy format, change as needed)
    mixture_data = np.load(os.path.join(mixture_path, file_name))
    regular_data = np.load(os.path.join(regular_path, file_name))
    
    # Plot the data
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    ax.set_xlabel("Iterations")
    ax.set_ylabel(file_name.split('.')[0].upper())  # Assuming the filename describes the variable
    
    # Assuming mixture_data and regular_data have the shape (iterations, N)
    for i in range(N):
        ax.plot(iteration_array, mixture_data[:, i], color=ROBOT_COLOR[i], linestyle='-', label=f'MixtureGP Robot {i+1}')
        ax.plot(iteration_array, regular_data[:, i], color=ROBOT_COLOR[i], linestyle='--', label=f'RegularGP Robot {i+1}')
    
    ax.legend()
    ax.grid(True)
    plt.title(f'Comparison of {file_name} between MixtureGP and RegularGP')

    # Save the plot
    plot_filename = os.path.join(plot_path, f'{file_name.split(".")[0]}_comparison.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the figure to free up memory

    print(f'Saved plot for {file_name} as {plot_filename}')
