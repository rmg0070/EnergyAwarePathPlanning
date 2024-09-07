import numpy as np
import matplotlib.pyplot as plt

def plot_rmse(ax_rmse, iteration_array, rmse_array, label, color):
    """
    Plots the RMSE on the given axis.

    Args:
        ax_rmse: The matplotlib axis to plot on.
        iteration_array: The array of iteration values.
        rmse_array: The RMSE values to plot.
        label: Label for the plot legend.
        color: Color for the plot line.
    """
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_xlabel("Iterations")
    ax_rmse.plot(iteration_array, rmse_array, color=color, label=label)
    ax_rmse.legend()
    ax_rmse.grid(True)

def plot_variance(ax_var, iteration_array, variance_array, label, color):
    """
    Plots the Variance on the given axis.

    Args:
        ax_rmse: The matplotlib axis to plot on.
        iteration_array: The array of iteration values.
        rmse_array: The RMSE values to plot.
        label: Label for the plot legend.
        color: Color for the plot line.
    """
    ax_var.set_ylabel("Variance")
    ax_var.set_xlabel("Iterations")
    ax_var.plot(iteration_array, variance_array, color=color, label=label)
    ax_var.legend()
    ax_var.grid(True)

iteration_array = np.load('C:\\Users\\LibraryUser\\Desktop\\Proposed_Approach\\AOC_IPP_python_v5\\temp\\1\\variables\\iteration_array.npy')

rmse_array_proposed = np.load('C:\\Users\\LibraryUser\\Desktop\\Proposed_Approach\\AOC_IPP_python_v5\\temp\\1\\variables\\rmse_array.npy')
rmse_array_eac = np.load('C:\\Users\\LibraryUser\\Desktop\\EnergyAware\\temp\\1\\variables\\rmse_array.npy')
rmse_array_cmu = np.load('C:\\Users\\LibraryUser\\Downloads\\CMU\\temp\\1\\variables\\rmse_array.npy')
rmse_array_vec =  np.load('C:\\Users\\LibraryUser\\Desktop\\VEC\\temp\\1\\variables\\rmse_array.npy')

fig_rmse = plt.figure()
ax_rmse = fig_rmse.add_subplot()

# Plot the RMSE arrays from both folders

plot_rmse(ax_rmse, iteration_array, rmse_array_proposed, label='Propsed', color='red')
plot_rmse(ax_rmse, iteration_array, rmse_array_eac, label='EAC', color='green')
plot_rmse(ax_rmse, iteration_array, rmse_array_cmu, label='MRIP', color='blue')
plot_rmse(ax_rmse, iteration_array, rmse_array_vec, label='VEC', color='black')


# plt.show()




var_array_proposed = np.load('C:\\Users\\LibraryUser\\Desktop\\Proposed_Approach\\AOC_IPP_python_v5\\temp\\1\\variables\\variance_array.npy')
var_array_eac = np.load('C:\\Users\\LibraryUser\\Desktop\\EnergyAware\\temp\\1\\variables\\variance_array.npy')  # Uncomment if needed
var_array_cmu = np.load('C:\\Users\\LibraryUser\\Downloads\\CMU\\temp\\1\\variables\\variance_array.npy')
var_array_vec = np.load('C:\\Users\\LibraryUser\\Desktop\\VEC\\temp\\1\\variables\\variance_array.npy')


fig_var = plt.figure()
ax_var = fig_var.add_subplot()


# plot_variance(ax_var, iteration_array, var_array_regipp, label='RegIpp ', color='black')
plot_variance(ax_var, iteration_array, var_array_proposed, label='Propsed', color='red')
plot_variance(ax_var, iteration_array, var_array_eac, label='EAC', color='green')
plot_variance(ax_var, iteration_array, var_array_cmu, label='MRIP', color='Blue')
plot_variance(ax_var, iteration_array, var_array_vec, label='VEC', color='darkviolet')



file_paths = [
    "C:\\Users\\LibraryUser\\Desktop\\Proposed_Approach\\AOC_IPP_python_v5\\temp\\1\\variables\\cumulative_dist_array.npy",  # EAC
    "C:\\Users\\LibraryUser\\Desktop\\EnergyAware\\temp\\1\\variables\\cumulative_dist_array.npy",  # CMU
    "C:\\Users\\LibraryUser\\Downloads\\CMU\\temp\\1\\variables\\cumulative_dist_array.npy",
    "C:\\Users\\LibraryUser\\Desktop\\VEC\\temp\\1\\variables\\cumulative_dist_array.npy"  # VEC
        # VEC
]


label_array = ["Proposed", "EAC", "MRIP", "VEC"]
linestyle_array = ['-', '--', '-.', ':', 'solid']
linewidth_array = [2, 2, 2, 2, 2]
marker_array = ['o', 's', '^', 'd', '*']  # Adding distinct markers for each line
ROBOT_COLOR = {0: "red", 1: "green", 2: "blue", 3: "black", 4: "darkmagenta"}

fig, ax = plt.subplots()
for idx, file in enumerate(file_paths):
    try:
        # Load the cumulative distance array for each file
        cumulative_dist_array = np.load(file)
        
        # Calculate the mean and standard deviation across all iterations and robots
        cumulative_dist_mean = np.mean(cumulative_dist_array, axis=1)
        cumulative_dist_std = np.std(cumulative_dist_array, axis=1)
        
        # Generate an iteration array for the x-axis, using the number of iterations
        it_array = np.arange(cumulative_dist_array.shape[0])
        
        # Plotting with distinct style, color, and markers
        ax.plot(it_array, cumulative_dist_mean, 
                label=label_array[idx], 
                color=ROBOT_COLOR[idx], 
                linestyle=linestyle_array[idx], 
                linewidth=linewidth_array[idx],
                marker=marker_array[idx],  # Adding marker to differentiate
                markevery=5)  # Marker interval
        
        ax.fill_between(it_array, 
                        cumulative_dist_mean - cumulative_dist_std, 
                        cumulative_dist_mean + cumulative_dist_std, 
                        color=ROBOT_COLOR[idx], 
                        alpha=0.2)
    except FileNotFoundError:
        print(f"File not found: {file}")
    except Exception as e:
        print(f"Error processing file {file}: {e}")

# Setting the labels, title, and legend
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean Cumulative Distance')
ax.set_title('Mean Cumulative Distance and Variance')
ax.legend(loc='upper right')

# plt.show()

fig_var.savefig('C:\\Users\\LibraryUser\\Downloads\\CMU\\temp\\variance.png')
fig_rmse.savefig('C:\\Users\\LibraryUser\\Downloads\\CMU\\temp\\rmse.png')
fig.savefig('C:\\Users\\LibraryUser\\Downloads\\CMU\\temp\\cumulative.png')