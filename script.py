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
# "C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\CMU\\variables\\rmse_array.npy"
iteration_array = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\CMU\\variables\\iteration_array.npy")
rmse_array_regipp = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\REG\\variables\\rmse_array.npy")
rmse_array_proposed = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\Proposed\\variables\\rmse_array.npy")

rmse_array_eac = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\EAC\\variables\\rmse_array.npy")

rmse_array_cmu = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\CMU\\variables\\rmse_array.npy")

rmse_array_vec = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\VEC\\variables\\rmse_array.npy")



iteration_array_regipp = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\REG4\\variables\\iteration_array.npy")
iteration_array_proposed = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\Proposed4\\variables\\iteration_array.npy")
iteration_array_eac = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\EAC4\\variables\\iteration_array.npy")
iteration_array_cmu = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\CMU4\\variables\\iteration_array.npy")
iteration_array_vec = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\VEC4\\variables\\iteration_array.npy")

# Corresponding RMSE arrays, also updated with '4' in folder names
rmse_array_regipp = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\REG4\\variables\\rmse_array.npy")
rmse_array_proposed = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\Proposed4\\variables\\rmse_array.npy")
rmse_array_eac = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\EAC4\\variables\\rmse_array.npy")
rmse_array_cmu = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\CMU4\\variables\\rmse_array.npy")
rmse_array_vec = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\VEC4\\variables\\rmse_array.npy")
fig_rmse = plt.figure()
ax_rmse = fig_rmse.add_subplot()
ax_rmse.set_xlim([0,20])

plot_rmse(ax_rmse, iteration_array_regipp, rmse_array_regipp, label='RegIpp ', color='black')
plot_rmse(ax_rmse, iteration_array_proposed, rmse_array_proposed, label='Propsed', color='red')
plot_rmse(ax_rmse, iteration_array_eac, rmse_array_eac, label='EAC', color='green')
plot_rmse(ax_rmse, iteration_array_cmu, rmse_array_cmu, label='MRIP', color='blue')
plot_rmse(ax_rmse, iteration_array_vec, rmse_array_vec, label='VEC', color='darkviolet')
ax_rmse.grid(None)
fig_rmse.savefig("rmse4.png")



var_array_regipp = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\REG4\\variables\\variance_array.npy")
var_array_proposed = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\Proposed4\\variables\\variance_array.npy")

var_array_eac = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\EAC4\\variables\\variance_array.npy")

var_array_cmu = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\CMU4\\variables\\variance_array.npy")

var_array_vec = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\VEC4\\variables\\variance_array.npy")



fig_var = plt.figure()
ax_var = fig_var.add_subplot()
ax_var.set_xlim([0,20])
plot_variance(ax_var, iteration_array_regipp, var_array_regipp, label='RegIpp', color='black')
plot_variance(ax_var, iteration_array_proposed, var_array_proposed, label='Proposed', color='red')
plot_variance(ax_var, iteration_array_eac, var_array_eac, label='EAC', color='green')
plot_variance(ax_var, iteration_array_cmu, var_array_cmu, label='MRIP', color='blue')
plot_variance(ax_var, iteration_array_vec, var_array_vec, label='VEC', color='darkviolet')
ax_var.grid(None)
fig_var.savefig("variance4.png")


file_paths = [
    "C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\Proposed4\\variables\\cumulative_dist_array.npy",
    "C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\REG4\\variables\\cumulative_dist_array.npy",
    "C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\CMU4\\variables\\cumulative_dist_array.npy",
    "C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\VEC4\\variables\\cumulative_dist_array.npy",
    "C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\EAC4\\variables\\cumulative_dist_array.npy"
]

label_array = ["Proposed", "IPP", "MRIP", "VEC", "EAC"]
linestyle_array = ['-', '--', '-.', ':', 'solid']
linewidth_array = [2, 2, 2, 2, 2]
marker_array = ['o', 's', '^', 'd', '*']  # Adding distinct markers for each line
ROBOT_COLOR = ["red", "green", "blue", "black", "darkmagenta"]

fig_cumu, ax = plt.subplots()
for idx, file in enumerate(file_paths):
    try:
        cumulative_dist_array = np.load(file)
        it_array = np.arange(cumulative_dist_array.shape[0])
        cumulative_dist_sum = np.sum(cumulative_dist_array, axis=1)
        cumulative_dist_std = np.std(cumulative_dist_array, axis=1)
        ax.plot(it_array, cumulative_dist_sum,
                label=label_array[idx],
                color=ROBOT_COLOR[idx],
                linestyle=linestyle_array[idx],
                linewidth=linewidth_array[idx],
                marker=marker_array[idx], 
                markevery=5) 

        ax.fill_between(it_array,
                        cumulative_dist_sum - cumulative_dist_std,
                        cumulative_dist_sum + cumulative_dist_std,
                        color=ROBOT_COLOR[idx],
                        alpha=0.2)
    except FileNotFoundError:
        print(f"File not found: {file}")
    except Exception as e:
        print(f"Error processing file {file}: {e}")
ax.set_xlim([0,20])
ax.set_xlabel('Iteration')
ax.set_ylabel('Sum Cumulative Distance')
ax.set_title('Sum of  Cumulative Distance and Variance')
ax.legend(loc='upper right')
fig_cumu.savefig("cumulative4.png")

plt.show()
