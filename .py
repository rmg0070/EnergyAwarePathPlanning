import numpy as np
import matplotlib.pyplot as plt

# Paths to data files
file_paths = [
    "C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\Proposed4\\variables\\cumulative_dist_array.npy",
    "C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\REG4\\variables\\cumulative_dist_array.npy",
    "C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\CMU4\\variables\\cumulative_dist_array.npy",
    "C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\VEC4\\variables\\cumulative_dist_array.npy",
    "C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\EAC4\\variables\\cumulative_dist_array.npy"
]
# file_paths = [
#     "C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\temp\\VEC3\\variables\\cumulative_dist_array.npy"
# ]

label_array = ["Proposed", "IPP", "MRIP", "VEC", "EAC"]
linestyle_array = ['-', '--', '-.', ':', 'solid']
linewidth_array = [2, 2, 2, 2, 2]
marker_array = ['o', 's', '^', 'd', '*']  # Adding distinct markers for each line
ROBOT_COLOR = ["red", "green", "blue", "black", "darkmagenta"]

fig_cumu, ax = plt.subplots()
for idx, file in enumerate(file_paths):
    try:
        cumulative_dist_array = np.load(file)
        for i in range(1, cumulative_dist_array.shape[0]):
            for j in range(cumulative_dist_array.shape[1]):
                if cumulative_dist_array[i, j] == 0:
                    cumulative_dist_array[i, j] = cumulative_dist_array[i - 1, j]
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
ax.set_title('Sum Cumulative Distance and Variance')
ax.legend(loc='upper right')
fig_cumu.savefig("cumulative5.png")

plt.show()
