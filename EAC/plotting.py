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
