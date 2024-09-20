import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import multivariate_normal
##########################################################################
# Paper ref:
#[1] Maria Santos, Udari Madhushani, Alessia Benevento,and Naomi Ehrich Leonard. Multi-robot learning and coverage of unknown spatial fields.
#    In 2021 International Symposium on Multi-Robot and Multi-Agent Systems (MRS), pages 137–145. IEEE, 2021
##########################################################################

# distance Calculation
def dist(x, y, pos):
    return math.sqrt(((pos[0]-x)**2) + ((pos[1]-y)**2))
 
# partitionFinder
def partitionFinder(ax, robotsPositions, envSize_X, envSize_Y, resolution, densityArray):
    hull_figHandles = []
    distArray = np.zeros(robotsPositions.shape[0])
    colorList = ["red","green","blue","black","grey","orange"]
    locations = [[] for _ in range(robotsPositions.shape[0])]
    robotDensity = [[] for _ in range(robotsPositions.shape[0])]
    locationsIdx = [[] for _ in range(robotsPositions.shape[0])]
    text_handles = []
    # for partition display 
    # partition display require finer resolution
    res = 0.5
    # res = 0.02
    x_global_values = np.arange(envSize_X[0], envSize_X[1] + res, res) 
    y_global_values = np.arange(envSize_Y[0], envSize_Y[1] + res, res)
    for i, x_pos in enumerate(x_global_values):
        for j, y_pos in enumerate(y_global_values):
            for r in range(robotsPositions.shape[0]):    
                distanceSq = (robotsPositions[r, 0] - x_pos) ** 2 + (robotsPositions[r, 1] - y_pos) ** 2
                #distArray[r] = abs(math.sqrt(distanceSq))
                distArray[r] = math.sqrt(distanceSq)
            minIndex = np.argmin(distArray)
            locations[minIndex].append([x_pos, y_pos])
            # minValue = np.min(distArray)
            # minIndices = np.where(distArray == minValue)[0]
            # for r in minIndices:
            #     locations[r].append([x_pos, y_pos])

    # There is no-builtin library in python that supports good visulaization for voronoi
    # Therefore, using convex hull to draw the boundary of each partition
    # It requires deleting previous boundaries and plotting new ones at every iteration
    # Need this object handles so we can remove previous boundaries before calling the partitionFinder function from main file
    for r in range(robotsPositions.shape[0]):
        robotsLocation = np.array(locations[r])
        if len(robotsLocation)!=0:
            hull = ConvexHull(robotsLocation)
            # Get the vertices of the convex hull
            x, y = robotsLocation[hull.vertices, 0], robotsLocation[hull.vertices, 1]
            # boundary_points = robotsLocation[hull.vertices]
            # # Extract x and y coordinates
            # x, y = boundary_points[:, 0], boundary_points[:, 1]
            hullHandle, =  ( ax.plot(x, y, marker='None', linestyle='-', color="black", markersize=6, linewidth =4))
            hull_figHandles.append(hullHandle)
        
    # for centroid calculation
    # centroid calculation can be done using lower resolution
    # Eq 1 from paper [1]
    # LocationalCost Equation: H(x, phi) = sum(h_i(x, phi)) = sum(integral_{V_i(x)} ||q - x_i||^2 * phi(q) dq) for i from 1 to N
    # Eq 2 from paper [1]
    # Cenroid Equation: c_i(x) = integral_{V_i(x)} q * phi(q) dq / integral_{V_i(x)} phi(q) dq
    x_global_values = np.arange(envSize_X[0], envSize_X[1] + resolution, resolution) 
    y_global_values = np.arange(envSize_Y[0], envSize_Y[1] + resolution, resolution)
    locations = [[] for _ in range(robotsPositions.shape[0])]
    for i, x_pos in enumerate(x_global_values):
        for j, y_pos in enumerate(y_global_values):
            for r in range(robotsPositions.shape[0]):    
                distanceSq = (robotsPositions[r, 0] - x_pos) ** 2 + (robotsPositions[r, 1] - y_pos) ** 2
                #distArray[r] = abs(math.sqrt(distanceSq))
                distArray[r] = abs(math.sqrt(distanceSq))
            minValue = np.min(distArray)
            minIndices = np.where(distArray == minValue)[0]
            for r in minIndices:
                locations[r].append([x_pos, y_pos])
                locationsIdx[r].append([i,j])
                robotDensity[r].append(densityArray[i,j])   


    Mass = np.zeros(robotsPositions.shape[0])
    C_x = np.zeros(robotsPositions.shape[0])
    C_y = np.zeros(robotsPositions.shape[0])
    locationalCost = 0
    
    for r in range(robotsPositions.shape[0]):
        Cx_r = 0
        Cy_r = 0
        Mass_r = 0
        locationInRobotRegion = np.array(locations[r])
        currentrobotLoc = robotsPositions[r]
        r_dens = robotDensity[r]  
        for pos in range(locationInRobotRegion.shape[0]):
            dens = resolution * resolution * r_dens[pos]   
            Mass_r += dens
            Cx_r += dens * locationInRobotRegion[pos, 0]
            Cy_r += dens * locationInRobotRegion[pos, 1]
            positionDiffSq = (locationInRobotRegion[pos, 0] - currentrobotLoc[0]) ** 2 + (locationInRobotRegion[pos, 1] - currentrobotLoc[1]) ** 2  
            # We are not integrating so this implementation includes resolution as well
            locationalCost += dens *resolution* (positionDiffSq - 1) 
        if(Mass_r!=0):
            Cx_r /= Mass_r
            Cy_r /= Mass_r
            C_x[r] = Cx_r
            C_y[r] = Cy_r
            Mass[r] = Mass_r
    return C_x, C_y, locationalCost, Mass,hull_figHandles,text_handles,locationsIdx,locations
    
    
# Function to generate random means within the given domain
def generate_random_means(num_distributions, domain):
    return np.random.uniform(domain[0], domain[1], size=(num_distributions, 2))

# Function to generate random covariance matrix with the same variance
def generate_covariance_matrix(variance):
    return variance * np.identity(2)

# Function to calculate density function value for the entire function
def density_function(x, y, means, variances):
    num_distributions = len(means)
    result = 0
    for i in range(num_distributions):
        covariance_matrix = generate_covariance_matrix(variances[i])
        result += gaussian_density(x, y, means[i], covariance_matrix)
    return result*1000

# Function to calculate density function value at a point (x, y) for a given Gaussian distribution
def gaussian_density(x, y, mean, covariance_matrix):
    rv = multivariate_normal(mean=mean, cov=covariance_matrix)
    return rv.pdf([x, y])

# plot mean and var using the given plot axes for pred_mean and pred_var
def plot_mean_and_var(X,Y,pred_mean,pred_var,Z_phi,pred_mean_fig=None,pred_var_fig=None):
  
    # print(mean_min,mean_max)
    # print(f"max value in pred_mean:{np.max(pred_mean)} and min value {np.min(pred_mean)}")
    pred_mean_fig.clf()
    ax_2d_mean = pred_mean_fig.add_subplot()
    contour_plot_mean = ax_2d_mean.pcolor(X, Y, pred_mean.reshape(X.shape[0],X.shape[1]), cmap='viridis',vmin = np.min(Z_phi),vmax = np.max(Z_phi))

    ax_2d_mean.set_title( 'predicted mean')
    ax_2d_mean.set_xlabel('X')
    ax_2d_mean.set_ylabel('Y')
    plt.colorbar(contour_plot_mean, ax= ax_2d_mean)

    pred_var_fig.clf()
    ax_2d_var = pred_var_fig.add_subplot()
    contour_plot_var = ax_2d_var.pcolor(X, Y, pred_var.reshape(X.shape[0],X.shape[1]), cmap='Reds',vmin = 0,vmax = 4.0)

    ax_2d_var.set_title( 'predicted std')
    ax_2d_var.set_xlabel('X')
    ax_2d_var.set_ylabel('Y')
    ax_2d_var.grid(False)
    plt.colorbar(contour_plot_var, ax= ax_2d_var)

def energy_depleted(current_energy,dist_travelled,robo_alpha,robo_beta,timestep):

    energy_depleted_per_meter = robo_alpha*timestep + robo_beta * dist_travelled 

    return energy_depleted_per_meter

def sigmoid(x, k=15, theta=0.6):
    return 1 / (1 + np.exp((-1)*k * (x - theta)))