import rps.robotarium as Robotarium_
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from utilityFunctions_AOC_IPP import *
from GP import GP
from Scenario import *
import os

def executeIPP_py(scenario_number,N=6,resolution=0.1, number_of_iterations=20, save_fig_flag=True,):
    distance_to_centroid_threshold= -0.1
    file_path = ""
    ROBOT_COLOR = {0: "red", 1: "green", 2: "blue", 3:"black",4:"grey",5:"orange"}
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1    

    robot_alpha,robot_beta,initial_robots_energy,exp_name,exp_trail = get_scenario_params(scenario_number,N)
    file_path = f"{exp_name}\\{exp_trail}"
    # generate random initial values
    current_robotspositions =  np.random.uniform(x_min, x_max, size=(2, N))
    # to show interactive plotting
    current_robot_energy= initial_robots_energy

    plt.ion()
    main_fig = plt.figure()
    main_axes = main_fig.add_subplot()
    pred_mean_fig = plt.figure()
    pred_var_fig = plt.figure()
    boundary_points = [[x_min,y_min],[x_min,y_max],[x_max,y_max],[x_max,y_min]] 
    bound_x, bound_y = zip(*boundary_points) 
    cumulative_dist_array = np.zeros((number_of_iterations,N))
    centroid_dist_array = np.zeros((number_of_iterations,N))
    areaArray = np.zeros((number_of_iterations,N))
    cumulative_distance = np.zeros(N)
    rmse_array = np.zeros(number_of_iterations)
    positions_array = np.zeros((number_of_iterations,2,N))
    positions_last_timeStep = np.zeros((2,N)) 
    regret_array = np.ones(number_of_iterations)
    iteration_array = np.zeros(number_of_iterations)
    beta_val_array = np.ones(number_of_iterations)
    rt_array = np.ones(number_of_iterations)
    locational_cost = np.ones(number_of_iterations)
    current_position_marker_handle = []
    battery_levels_over_time =  np.zeros((number_of_iterations,N))
    energy_spent_meter = np.zeros((number_of_iterations,N))
    time_step = 1
    coeff_sampling = np.zeros((number_of_iterations,N))
    coeff_recharge = np.zeros((number_of_iterations,N))

    # generate 9 Gauusian distribution for ground_truth
    # Using Z_phi here to represent ground_truth (phi(q))
    # Number of Gaussian distributions
    num_distributions = 9
    # Variances for all distributions
    variances = np.ones(num_distributions) * 0.05  # Adjusted variance for visibility
    # Generate random means for both density functions
    means_phi = generate_random_means(num_distributions, (x_min,x_max))
    # Create a grid of points for plotting
    x_vals = np.arange(x_min, x_max + resolution, resolution) 
    y_vals = np.arange(y_min, y_max + resolution, resolution)
    X = np.zeros((len(x_vals), len(y_vals)))
    Y = np.zeros((len(x_vals), len(y_vals)))
    # Fill X and Y arrays using for loops
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            X[i, j] = x
            Y[i, j] = y
    Z_phi = np.vectorize(lambda x, y: density_function(x, y, means_phi, variances))(X, Y)
    test_X = np.column_stack((X.flatten(),Y.flatten()))
    # plot the density to main fig to get an idea of where robots are going
    # can comment this line to turn off
    main_axes.pcolor(X, Y, Z_phi,shading="auto")
    train_X = np.transpose(current_robotspositions)
    Z_phi_current_pos = np.vectorize(lambda x, y: density_function(x, y, means_phi, variances))(current_robotspositions[0,:], current_robotspositions[1,:])
    train_Y = Z_phi_current_pos.reshape(train_X.shape[0],1)
    model = GP(sigma_f=np.var(Z_phi))
    model.train(train_X,train_Y)
    pred_mean, pred_var = model.predict(test_X)
    # for the 1st iteration setting the surrogate_mean
    surrogate_mean = copy.deepcopy(pred_mean)   
    beta_val = 0
    r_t = 0
    #########################################################  Coverage Script
    goal_for_centroid = copy.deepcopy(current_robotspositions)
    sampling_goal = copy.deepcopy(current_robotspositions)
    recharge_goal = np.empty((2, N))
    for iteration in range(number_of_iterations):
        positions_array[iteration,:,:] = current_robotspositions[:2,:]
        [main_axes.scatter(current_robotspositions[0,:], current_robotspositions[1,:], c="green", s=60, marker="x", linewidths=1) for i in range(N)]
        if iteration>0:
            [plot.remove() for plot in positions_plots]
        positions_plots = [main_axes.plot(positions_array[:iteration+1, 0, i], positions_array[:iteration+1, 1, i], color="green")[0] for i in range(N)]
    	# r_t for Eq 11 - control law from [1]
        if iteration > 0:
            r_t = 1/iteration
            discretization = ((X.shape[0])*(X.shape[1])) * math.pi* math.pi* iteration *iteration 
        else: 
            r_t = 1
            discretization = ((X.shape[0])*(X.shape[1])) * math.pi* math.pi
        print(f"iteration:{iteration}")
        iteration_array[iteration] = iteration
        dist_to_centroid = np.ones((N))*5
        # remove boundaries for previous voronoi partition
        # before plotting on top of
        if(iteration>0):
            for robot_r in range(len(global_hull_figHandles)):
                current_position_marker_handle[robot_r].remove()
                hullObject = global_hull_figHandles[robot_r]
                hullObject.remove()
                
        current_position_marker_handle = [main_axes.scatter(current_robotspositions[0,:], current_robotspositions[1,:], edgecolors="red", facecolors='none', s=100, marker="o", linewidths=3) for i in range(N)]
        main_axes.set_xlim([-1.2,1.2])
        main_axes.set_ylim([-1.2,1.2])
        coeff_mean = 0.25+(0.5)*sigmoid(x, k=15, theta=0.6)
        coeff_var = 1.0 - coeff_mean
        pred_mean_reshaped = pred_mean.reshape(pred_var.shape)
        density_array = coeff_mean * pred_mean + coeff_var * pred_var.reshape(pred_mean.shape)
        # density_array = coeff_mean * pred_mean + coeff_var * pred_var
        density_array = density_array.reshape(Z_phi.shape)
        # C_x, C_y , cost, area, global_hull_figHandles, global_hull_textHandles,locationIdx,locationx = partitionFinder(main_axes,np.transpose(current_robotspositions[:2,:N]), [x_min,x_max], [y_min,y_max], resolution, density_array) 
        
        C_x, C_y , cost, area, global_hull_figHandles, global_hull_textHandles,locationIdx,locationx = partitionFinder(main_axes,np.transpose(current_robotspositions[:2,:N]), [x_min,x_max], [y_min,y_max], resolution, pred_mean.reshape(Z_phi.shape)) 
        train_X = np.transpose(current_robotspositions)
        Z_phi_current_pos = np.vectorize(lambda x, y: density_function(x, y, means_phi, variances))(current_robotspositions[0,:], current_robotspositions[1,:])
        # Beta for Eq 9 - surrogate_mean from [1]
        beta_val = 2*math.log( discretization/(6*0.7))
        beta_val_array[iteration] = beta_val
        rt_array[iteration] = r_t
        train_Y = Z_phi_current_pos.reshape(train_X.shape[0],1)
        if iteration>0:
            model.train(train_X,train_Y)
        pred_mean, pred_var = model.predict(test_X)
        pred_var = pred_var.reshape(X.shape[0],X.shape[1])
        pred_std = np.sqrt(pred_var)
        plot_mean_and_var(X,Y,pred_mean,pred_std.reshape(pred_mean.shape),pred_mean_fig=pred_mean_fig,pred_var_fig=pred_var_fig)
        # Eq 9 from paper [1]
        # phi^(t)(q) = mu^(t-1)(q) - sqrt(beta^(t)) * sigma^(t-1)(q), for all q in D
        surrogate_mean = copy.deepcopy(pred_mean) - (math.sqrt(beta_val)*pred_std.reshape(pred_mean.shape))
        for robot in range(N):
            location_ids = np.array(locationIdx[robot])
            locations = np.array(locationx[robot])
            if len(location_ids) != 0:
                std_in_voronoi_region = (pred_std[location_ids[:, 0], location_ids[:, 1]])
                idx_with_max_std = np.argmax(std_in_voronoi_region)        
                if robot==0:
                    sampling_goal[:,robot] = locations[idx_with_max_std] 
                else:   
                    # a quick fix for robots getting same position for sampling
                    # it can happen because they share same voronoi boundary
                    # check with other robots if they have similar positions
                    # In case of similar positions, assign a larger negative value to that standard deviation and then recalculate the position
                    similar_goal = True
                    while similar_goal:
                        sampling_goal[:,robot] = locations[idx_with_max_std] 
                        similar_goal = np.any(np.all(sampling_goal[:,:robot] == (sampling_goal[:,robot]).reshape(2,1), axis=0))
                        if similar_goal==False:
                            break           
                        std_in_voronoi_region[idx_with_max_std] = -10000
                        idx_with_max_std = np.argmax(std_in_voronoi_region)
        locational_cost[iteration] = cost
        if iteration>0:
            regret_array[iteration] = cost - np.min(locational_cost[:iteration])
        centroid = (np.array([C_x,C_y]))
        # Eq 11 from [1]
        # Control law to find x_i dot : \dot{x}_i^(t) = (1 - gamma) * c_i^(t) + gamma * e_i^(t), for i in {1, ..., N}
        for bot in range(N):
            coeff_sampling[iteration,bot] = sigmoid(x, k=15, theta=0.6)
            coeff_recharge[iteration,bot] = 1.0 - coeff_sampling[iteration,robot]
        # robotsPositions_diff = ((coeff_sampling)*sampling_goal)  + (coeff_recharge*recharge_goal) - current_robotspositions[:2,:]
        robotsPositions_diff = ((1-r_t)*centroid)  + (r_t*sampling_goal) - current_robotspositions[:2,:]
        for robot in range(N):
            areaArray[iteration,robot] = area[robot]
            dist_to_centroid[robot] = (math.sqrt((current_robotspositions[ 0,robot] - C_x[robot]) ** 2 + (current_robotspositions[1,robot] - C_y[robot]) ** 2))
            dist_to_centroid[robot] =  round(dist_to_centroid[robot], 2)
            centroid_dist_array[iteration,robot] = dist_to_centroid[robot]
            # find goal for balancing both centroid and sampling 
            # using  X(t+1) = x(t) + \dot{x}_i
            goal_for_centroid[:,robot] = (current_robotspositions[:2,robot] + np.round(robotsPositions_diff[:2,robot],decimals=2))
            if iteration>0:
                d = dist(positions_last_timeStep[0,robot], positions_last_timeStep[1,robot], (current_robotspositions[0,robot],current_robotspositions[1,robot]))
                cumulative_distance[robot] = cumulative_distance[robot] + d
                cumulative_dist_array[iteration,robot]=cumulative_distance[robot]
                energy_spent_meter[iteration,robot] = energy_depleted(current_robot_energy[robot],d,robot_alpha[robot],robot_beta[robot],time_step)
                current_robot_energy[robot] = current_robot_energy[robot]  - energy_spent_meter[iteration,robot]
                battery_levels_over_time[iteration,robot]=current_robot_energy[robot]
            else:
                battery_levels_over_time[iteration,robot]=current_robot_energy[robot]
        # Equation: RMSE = sqrt(mean((y_true - y_pred)**2))
        rmse = np.sqrt(np.mean(np.square(Z_phi.flatten() - pred_mean.flatten())))
        rmse_array[iteration] = rmse
        positions_last_timeStep = copy.deepcopy(current_robotspositions)
        # Currently the next positions of robots are calculated using sampling goal only - 
        # based on max std in robot's current partition
        current_robotspositions = copy.deepcopy(sampling_goal) 
        plt.pause(5)

    if(save_fig_flag):
        # Fig: Area
        fig_area = plt.figure()
        ax_area = fig_area.add_subplot()
        ax_area.set_ylabel("Area")
        ax_area.set_xlabel("Iterations")
        area_plot = [ax_area.plot(iteration_array,areaArray[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        ax_area.legend()

        fig_rmse = plt.figure()
        ax_rmse = fig_rmse.add_subplot()
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.set_xlabel("Iterations")
        rmse_plot = ax_rmse.plot(iteration_array,rmse_array, color="black")

        fig_regret = plt.figure()
        ax_regret = fig_regret.add_subplot()
        ax_regret.set_ylabel("Regret (t)")
        ax_regret.set_xlabel("Iterations")
        regret_plot = ax_regret.plot(iteration_array,regret_array, color="black")

        fig_beta_val = plt.figure()
        ax_beta_val = fig_beta_val.add_subplot()
        ax_beta_val.set_ylabel("Beta Val")
        ax_beta_val.set_xlabel("Iterations")
        beta_val_plot = ax_beta_val.plot(iteration_array,beta_val_array)

        fig_rt_val = plt.figure()
        ax_rt_val = fig_rt_val.add_subplot()
        ax_rt_val.set_ylabel("r_t")
        ax_rt_val.set_xlabel("Iterations")
        rt_val_plot = ax_rt_val.plot(iteration_array,rt_array)
        
        # Fig: Centroid
        fig_dis_centroid = plt.figure()
        ax_dis_centroid = fig_dis_centroid.add_subplot()
        ax_dis_centroid.set_xlabel("Iterations")
        ax_dis_centroid.set_ylabel("Centroid Distance")
        centroid_plot = [ax_dis_centroid.plot(iteration_array,centroid_dist_array[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        ax_dis_centroid.legend()
        
        # Fig: Cumulative Distance
        fig_cumulative_distance = plt.figure()
        ax_cumulative_dis = fig_cumulative_distance.add_subplot()
        ax_cumulative_dis.set_xlabel("Iterations")
        ax_cumulative_dis.set_ylabel("Cumulative Distance")
        cum_dis_plot = [ax_cumulative_dis.plot(iteration_array,cumulative_dist_array[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        ax_cumulative_dis.legend()

        # Fig: Cumulative Distance
        fig_cost = plt.figure()
        ax_cost = fig_cost.add_subplot()
        ax_cost.set_xlabel("Iterations")
        ax_cost.set_ylabel("Locational Cost")
        cum_dis_plot = ax_cost.plot(iteration_array,locational_cost, color = "black",label="locationalCost")
        
        # Fig: Energy spent per meter
        fig_energy_spent = plt.figure()
        ax_energy_spent = fig_energy_spent.add_subplot()
        ax_energy_spent.set_ylabel("Energy Spent")
        ax_energy_spent.set_xlabel("Iterations")
        energy_spent_plot = [ax_energy_spent.plot(iteration_array, energy_spent_meter[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        ax_energy_spent.legend()

        # Fig: current_robot_energy
        fig_current_energy = plt.figure()
        ax_current_energy = fig_current_energy.add_subplot()
        ax_current_energy.set_ylabel("Current Energy")
        ax_current_energy.set_xlabel("Iterations")
        current_energy_plot = [ax_current_energy.plot(iteration_array, battery_levels_over_time[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        ax_current_energy.legend()


        fig_coeff_sampling = plt.figure()
        ax_coeff_sampling = fig_coeff_sampling.add_subplot()
        ax_coeff_sampling.set_ylabel("Sampling Coefficient")
        ax_coeff_sampling.set_xlabel("Iterations")
        sampling_plot = [ax_coeff_sampling.plot(iteration_array, coeff_sampling[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        ax_coeff_sampling.legend()

        # Plot coeff_recharge
        fig_coeff_recharge = plt.figure()
        ax_coeff_recharge = fig_coeff_recharge.add_subplot()
        ax_coeff_recharge.set_ylabel("Recharge Coefficient")
        ax_coeff_recharge.set_xlabel("Iterations")
        recharge_plot = [ax_coeff_recharge.plot(iteration_array, coeff_recharge[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        ax_coeff_recharge.legend()
        #saveFigs
        if save_fig_flag:
            variables = {
                "iteration_array": iteration_array,
                "areaArray": areaArray,
                "rmse_array": rmse_array,
                "regret_array": regret_array,
                "beta_val_array": beta_val_array,
                "rt_array": rt_array,
                "centroid_dist_array": centroid_dist_array,
                "cumulative_dist_array": cumulative_dist_array,
                "locational_cost": locational_cost,
                "energy_spent_meter":energy_spent_meter,
                "current_robot_energy":current_robot_energy,
                "coeff_sampling": coeff_sampling,
                "coeff_recharge": coeff_recharge
            }
            file_path = f"{exp_name}\\{exp_trail}\\variables"
            os.makedirs(file_path, exist_ok=True) 
            for key, value in variables.items():
                if isinstance(value, np.ndarray):
                    filepath = os.path.join(file_path, f"{key}.npy")
                    np.save(filepath, value)
                else:
                    filepath = os.path.join(file_path, f"{key}.npy")
                    np.save(filepath, np.array(value))
            
            file_path = f"{exp_name}\\{exp_trail}\\figures"
            os.makedirs(file_path, exist_ok=True)
            fig_dis_centroid.savefig(file_path+"\\distance_to_centroid.png")
            fig_area.savefig(file_path+"\\area.png")
            fig_rmse.savefig(file_path+"\\rmse.png")
            fig_cumulative_distance.savefig(file_path+"\\cumulative_distance.png")
            main_fig.savefig(file_path+"\\coverage.png")
            fig_cost.savefig(file_path+"\\cost.png")
            fig_beta_val.savefig(file_path+"\\beta_val.png")
            fig_rt_val.savefig(file_path+"\\rt_val.png")
            fig_energy_spent.savefig(file_path+"\\energy_spent_per_meter.png")
            fig_current_energy.savefig(file_path+"\\current_robot_energy.png")
            fig_coeff_recharge.savefig(file_path+"\\coeff_recharge.png")
            fig_coeff_sampling.savefig(file_path+"\\coeff_sampling.png")
            
        plt.show()
        plt.pause(15)
      
    return 


if __name__=="__main__":
    N = 4
    scenario_number = '1'
    executeIPP_py(scenario_number,N ,resolution=0.1,number_of_iterations=5)     
