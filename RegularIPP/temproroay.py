import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from utilityFunctions_AOC_IPP import *
from gp import GP
from regGP import GP as RegGP
from Scenario import *
import os
import sys
sys.path.append('C:\\Users\\LibraryUser\\Downloads\\AOC_IPP_python_v5\\AOC_IPP_python_v5\\script')
# from script import Gp_mixture
from Gp_mixture import MixtureGaussianProcess


# model = MixtureGaussianProcess(4,[GP(sigma_f=np.var(Z_phi)) for i in range(4)],[GP(sigma_f=np.var(Z_phi)) for i in range(4)])
##################################################################################
#Paper ref:
#[1] Maria Santos, Udari Madhushani, Alessia Benevento, and Naomi Ehrich Leonard. Multi-robot learning and coverage of unknown spatial fields. 
#In 2021 International Symposium on Multi-Robot and Multi-Agent Systems (MRS), pages 137–145. IEEE, 2021.
##################################################################################

def executeIPP_py(scenario_number,N=6,resolution=0.1, number_of_iterations=20, save_fig_flag=True,):
    distance_to_centroid_threshold= -0.1
    file_path = ""
    ROBOT_COLOR = {0: "red", 1: "green", 2: "blue", 3:"black",4:"grey",5:"orange"}
    x_min, x_max = 0.0, 20.0
    y_min, y_max = 0.0, 20.0   

    robot_alpha,robot_beta,initial_robots_energy,exp_name,exp_trail = get_scenario_params(scenario_number,N)
    file_path = f"{exp_name}\\{exp_trail}"
    # generate random initial values
    current_robotspositions =  np.random.uniform(x_min, x_max, size=(2, N))
    Model = "MiGp"
    current_robotspositions = np.array([
        [19.71047901, 3.34092571],
        [14.71047901, 14.34092571],
        [2.33815788, 3.87953727],
        [5, 12]
    ]).T
 

    # to show interactive plotting
    current_robot_energy = initial_robots_energy
    max_energy = np.max(current_robot_energy)
    energy_percentage = (current_robot_energy / max_energy)

    plt.ion()
    main_fig = plt.figure()
    main_axes = main_fig.add_subplot()
    pred_mean_fig = plt.figure()
    pred_var_fig = plt.figure()
    boundary_points = [[x_min,y_min],[x_min,y_max],[x_max,y_max],[x_max,y_min]] 
    bound_x, bound_y = zip(*boundary_points) 
    cumulative_dist_array = np.zeros((number_of_iterations,N))
    cumulative_energy_array = np.zeros((number_of_iterations,N))
    centroid_dist_array = np.zeros((number_of_iterations,N))
    areaArray = np.zeros((number_of_iterations,N))
    cumulative_distance = np.zeros(N)
    rmse_array = np.zeros(number_of_iterations)
    variance_array = np.zeros(number_of_iterations)
    
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
    coeff_sampling_overtime = np.zeros((number_of_iterations,N))
    coeff_recharge_overtime = np.zeros((number_of_iterations,N))
    coeff_mean = np.zeros((number_of_iterations,N))
    coeff_var = np.zeros((number_of_iterations,N))

    # generate 9 Gauusian distribution for ground_truth
    # Using Z_phi here to represent ground_truth (phi(q))
    # Number of Gaussian distributions
    num_distributions = 9
    # Variances for all distributions
    variances = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\variances.npy")
    # variances = np.ones(num_distributions) * 10.0 # Adjusted variance for visibility

    # Generate random means for both density functions
    # means_phi = generate_random_means(num_distributions, (x_min,x_max))
    # np.save("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\"+"means_phi.npy",means_phi)
    means_phi =  np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\means_phi.npy")
    Z_phi = np.load("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\Z_phi.npy")


    means_min = np.min(means_phi)
    means_max = np.max(means_phi)
    # np.save(os.path.join(file_path, f"means_phi.npy"),means_phi)

    # Create a grid of points for plotting
    x_vals = np.arange(x_min, x_max + resolution, resolution) 
    y_vals = np.arange(y_min, y_max + resolution, resolution)
    X = np.zeros((len(x_vals), len(y_vals)))
    Y = np.zeros((len(x_vals), len(y_vals)))
    points_grid_x = np.ones((len(x_vals), len(y_vals)))
    # Fill X and Y arrays using for loops
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            # print(x,y,i,j)
            X[i, j] = x
            Y[i, j] = y
    # Z_phi = np.vectorize(lambda x, y: density_function(x, y, means_phi, variances))(X, Y)
    # np.save("C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\"+"Z_phi.npy",Z_phi)
    print(Z_phi.shape)
    print(f"max value in pred_mean:{np.max(Z_phi)} and min value {np.min(Z_phi)}")
    test_X = np.column_stack((X.flatten(),Y.flatten()))
    # print(test_X.shape)
    # plot the density to main fig to get an idea of where robots are going
    # can comment this line to turn off
    # main_axes.pcolor(X, Y, Z_phi,shading="auto")
    modeling_gps = []
    gating_gps = []
    for i in range(4):
        modeling_gps.append(GP(0.5, 0.5, 0.1))
        gating_gps.append(GP(0.5, 0.5, 0.1))

    contour_plot_mean = main_axes.contourf(X, Y, Z_phi.reshape(X.shape[0],X.shape[1]), cmap='viridis')
    plt.colorbar(contour_plot_mean, ax= main_axes)
    train_X = np.transpose(current_robotspositions)
    Z_phi_current_pos = np.vectorize(lambda x, y: density_function(x, y, means_phi, variances))(current_robotspositions[0,:], current_robotspositions[1,:]) + np.random.uniform(-2,2,size=N)
    train_Y = Z_phi_current_pos.reshape(train_X.shape[0],1)
    # /\print(np.var(Z_phi))
    # model = GP(sigma_f=np.var(Z_phi))

    # model = MixtureGaussianProcess(4,modeling_gps,gating_gps,0.03,200)

    # model.train(train_X,train_Y)
    # model.AddSample(train_X,train_Y)
    # pred_mean, pred_var = model.predict(test_X)
    # pred_mean, pred_var = model.Predict(test_X)
    if Model == "MixGp":
        model = MixtureGaussianProcess(4,modeling_gps,gating_gps,0.03,200)
        # model = MixtureGaussianProcess(4,[GP(sigma_f=np.var(Z_phi)) for i in range(4)],[GP(sigma_f=np.var(Z_phi)) for i in range(4)])
        train_Y = Z_phi_current_pos.reshape(-1)
        
        model.AddSample(train_X,train_Y)
        pred_mean, pred_var = model.Predict(test_X)
    else:
        model = RegGP(sigma_f=np.var(Z_phi))
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
        [main_axes.scatter(current_robotspositions[0, i], current_robotspositions[1, i], c=ROBOT_COLOR[(i+1) % len(ROBOT_COLOR)], s=60, marker="x", linewidths=1) for i in range(N)]

        if iteration > 0:
            [plot.remove() for plot in positions_plots]

        # Line plot for the robot trajectories with unique colors
        positions_plots = [main_axes.plot(positions_array[:iteration+1, 0, i], positions_array[:iteration+1, 1, i], color=ROBOT_COLOR[(i+1) % len(ROBOT_COLOR)])[0] for i in range(N)]
        


        max_energy = np.max(current_robot_energy)
        energy_percentage = (current_robot_energy / max_energy)
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
        # main_axes.set_xlim([-1.2,1.2])
        C_x, C_y , cost, area, global_hull_figHandles, global_hull_textHandles,locationIdx,locationx = partitionFinder(main_axes,np.transpose(current_robotspositions[:2,:N]), [x_min,x_max], [y_min,y_max], resolution, pred_mean.reshape(Z_phi.shape)) 
        # main_axes.set_ylim([-1.2,1.2])e
        # 
        # C_x, C_y , cost, area, global_hull_figHandles, global_hull_textHandles,locationIdx,locationx = partitionFinder(main_axes,np.transpose(current_robotspositions[:2,:N]), [x_min,x_max], [y_min,y_max], resolution,surrogate_mean) 
        train_X = np.transpose(current_robotspositions)
        Z_phi_current_pos = np.vectorize(lambda x, y: density_function(x, y, means_phi, variances))(current_robotspositions[0,:], current_robotspositions[1,:]) + np.random.uniform(-2,2,size=N)
        # Beta for Eq 9 - surrogate_mean from [1]
        beta_val = 2*math.log( discretization/(6*0.7))
        beta_val_array[iteration] = beta_val
        rt_array[iteration] = r_t
        train_Y = Z_phi_current_pos.reshape(train_X.shape[0],1)
        # if iteration>0:
        #     model.AddSample(train_X,train_Y)
        #     # model.train(train_X,train_Y)
        # # pred_mean, pred_var = model.predict(test_X)
        if iteration>0:
            if Model == "MixGp":
                train_Y = Z_phi_current_pos.reshape(-1)

                model.AddSample(train_X,train_Y)
                pred_mean, pred_var = model.Predict(test_X)
            else:
                model.train(train_X,train_Y)
                pred_mean, pred_var = model.predict(test_X)
        pred_var = pred_var.reshape(X.shape[0],X.shape[1])
        print(f"shape of trainx {train_X.shape},trainY {train_Y.shape}")
        print(f"pred var{pred_var}")
        pred_std = np.sqrt(pred_var)
        plot_mean_and_var(X,Y,pred_mean,pred_std.reshape(pred_mean.shape),Z_phi,pred_mean_fig=pred_mean_fig,pred_var_fig=pred_var_fig)
        # Eq 9 from paper [1]
        # phi^(t)(q) = mu^(t-1)(q) - sqrt(beta^(t)) * sigma^(t-1)(q), for all q in D
        surrogate_mean = copy.deepcopy(pred_mean) - (math.sqrt(beta_val)*pred_std.reshape(pred_mean.shape))
        
        for robot in range(N):
            location_ids = np.array(locationIdx[robot])
            locations = np.array(locationx[robot])

           
            beta = 0.25+(0.5)*sigmoid(energy_percentage[robot], k=15, theta=0.6)
            alpha = 1.0 - beta  
            # print(f"beta for {robot} is {beta}")
            # print(f"alpha for {robot} is {alpha}")

            coeff_mean[iteration,robot] = alpha
            coeff_var[iteration,robot] = beta

            if len(location_ids) != 0:
                var_in_voronoi_region = (pred_var[location_ids[:, 0], location_ids[:, 1]])
                counts_in_voronoi_region = (points_grid_x[location_ids[:, 0], location_ids[:, 1]])
                pred_mean_reshaped = pred_mean.reshape(X.shape[0],X.shape[1])
                mean_in_voronoi_region = (pred_mean_reshaped[location_ids[:, 0], location_ids[:, 1]])
                weighted_sampling_goal = alpha * mean_in_voronoi_region + beta * var_in_voronoi_region
                # idx_with_max_weight = np.argmax(weighted_sampling_goal)
                idx_with_max_weight = np.argmax(weighted_sampling_goal/counts_in_voronoi_region)
                Sampling_goal = location_ids[idx_with_max_weight]
                points_grid_x[Sampling_goal[0],Sampling_goal[1]] = points_grid_x[Sampling_goal[0],Sampling_goal[1]] +1
                
                if robot==0:
                    # sampling_goal[:,robot] = locations[idx_with_max_std] 
                    sampling_goal[:,robot] = locations[idx_with_max_weight] 
                else:   
                    # a quick fix for robots getting same position for sampling
                    # it can happen because they share same voronoi boundary
                    # check with other robots if they have similar positions
                    # In case of similar positions, assign a larger negative value to that standard deviation and then recalculate the position
                    similar_goal = True
                    while similar_goal:
                        sampling_goal[:,robot] = locations[idx_with_max_weight]
                        # sampling_goal[:,robot] = locations[idx_with_max_std] 
                        similar_goal = np.any(np.all(sampling_goal[:,:robot] == (sampling_goal[:,robot]).reshape(2,1), axis=0))
                        if similar_goal==False:
                            break           
                        weighted_sampling_goal[idx_with_max_weight] = -10000
                        idx_with_max_weight = np.argmax(weighted_sampling_goal)

        locational_cost[iteration] = cost
        print(sampling_goal)
        if iteration>0:
            regret_array[iteration] = cost - np.min(locational_cost[:iteration])
        centroid = (np.array([C_x,C_y]))
        # Eq 11 from [1]
        # Control law to find x_i dot : \dot{x}_i^(t) = (1 - gamma) * c_i^(t) + gamma * e_i^(t), for i in {1, ..., N}
        coeff_sampling = np.zeros(N)
        coeff_recharge = np.zeros(N)
        for bot in range(N):
            coeff_sampling[bot] = sigmoid(energy_percentage[bot], k=15, theta=0.25)
            coeff_recharge[bot] = 1.0 - coeff_sampling[bot]
            coeff_sampling_overtime[iteration,bot] = sigmoid(energy_percentage[bot], k=15, theta=0.25)
            coeff_recharge_overtime[iteration,bot] = 1.0 - coeff_sampling_overtime[iteration,robot]
        print(f'sampling {coeff_sampling}')
        # robotsPositions_diff = ((coeff_sampling)*sampling_goal)  + (coeff_recharge*recharge_goal) - current_robotspositions[:2,:]
        # robotsPositions_diff = ((1-r_t)*centroid)  + (r_t*sampling_goal) - current_robotspositions[:2,:]
        robotsPositions_diff = (sampling_goal) - current_robotspositions[:2,:]

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
                cumulative_energy_array[iteration,robot] = energy_spent_meter[iteration,robot]
                current_robot_energy[robot] = current_robot_energy[robot]  - energy_spent_meter[iteration,robot]
                battery_levels_over_time[iteration,robot]=current_robot_energy[robot]
            else:
                battery_levels_over_time[iteration,robot]=current_robot_energy[robot]
        # Equation: RMSE = sqrt(mean((y_true - y_pred)**2))
        rmse = np.sqrt(np.mean(np.square(Z_phi.flatten() - pred_mean.flatten())))
        rmse_array[iteration] = rmse

        variance_metric = np.mean(pred_var.flatten())
        variance_array[iteration] = variance_metric
        positions_last_timeStep = copy.deepcopy(current_robotspositions)
        # Currently the next positions of robots are calculated using sampling goal only - 
        # based on max std in robot's current partition
        current_robotspositions = copy.deepcopy(sampling_goal) 
        os.makedirs(file_path+"\\coverage\\", exist_ok=True) 
        main_fig.savefig(file_path+f"\\coverage\\{iteration}.png")
        plt.pause(2)

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

        fig_var = plt.figure()
        ax_var = fig_var.add_subplot()
        ax_var.set_ylabel("Variance")
        ax_var.set_xlabel("Iterations")
        rmse_plot = ax_var.plot(iteration_array,variance_array, color="black")

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

        fig_coeff_mean = plt.figure()
        ax_coeff_mean = fig_coeff_mean.add_subplot()
        ax_coeff_mean.set_ylabel("alpha")
        ax_coeff_mean.set_xlabel("Iterations")
        mean_plot = [ax_coeff_mean.plot(iteration_array, coeff_mean[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        ax_coeff_mean.legend()

        # Plot coeff_recharge
        fig_coeff_var = plt.figure()
        ax_coeff_var = fig_coeff_var.add_subplot()
        ax_coeff_var.set_ylabel("beta")
        ax_coeff_var.set_xlabel("Iterations")
        var_plot = [ax_coeff_var.plot(iteration_array, coeff_var[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        ax_coeff_var.legend()
        #saveFigs

        #plot cummilate_energy
        fig_cumulative_energy = plt.figure()
        ax_cumulative_ene = fig_cumulative_energy.add_subplot()
        ax_cumulative_ene.set_xlabel("Iterations")
        ax_cumulative_ene.set_ylabel("Cumulative Energy")
        cum_dis_plot = [ax_cumulative_ene.plot(iteration_array,cumulative_energy_array[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        ax_cumulative_ene.legend()

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
                "current_robot_energy":battery_levels_over_time,
                "coeff_sampling": coeff_sampling,
                "coeff_recharge": coeff_recharge,
                "robots_positons":positions_array,
                "Coeff_mean":coeff_mean,
                "Coeff_var":coeff_var,
                "variance_array":variance_array
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
            fig_coeff_mean.savefig(file_path+"\\alpha.png")
            fig_coeff_var.savefig(file_path+"\\beta.png")
            pred_mean_fig.savefig(file_path+"\\GPmean.png")
            pred_var_fig.savefig(file_path+"\\GPvar.png")
            fig_var.savefig(file_path+"\\variance.png")
        plt.show()
        plt.pause(1)
      
    return 


if __name__=="__main__":
    N = 4
    scenario_number = '1'
    executeIPP_py(scenario_number,N ,resolution=1.0,number_of_iterations=50)     

