import numpy as np

def get_scenario_params(scenario_name, N):
    # Initialize parameters
    initial_energy = np.full(N, 100.0)
    initial_energy[1] = 60
    initial_energy[3] = 60
    robot_beta = np.array([0.5,0.5,0.5,0.5])
    robot_alpha = np.array([0.1,0.1,0.1,0.1])
    exp_name = "temp"
    exp_trail = "CMU"
    # Adjust parameters based on scenario
    if scenario_name == '1':
        print(scenario_name)
    elif scenario_name == '2':
        robot_beta[3] = 5 
    elif scenario_name == '3':
        robot_alpha[3] = 2 
    elif scenario_name == '4':
        initial_energy[1] = 70  # Adjust the second element
    elif scenario_name == '5':
        initial_energy = np.array([50.0, 50.0, 100.0, 50.0])
        robot_alpha = np.array([1, 1, 2, 1])
    else:
        print(' not defined..')
    
    return robot_alpha, robot_beta,initial_energy,exp_name,exp_trail


# Z_phi = np.load("C:\\Users\\rmg00\\Downloads\\AOC_IPP_python_v5 1\\AOC_IPP_python_v5\\Z_phi.npy")

# NameError("name 'array' is not defined")
# 
# array([1.85767141e-72, 2.07053314e-57, 4.22685692e-44, 1.58042897e-32,   
#     1.08231702e-22, 1.35755073e-14, 3.11874394e-08, 1.31227676e-03,  
#      1.00735011e+00, 3.08010167e+00, 3.49713531e+00]