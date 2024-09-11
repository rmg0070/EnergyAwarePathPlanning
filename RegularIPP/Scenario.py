import numpy as np

def get_scenario_params(scenario_name, N):
    # Initialize parameters
    weights = np.ones(N)
    initial_energy = np.full(N, 100.0)
    # initial_energy[1] = 70
    initial_energy[1] = 60
    initial_energy[3] = 60
    robot_beta = np.array([0.5,0.5,0.5,0.5])
    robot_alpha = np.array([0.1,0.1,0.1,0.1])
    exp_name = "temp"
    exp_trail = "1"
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