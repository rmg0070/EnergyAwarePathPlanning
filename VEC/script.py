import subprocess

# with open('output.log', 'w') as f: 
#     subprocess.run(['python', 'C:\\\\Users\\\\LibraryUser\\\\Desktop\\\\RegularIPP\\\\exceuteipp.py'], stdout=f, stderr=subprocess.STDOUT)

# # print (scikit-learn.__version__)
import numpy as np


z_phi = np.load('C:\\Users\\LibraryUser\\Desktop\\EnergyAwarePathPlanning\\Z_phi.npy')

print(z_phi.shape)