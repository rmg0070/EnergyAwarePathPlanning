import subprocess

with open('output.log', 'w') as f:
    subprocess.run(['python', 'C:\\Users\\LibraryUser\\Downloads\\AOC_IPP_python_v5\\AOC_IPP_python_v5\\executeIPP.py'], stdout=f, stderr=subprocess.STDOUT)

# print (scikit-learn.__version__)