import subprocess

with open('output.log', 'w') as f: 
    subprocess.run(['python', 'C:\\Users\\LibraryUser\\Desktop\\RegularIPP\\exceuteipp.py'], stdout=f, stderr=subprocess.STDOUT)

# print (scikit-learn.__version__)
