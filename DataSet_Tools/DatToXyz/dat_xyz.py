import os.path
import numpy as np
import pickle

dir = ""

def listFiles(dir, ext, ignoreExt=None):
    """
    Return array of all files in dir ending in ext but not ignoreExt.
    """
    matches = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            if f.endswith(ext):
                if not ignoreExt or (ignoreExt and not f.endswith(ignoreExt)):
                    matches.append(os.path.join(root, f))
    return matches


dat_files = listFiles(dir, ".dat")

length = len(dat_files)
for idx, dat in enumerate(dat_files):
    if(idx % 50 == 0):
        print(idx, "/", length)

    if(os.path.exists(dat.replace(".dat", ".xyz"))):
        continue

    with open(dat, 'rb') as file:
        data = pickle.load(file)
        np.savetxt(dat.replace(".dat", ".xyz"), data, delimiter =' ')

print("Finished")
