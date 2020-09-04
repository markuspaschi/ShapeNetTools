import os.path
import pickle
from PIL import Image
import numpy as np

directorypath = "/datasets_nas/mapa3789/Pixel2Mesh/HandToolsRendered/ShapeNetHandTools_V6/"
train_path = "/home/mapa3789/Master/ShapeNetTrainingSplitGenerator/train_list.txt"
test_path = "/home/mapa3789/Master/ShapeNetTrainingSplitGenerator/test_list.txt"

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

print("Loading files...")

dat_files = listFiles(directorypath, ".dat")
train_dat_files = np.genfromtxt(train_path, dtype='str').tolist()
test_dat_files = np.genfromtxt(test_path, dtype='str').tolist()
png_files = listFiles(directorypath, ".png")


if len(dat_files) == len(png_files):
   print("Success: Length of dat and png files match {}".format(len(dat_files)))
else:
   print("Error: Length of dat and png files do not match!")


print("Checking dat files...")

for i, file in enumerate(dat_files):
    with open(file, 'rb') as file:
        data = pickle.load(file)

        x,y = data.shape

        if(x < 10) or (y != 6):
            print("ERROR: wrong dimension: {} {}".format(x, file))


print("Checking png files...")

for i, file in enumerate(png_files):
    try:
      img = Image.open(file) # open the image file
      img.verify() # verify that it is, in fact an image
    except (IOError, SyntaxError) as e:
      print('Bad file:', filename) # print out the names of corrupt files

print("DONE")
