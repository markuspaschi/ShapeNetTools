import sys,os, shutil, pickle
import numpy as np
import subprocess
import logging
from subprocess import Popen

logging.basicConfig( level=logging.DEBUG, filename='cache/downloaded_models.txt')

BASE_PATH = "/home/mapa3789/Master/Pixel2Mesh/"
#TRAIN_PATH = BASE_PATH + "Data/train_list.txt"
TRAIN_PATH = BASE_PATH + "Data/test_list.txt"
DATASET_PATH = BASE_PATH + "Data/ShapeNetP2M"


class CacheManager(object):
    def save_obj(obj, name):
        if not os.path.exists("cache"):
            os.makedirs("cache")

        with open('cache/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(name):
        if not os.path.isfile('cache/' + name + '.pkl'):
            return None

        with open('cache/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def load_txt_to_list(name):
        fp = 'cache/' + name + '.txt'
        if not os.path.isfile(fp):
            return None

        return np.genfromtxt(fp,dtype='str').tolist()

def load_txt_to_list(fp):
    if not os.path.isfile(fp):
        return None

    return np.genfromtxt(fp,dtype='str').tolist()


def load_current_models():
    files_to_convert = []
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith(".dat"):
                files_to_convert.append(os.path.join(root,file).replace(BASE_PATH,''))

    return files_to_convert

def chunkArray(self, array, num):
    avg = len(array) / float(num)
    out = []
    last = 0.0

    while last < len(array):
        out.append(array[int(last):int(last + avg)])
        last += avg

    return out

def download_single_obj_file(filepath):

    tail = os.path.split(os.path.split(filepath)[-2])[-2]

    category = os.path.split(os.path.split(tail)[-2])[1]
    obj_name = os.path.split(tail)[-1]

    subprocess.call(["wget", "-q", "-r", "-np", "-nH", "--cut-dirs=3", "-R", "index.html*",
             "http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/" + category + "/" + obj_name + "/"])

    path = "./" + category + "/" + obj_name + "/"

    try:
        os.mkdir(path + "models/")
    except OSError:
        # The directory already existed, nothing to do
        pass

    try:
        subprocess.call("mv " + os.getcwd() + "/" + category + "/" + obj_name + "/* " + os.getcwd() + "/" + category + "/" + obj_name + "/models/", shell=True)
    except:
        pass

def download_single_file(filepath):
    tail = os.path.split(os.path.split(filepath)[-2])[-2]

    category = os.path.split(os.path.split(tail)[-2])[1]
    obj_name = os.path.split(tail)[-1]

    subprocess.call(["wget", "-q", "-r", "-np", "-nH", "--cut-dirs=3", "-R", "index.html*",
             "http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/" + category + "/" + obj_name + "/"])

def download_multiple_files(commands):
    procs = [ Popen(i) for i in commands ]
    for p in procs:
        p.wait()



def move_file(filepath):

    tail = os.path.split(os.path.split(filepath)[-2])[-2]

    category = os.path.split(os.path.split(tail)[-2])[1]
    obj_name = os.path.split(tail)[-1]

    path = "./" + category + "/" + obj_name + "/"

    try:
        os.mkdir(path + "models/")
    except OSError:
        # The directory already existed, nothing to do
        pass

    try:
        subprocess.call("mv " + os.getcwd() + "/" + category + "/" + obj_name + "/* " + os.getcwd() + "/" + category + "/" + obj_name + "/models/", shell=True)
#        subprocess.call("mv -vt " + os.getcwd() + "/" + category + "/" + obj_name + "/models/ " + os.getcwd() + "/" + category + "/" + obj_name + "/", shell=True)
    except:
        pass


# check in cache if we already have some xyz files already created -> dont create them again
# this can happen, if the script was killed or w/e
current_models = CacheManager.load_obj('current_models')
downloaded_models = CacheManager.load_txt_to_list('downloaded_models')

#already cached xyz models ->
if current_models is None:
    current_models = load_current_models()
    CacheManager.save_obj(current_models, "current_models")

train_list = load_txt_to_list(TRAIN_PATH)
diff = list(set(train_list) - set(current_models))
print("train_list: " + str(len(train_list)))
print("current_models: " + str(len(current_models)))
print("to_download (x5): " + str(len(diff)))

#already cached xyz models ->
if downloaded_models is not None:
    print("skipping {} already downloaded files".format(len(downloaded_models)))
    diff = [sent for sent in diff if not any(word[-36:-17] in sent for word in downloaded_models)]

print("before removing duplicates: " + str(len(diff)))
diff = [sent for sent in diff if not any(word in sent for word in ["01.dat", "02.dat", "03.dat", "04.dat"])]
print("after removing duplicates: " + str(len(diff)))

for index, file in enumerate(diff):
    print("{}/{}".format(index, str(len(diff))))
    download_single_obj_file(file)
    logging.info(file)


print("Finished")
