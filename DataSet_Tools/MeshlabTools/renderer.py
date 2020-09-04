#!/usr/bin/env python

import os, sys, pickle, cv2, trimesh, sklearn.preprocessing, subprocess, math, glob
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

TO_RENDER = "/Users/markuspaschke/Desktop/Master_backup/TO_RENDER"
MESHLAB_PATH = "/Applications/meshlab.app/Contents/MacOS"
CURRENT_DIR = os.getcwd()

os.chdir(MESHLAB_PATH)

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

class P2MDataGenerator:

    def start(self):
        self.add_ambient_occlusion()

    def add_ambient_occlusion(self):

        objs = listFiles(TO_RENDER, "model.obj")

        with Parallel(n_jobs=5) as parallel:
            parallel(delayed(self.ambient_occlusion)(obj) for obj in objs)


    def ambient_occlusion(self, obj_path):
        done = os.path.join(os.path.dirname(obj_path), "done")
        if not (os.path.exists(done)):
            os.system('./meshlabserver -i %s -o %s -m vc vn fc wt -s %s ' %
                      (obj_path, obj_path, CURRENT_DIR + "/recompute_normals.mlx"))
            open(done, 'a').close()

if __name__ == "__main__":
    P2MDataGenerator().start()
