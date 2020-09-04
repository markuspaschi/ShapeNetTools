import os, sys, math
import argparse
from collections import defaultdict
import numpy as np


class FindMissingDatPng():

    def __init__(self):
        pass

    def listFiles(self, dir, ext, ignoreExt=None):
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

    def find_missing_files(self, list_a, list_b):
        for file in list_a:
            if file.endswith(".png"):
                exists = os.path.exists(file.replace(".png", ".dat"))
                if not exists:
                    print("MISSING {}".format(file.replace(".png", ".dat")))
            else:
                exists = os.path.exists(file.replace(".dat", ".png"))
                if not exists: 
                    print("MISSING {}".format(file.replace(".dat", ".png")))


def main():
    gen = FindMissingDatPng()
    rendering_dats = gen.listFiles(args.dataset, ".dat")
    rendering_images = gen.listFiles(args.dataset, ".png")

    gen.find_missing_files(rendering_dats, rendering_images)
    gen.find_missing_files(rendering_images, rendering_dats)

def get_args():
    global args

    parser = argparse.ArgumentParser(description='Generates a train and test split.')
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()

def validate():
    if not (os.path.exists(args.dataset)):
        sys.exit("DataSet path does not exist")

if __name__ == "__main__":
    get_args()
    validate()
    main()
