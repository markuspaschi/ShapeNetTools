import os, sys, math
import argparse
from collections import defaultdict
import numpy as np


class ShapeNetTrainGenerator():

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

    # TODO: rework with pandas -> faster
    def split(self, rendering_images):

        cat_models = {}

        model_train_list = []
        model_test_list = []

        print("Generate model list...")

        # add categories and corresponding models
        for i in range(0, len(rendering_images)):
            currentItem = rendering_images[i]
            cat_models.setdefault(self.catFromPath(currentItem), []).append(currentItem)

        # check model size per category
        for cat, path in cat_models.items():
            # remove duplicate models
            model_amount = len(set([self.modelFromPath(x) for x in path]))
            if(math.ceil(model_amount < args.min_cat_train_size)):
                # we have not enough model for train set (in this cat)
                # add all to test set
                model_test_list += path
            else:
                train, test = self.split_list(path)
                model_train_list += train
                model_test_list += test

        train_list = [str for str in rendering_images if
                      any(sub in str for sub in model_train_list)]

        test_list = [str for str in rendering_images if
                      any(sub in str for sub in model_test_list)]

        train_path = os.path.join(args.output_path, 'train_list.txt')
        test_path = os.path.join(args.output_path, 'test_list.txt')

        print("Saving...")

        np.savetxt(train_path, train_list, fmt="%s")
        np.savetxt(test_path, test_list, fmt="%s")

        print("Saved to {}".format(os.path.abspath(args.output_path)))

        print("------")
        print("Training set: {}".format(len(train_list)))
        print("Testing set: {}".format(len(test_list)))


    def catFromPath(self, path):
        cat_dir = os.path.abspath(os.path.join(path, os.pardir, os.pardir, os.pardir))
        cat = os.path.split(os.path.abspath(cat_dir))[1]
        return cat

    def modelFromPath(self, path):
        model_dir = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
        model = os.path.split(os.path.abspath(model_dir))[1]
        return model

    def split_list(self, models):
        half = int(round(len(models)*args.train_ratio))
        return models[:half], models[half:]


def main():
    gen = ShapeNetTrainGenerator()
    rendering_images = gen.listFiles(args.dataset, ".dat")
    gen.split(rendering_images)



def get_args():
    global args

    parser = argparse.ArgumentParser(description='Generates a train and test split.')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Defaults to 0.8 (0.8 train files / 0.2 test files).')
    parser.add_argument('--min_cat_train_size', type=float, default=10, help='Minimum amount of models for splitting per category')
    parser.add_argument('--output_path', type=str, default="./")


    args = parser.parse_args()

def validate():
    if not (os.path.exists(args.dataset)):
        sys.exit("DataSet path does not exist")

if __name__ == "__main__":
    get_args()
    validate()
    main()
