import os
import shutil
import glob
import numpy as np

# dir_to_check = "/Users/markuspaschke/Desktop/Master_backup/ShapeNetToolsPicked/ShapeNetHandToolsAO"
dir_to_check = "/Users/markuspaschke/Desktop/Master_backup/TO_RENDER/ShapeNetHandTools"

def listDirs(dir, names):
    """
    Return array of all files in dir ending in ext but not ignoreExt.
    """
    matches = []
    for root, dirs, files in os.walk(dir):
        for dir in dirs:
            if dir.endswith(names):
                matches.append(os.path.join(root, dir))
    return matches

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


def print_folders_without_model_file():
    #has to be in form of category/model_id/... - somewhere - .../model.obj
    cat_dirs = glob.glob(os.path.join(dir_to_check, '*'))
    for cat_dir in cat_dirs:
        model_dirs = glob.glob(os.path.join(dir_to_check, cat_dir, '*[!.txt]'))
        for model_dir in model_dirs:
            model_objs = listFiles(os.path.join(dir_to_check, cat_dir, model_dir), "model.obj")
            if len(model_objs) == 0:
                print(model_dir)
                # shutil.rmtree(model_dir)

def delete_folders_without_model_file():
    # has to be in form of ... - somewhere - .../models/model.obj
    model_dirs = listDirs(dir_to_check, "models")
    for model_dir in model_dirs:
        path = os.path.join(model_dir, "model.obj")
        if not os.path.exists(path):
            print(os.path.split(model_dir)[0])
            # shutil.rmtree(os.path.split(model_dir)[0])

def add_prefix_to_model_folder(prefix):
    # has to be in form of ... - somewhere - .../models/model.obj
    model_dirs = listDirs(dir_to_check, "models")
    for model_dir in model_dirs:
        path = os.path.split(model_dir)[0]

        old = os.path.split(model_dir)[0]
        new = os.path.join(os.path.split(path)[0], prefix + os.path.split(path)[1])
        print("old: {}".format(old))
        print("new: {}".format(new))
        os.rename(old, new)

def move_obj_files_in_subdirectory():
    # has to be in form of category/model_id/* and gets moved to:
    # category/model_id/models/*
    cat_dirs = glob.glob(os.path.join(dir_to_check, '*'))
    for cat_dir in cat_dirs:
        model_dirs = glob.glob(os.path.join(dir_to_check, cat_dir, '*'))
        for model_dir in model_dirs:
            old_path = os.path.join(dir_to_check, cat_dir, model_dir)
            new_path = os.path.join(old_path, "models")

            files = listFiles(old_path, "", "thumbnail.png")
            files_new = [file.replace(old_path,new_path) for file in files]

            for index,_ in enumerate(files):
                os.makedirs(os.path.dirname(files_new[index]), exist_ok=True)
                shutil.move(files[index], files_new[index])

def getAllModelIds():
    all_model_ids = []
    cat_dirs = glob.glob(os.path.join(dir_to_check, '*'))
    for cat_dir in cat_dirs:
        model_dirs = glob.glob(os.path.join(dir_to_check, cat_dir, '*[!.txt]'))
        all_model_ids += [os.path.split(model_dir)[1] for model_dir in model_dirs]
    return all_model_ids

def remove_bad_files():
    good_files_txt = "/Users/markuspaschke/Desktop/Master_backup/ShapeNetToolsPicked/shapenet_good_models.txt"
    folder_to_check = "/Users/markuspaschke/Desktop/Master_backup/TO_RENDER/ShapeNetHandTools/"
    good_models = np.genfromtxt(good_files_txt, delimiter="\n", dtype=str)

    cat_dirs = glob.glob(os.path.join(folder_to_check, '*'))
    for cat_dir in cat_dirs:
        model_dirs = glob.glob(os.path.join(folder_to_check, cat_dir, '*'))
        for model_dir in model_dirs:
            model_id = os.path.split(model_dir)[1]
            if not model_id in good_models:
                print("deleted: {}".format(model_dir))
                shutil.rmtree(model_dir)



## get "good files and save them"
# model_ids = getAllModelIds()
# np.savetxt("./model_ids.txt", model_ids, fmt="%s")


## remove models, which are not in the "good_files" model list
# remove_bad_files()


add_prefix_to_model_folder("sn_")
