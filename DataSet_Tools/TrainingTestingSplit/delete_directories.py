import shutil
import os.path

dir_to_delete = "/datasets_nas/mapa3789/ShapeNetHandToolsAO/"

bad_files_path = "/datasets_nas/mapa3789/ShapeNetHandToolsAll/00_bad_files.txt"


with open(bad_files_path, "r") as f:
    lines = f.readlines()

for line in lines:
    bad_folder = os.path.join(dir_to_delete, line.strip())

    if os.path.exists(bad_folder):
        print("deleting: {}".format(bad_folder))
        shutil.rmtree(bad_folder)
