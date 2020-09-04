#!/usr/bin/env python

import os, sys, pickle, cv2, trimesh, sklearn.preprocessing, subprocess, math
import numpy as np

from log.logger import setup_formatted_logger, setup_simple_logger
logging = setup_formatted_logger('generation_logger', 'log/generation.log')
xyz_logging = setup_simple_logger('xyz_logger', 'cache/xyz_models.txt')

DATASET_FILE_PATH = "./DataSet"
P2M_BLENDER_PATH = "../p2m_blender/"

# Step 0 generate list of files in all subdirs
# Step 1 generate blender images
# Step 2 generate xyz files with output of #1
# Step 3 generate dat files from output of #2

class P2MDataGenerator:
    def __init__(self):
        self.pg = PointCloudGenerator()

    def start(self):
        #Step 0
        category_models_dict = self.generate_file_list(DATASET_FILE_PATH)
        model_list = [item for sublist in category_models_dict.values() for item in sublist]

        if len(model_list) == 0:
            sys.exit("Please provide at least one model in ./DataSet!")

        #Step 1
        self.generate_blender_images(model_list)

        #Step 2 / 3
        self.pg.generate_xyz_files(model_list)


    def generate_file_list(self, path):

        # load models dirs from cache
        category_models_dict = self.load_obj('model_dict')
        if category_models_dict is not None:
            logging.info("found cached model dir list in cache/model_dict.pkl")
            return category_models_dict

        category_models_dict = {}
        categories = next(os.walk(path))[1]

        for category in categories:
            subdir = os.path.join(path, category)
            model_dirnames = next(os.walk(subdir))[1]
            #FOR SHAPENETv2 Dataset
            model_dirs = [os.path.abspath(os.path.join(path,category,model_dirname,"models","model.obj")) for model_dirname in model_dirnames]
            #FOR P2M Shapenet Dataset
            #model_dirs = [os.path.abspath(os.path.join(path,category,model_dirname, "model.obj")) for model_dirname in model_dirnames]
            category_models_dict[category] = model_dirs

        self.save_obj(category_models_dict, "model_dict")

        return category_models_dict

    def generate_blender_images(self, model_list):

        # check in cache if we already have some images already created -> dont create them again
        # this can happen, if the script was killed or w/e
        rendered_images = self.load_txt_to_list('rendered_models')

        #already cached images ->
        if rendered_images is not None:
            logging.info("skipping {} already rendered_images".format(len(rendered_images)))
            model_list = list(set(model_list) - set(rendered_images))
            logging.info("generating images for {}".format(len(model_list)))

        # we might exceed the MAX_ARG_STRLEN for commands (which is 131072 for a single string argument)
	# so split it up. (subprocess.call will wait to finish before executing the next one)
        array_size = sys.getsizeof(model_list)
        chunkNumbers = math.ceil(array_size / 131000)
        #temporary fix
        chunkNumbers = 20
        chunkArray = self.chunkArray(model_list, chunkNumbers)

        logging.info("generating images now")
        logging.info("images split into {} parts".format(chunkNumbers))
        for index, arrayPart in enumerate(chunkArray):
            logging.info("current {} / {} : images {}".format(index + 1, chunkNumbers, len(arrayPart)))

            '''
            subprocess.call(["p2m_blender/blender_macos/blender.app/Contents/MacOS/blender",
                     "--background", "--python", "p2m_blender/render_blender_batch.py", "--",
                     "--views", "5", "--input_list"] + arrayPart)
            '''
            #'''
            subprocess.call(["p2m_blender/blender_linux/blender",
                     "--background", "--python", "p2m_blender/render_blender_batch.py",  "--",
                     "--views", "5", "--input_list"] + arrayPart)
            #'''

    def chunkArray(self, array, num):
        avg = len(array) / float(num)
        out = []
        last = 0.0

        while last < len(array):
            out.append(array[int(last):int(last + avg)])
            last += avg

        return out

    def save_obj(self, obj, name ):
        if not os.path.exists("cache"):
            os.makedirs("cache")

        with open('cache/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        if not os.path.isfile('cache/' + name + '.pkl'):
            return None

        with open('cache/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def load_txt_to_list(self, name):
        fp = 'cache/' + name + '.txt'
        if not os.path.isfile(fp):
            return None

        return np.genfromtxt(fp,dtype='str').tolist()

    #def clearCache():
    #    for f in glob.glob("cache/*"):
    #        os.remove(f)

class PointCloudGenerator:

    def generate_xyz_files(self, model_list):

        # check in cache if we already have some xyz files already created -> dont create them again
        # this can happen, if the script was killed or w/e
        cached_xyz_models = CacheManager.load_txt_to_list('xyz_models')

        #already cached xyz models ->
        if cached_xyz_models is not None:
            logging.info("skipping {} already generated xyz files".format(len(cached_xyz_models)))
            model_list = list(set(model_list) - set(cached_xyz_models))

        length = len(model_list)
        logging.info("generating {} xyz files".format(length))
        for index, model in enumerate(model_list):
            logging.info("{} / {}".format(index + 1, length))
            self.generate_single_xyz_file(model)
            xyz_logging.info(model)

    def generate_single_xyz_file(self, obj_path):

        #For ShapeNet.v2 Dataset
        view_path = os.path.join(os.sep.join(obj_path.split(os.sep)[:-2]), 'rendering', 'rendering_metadata.txt')

        #FOR P2M DataSet
        #view_path = os.path.join(os.path.split(obj_path)[0], 'rendering', 'rendering_metadata.txt')

        mesh_list = trimesh.load_mesh(obj_path)

        if not isinstance(mesh_list, list):
            mesh_list = [mesh_list]
        area_sum = 0
        for mesh in mesh_list:
            area_sum += np.sum(self.as_mesh(mesh).area_faces)

        sample = np.zeros((0,3), dtype=np.float32)
        normal = np.zeros((0,3), dtype=np.float32)

        for mesh in mesh_list:
            number = int(round(16384*np.sum(self.as_mesh(mesh).area_faces)/area_sum))
            if number < 1:
                continue
            points, index = trimesh.sample.sample_surface_even(self.as_mesh(mesh), number)
            sample = np.append(sample, points, axis=0)

            triangles = mesh.triangles[index]
            pt1 = triangles[:,0,:]
            pt2 = triangles[:,1,:]
            pt3 = triangles[:,2,:]
            norm = np.cross(pt3-pt1, pt2-pt1)
            norm = sklearn.preprocessing.normalize(norm, axis=1)
            normal = np.append(normal, norm, axis=0)

        self.transformToCameraView(sample, normal, view_path)


    def transformToCameraView(self, sample, normal, view_path):
        # 2 tranform to camera view
        position = sample * 0.57

        cam_params = np.loadtxt(view_path)
        for index, param in enumerate(cam_params):
            # camera tranform
            cam_mat, cam_pos = self.camera_info(param)

            pt_trans = np.dot(position-cam_pos, cam_mat.transpose())
            nom_trans = np.dot(normal, cam_mat.transpose())
            train_data = np.hstack((pt_trans, nom_trans))

            img_path = os.path.join(os.path.split(view_path)[0], '%02d.png'%index)
            #not necessary to save xyz file -> save dat file
            #np.savetxt(img_path.replace('png','xyz'), train_data)

            #### project for sure
            #img = cv2.imread(img_path)
            #img = cv2.resize(img, (224,224))

            #X,Y,Z = pt_trans.T
            #F = 250
            #h = (-Y)/(-Z)*F + 224/2.0
            #w = X/(-Z)*F + 224/2.0
            #h = np.minimum(np.maximum(h, 0), 223)
            #w = np.minimum(np.maximum(w, 0), 223)
            #img[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
            #img[np.round(h).astype(int), np.round(w).astype(int), 1] = 255
            #cv2.imwrite(img_path.replace('.png','_prj.png'), img)

            #self.generate_dat_file(img_path)
            self.generate_dat_file(img_path, train_data)

    def generate_dat_file(self, img_path, proj_points):
      xyz_path = img_path.replace('png','xyz')
      dat_path = img_path.replace('png','dat')
      #proj_points = np.loadtxt(xyz_path)
      serialized = pickle.dumps(proj_points, protocol=2)

      with open(dat_path,'wb') as file_object:
          file_object.write(serialized)

    def as_mesh(self, scene_or_mesh):
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in scene_or_mesh.geometry.values()])
        else:
            mesh = scene_or_mesh
        return mesh

    def camera_info(self, param):
        theta = np.deg2rad(param[0])
        phi = np.deg2rad(param[1])

        camY = param[3]*np.sin(phi)
        temp = param[3]*np.cos(phi)
        camX = temp * np.cos(theta)
        camZ = temp * np.sin(theta)
        cam_pos = np.array([camX, camY, camZ])

        axisZ = cam_pos.copy()
        axisY = np.array([0,1,0])
        axisX = np.cross(axisY, axisZ)
        axisY = np.cross(axisZ, axisX)

        cam_mat = np.array([axisX, axisY, axisZ])
        cam_mat = sklearn.preprocessing.normalize(cam_mat, axis=1)
        return cam_mat, cam_pos

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

    #def clearCache():
    #    for f in glob.glob("cache/*"):
    #        os.remove(f)

if __name__ == "__main__":
    P2MDataGenerator().start()
