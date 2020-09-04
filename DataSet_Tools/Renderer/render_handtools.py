#!/usr/bin/env python

import os, sys, pickle, cv2, trimesh, sklearn.preprocessing, subprocess, math, glob
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from log.logger import setup_formatted_logger, setup_simple_logger

generation_logging = setup_formatted_logger('generation_logger', 'log/generation.log')
xyz_logging = setup_simple_logger('xyz_logger', 'cache/xyz_models.txt')
render_logging = setup_formatted_logger('render_logger', 'log/render_blender.log')

DATASET_FILE_PATH = "/datasets_nas/mapa3789/HandToolsCombined"
OUTPUT_PATH = "/datasets_nas/mapa3789/Pixel2Mesh/HandToolsRendered/ShapeNetHandTools_2.5"

#DATASET_FILE_PATH = "/datasets_nas/mapa3789/ShapeNetCoreV1/"
#OUTPUT_PATH = "/datasets_nas/mapa3789/Pixel2Mesh/Data/ShapeNetP2M_OWN_V25"

BLENDER_PATH = "p2m_blender/blender_linux/"
RENDER_SCRIPT = "p2m_blender/render_blender_handtools_2.5.py"
N_JOBS_PARALLEL = 10

bad_meshes = []

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

		# Step 0.5 -> for subset
		# model_list = self.keep_dataset(model_list, "04401088")

		# Step 0 Clear cache -> rendered & xyz will be created (model list will be kept)
		#CacheManager.clearCacheExceptModelList()
		CacheManager.clearCache()

		#Step 1
		self.generate_blender_images(model_list)

		#Step 2 / 3
		self.pg.generate_xyz_files(model_list)

	def keep_dataset(self, model_list, keep):
		return [ x for x in model_list if keep in x ]

	def generate_file_list(self, path):

		# load models dirs from cache
		category_models_dict = self.load_obj('model_dict')
		if category_models_dict is not None:
			generation_logging.info("found cached model dir list in cache/model_dict.pkl")
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
			generation_logging.info("skipping {} already rendered_images".format(len(rendered_images)))
			model_list = list(set(model_list) - set(rendered_images))
			generation_logging.info("generating images for {}".format(len(model_list)))

		generation_logging.info("generating {} images now".format(len(model_list)))
		#generation_logging.info("images split into {} parts".format(chunkNumbers))

		with Parallel(n_jobs=N_JOBS_PARALLEL) as parallel:
			parallel(delayed(self.gen_obj)(model, index) for index,model in enumerate(model_list))


	def gen_obj(self, obj_path, index):
		render_logging.info("{}".format(str(index)))
		current_dir = os.getcwd() # > /dev/null 2>&1
#MACOS
		#os.system(current_dir + '/p2m_blender/blender_macos/blender.app/Contents/MacOS/blender --background --python p2m_blender/render_blender_v3.py -- %s' % (obj_path))
#LINUX
		os.system('%s/blender --background --python %s -- --output_folder %s %s' % (BLENDER_PATH, RENDER_SCRIPT, OUTPUT_PATH, obj_path))

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


class PointCloudGenerator:

	def generate_xyz_files(self, model_list):

		# check in cache if we already have some xyz files already created -> dont create them again
		# this can happen, if the script was killed or w/e
		cached_xyz_models = CacheManager.load_txt_to_list('xyz_models')

		#already cached xyz models ->
		if cached_xyz_models is not None:
			generation_logging.info("skipping {} already generated xyz files".format(len(cached_xyz_models)))
			model_list = list(set(model_list) - set(cached_xyz_models))

		length = len(model_list)
		generation_logging.info("generating {} xyz files".format(length))

		#with Parallel(n_jobs=6) as parallel:
		#    parallel(delayed(self.generate_single_xyz_file)(model, index, length) for index,model in enumerate(model_list))

		for index, model in enumerate(model_list):
			generation_logging.info("{} / {}".format(index + 1, length))
			mesh_list = trimesh.load_mesh(model)
			self.generate_single_xyz_file(model, mesh_list)
			xyz_logging.info(model)

		print("bad meshes! delete them in dataset")
		print(bad_meshes)

	def generate_single_xyz_file(self, obj_path, mesh_list):
		if not isinstance(mesh_list, list):
			mesh_list = [mesh_list]
		area_sum = 0

		print(obj_path)

		for mesh in mesh_list:
			try:
				area_sum += np.sum(self.as_mesh(mesh).area_faces)
			except Exception:
				pass

		sample = np.zeros((0,3), dtype=np.float32)
		normal = np.zeros((0,3), dtype=np.float32)

		for mesh in mesh_list:
			try:
				number = int(round(16384*np.nan_to_num(np.sum(self.as_mesh(mesh).area_faces)/area_sum)))
			except Exception:
				number = 0
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

		# bad meshes can happen in our dataset
		if len(sample) == 0 or len(normal) == 0:
			bad_meshes.append(obj_path)
			return

		self.transformToCameraView(sample, normal, obj_path)

	def transformToCameraView(self, sample, normal, obj_path):
		# 2 tranform to camera view
		rendering_path = self.get_rendering_path(obj_path)
		rendering_metadata_path = self.get_rendering_metadata_path(obj_path)
		scaling_path = self.get_scaling_path(obj_path)

		cam_params = np.loadtxt(rendering_metadata_path)
		scaling_params = np.loadtxt(scaling_path)

		# scale according to render script!
		render_scaling = scaling_params[3:4]
		position = sample * 0.5 * render_scaling

		# center around (0,0,0)
		mean = np.array([np.mean(position[:,0]), np.mean(position[:,1]), np.mean(position[:,2])])
		position = position - mean

		cam_params = np.loadtxt(rendering_metadata_path)
		for index, param in enumerate(cam_params):
			# camera tranform
			cam_mat, cam_pos = self.camera_info(param)

			pt_trans = np.dot(position-cam_pos, cam_mat.transpose())
			nom_trans = np.dot(normal, cam_mat.transpose())
			train_data = np.hstack((pt_trans, nom_trans))

			img_path = os.path.join(rendering_path, '%02d.png'%index)
			self.generate_dat_file(img_path, train_data)

	def generate_dat_file(self, img_path, proj_points):
	  #xyz_path = img_path.replace('png','xyz')
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

		#changed this to deg2rad!!!
		#camY = np.deg2rad(param[3])*np.sin(phi)
		#temp = np.deg2rad(param[3])*np.cos(phi)
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

	def get_rendering_path(self, obj_path):
		model_path = os.path.split(os.path.split(os.path.split(obj_path)[0])[0])
		category_identifier = os.path.split(model_path[0])[1]
		model_identifier = model_path[1]

		return os.path.join(OUTPUT_PATH, category_identifier, model_identifier , "rendering/")

	def get_rendering_metadata_path(self, obj_path):
		rendering_path = self.get_rendering_path(obj_path)
		return os.path.join(rendering_path, "rendering_metadata.txt")

	def get_scaling_path(self, obj_path):
		rendering_path = self.get_rendering_path(obj_path)
		return os.path.join(rendering_path, "scaling.txt")

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

	def clearCache():
		for f in glob.glob("cache/*"):
			os.remove(f)

	def clearCacheExceptModelList():
		for f in  glob.glob("cache/*.txt"):
			os.remove(f)

if __name__ == "__main__":
	P2MDataGenerator().start()
