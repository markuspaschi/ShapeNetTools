#!/usr/bin/env python3

import sys
import time
import os
import contextlib
from math import radians
from mathutils import Vector
import numpy as np
from tempfile import TemporaryFile
from contextlib import contextmanager
import bpy
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from log.logger import setup_formatted_logger, setup_simple_logger
model_logging = setup_simple_logger('model_logger', 'cache/rendered_models.txt')

MAX_CAMERA_DIST = 1.6

args = None

class BaseRenderer:
	model_idx   = 0

	def __init__(self):
		# bpy.data.scenes['Scene'].render.engine = 'CYCLES'
		# bpy.context.scene.cycles.device = 'GPU'
		# bpy.context.user_preferences.system.compute_device_type = 'CUDA'
		# bpy.context.user_preferences.system.compute_device = 'CUDA_1'

		# changing these values does affect the render.

		# remove the default cube
		bpy.ops.object.select_pattern(pattern="Cube")
		bpy.ops.object.delete()

		render_context = bpy.context.scene.render
		world  = bpy.context.scene.world
		camera = bpy.data.objects['Camera']
		light_1  = bpy.data.objects['Lamp']
		light_1.data.type = 'HEMI'

		# set the camera postion and orientation so that it is in
		# the front of the object
		camera.location       = (1, 0, 0)
		camera.rotation_mode  = 'ZXY'
		camera.rotation_euler = (0, radians(90), radians(90))

		# parent camera with a empty object at origin
		org_obj                = bpy.data.objects.new("RotCenter", None)
		org_obj.location       = (0, 0, 0)
		org_obj.rotation_euler = (0, 0, 0)
		bpy.context.scene.objects.link(org_obj)

		camera.parent = org_obj  # setup parenting

		# render setting
		render_context.resolution_percentage = 100
		world.horizon_color = (1, 1, 1)  # set background color to be white

		bpy.context.scene.render.image_settings.file_format = 'PNG'  # set output format to .png

		self.render_context = render_context
		self.org_obj = org_obj
		self.camera = camera
		self.light = light_1
		self._set_lighting()

	def initialize(self, obj_path, viewport_size_x, viewport_size_y):
		self.obj_path = obj_path
		self.render_context.resolution_x = viewport_size_x
		self.render_context.resolution_y = viewport_size_y

	def _set_lighting(self):
		pass

	def setViewpoint(self, azimuth, altitude, yaw, distance_ratio, fov):
		self.org_obj.rotation_euler = (0, 0, 0)
		self.light.location = (distance_ratio * (MAX_CAMERA_DIST + 2), 0, 0)
		self.camera.location = (distance_ratio * MAX_CAMERA_DIST, 0, 0)
		self.org_obj.rotation_euler = (radians(-yaw),radians(-altitude),radians(-azimuth))

	def setTransparency(self, transparency):
		""" transparency is either 'SKY', 'TRANSPARENT'
		If set 'SKY', render background using sky color."""
		self.render_context.alpha_mode = transparency

	def selectModel(self):
		bpy.ops.object.select_all(action='DESELECT')
		bpy.ops.object.select_pattern(pattern="RotCenter")
		bpy.ops.object.select_pattern(pattern="Lamp*")
		bpy.ops.object.select_pattern(pattern="Camera")
		bpy.ops.object.select_pattern(pattern="New Lamp")
		bpy.ops.object.select_all(action='INVERT')

	def printSelection(self):
		print(bpy.context.selected_objects)

	def clearModel(self):
		self.selectModel()
		bpy.ops.object.delete()

		# The meshes still present after delete
		for item in bpy.data.meshes:
			bpy.data.meshes.remove(item)
		for item in bpy.data.materials:
			bpy.data.materials.remove(item)

	def loadModel(self, file_path=None):
		if file_path is None:
			file_path = self.obj_path

		if file_path.endswith('obj'):
			bpy.ops.import_scene.obj(filepath=file_path)
		else:
			raise Exception("Loading failed: %s Model loading for type %s not Implemented" %
							(file_path, file_path[-4:]))

	def origin_to_center(self):
		global global_bbox_center
		selected_objects = bpy.context.selected_objects

		for o in selected_objects:
			bpy.context.scene.objects.active = o

		bpy.ops.object.join()

		objects = bpy.context.selected_objects

		mesh = objects[0]

		local_bbox_center =  0.125 * sum((Vector(b) for b in mesh.bound_box), Vector())
		global_bbox_center = o.matrix_world * local_bbox_center
		z_displacement = global_bbox_center[2]

		mesh.location = mesh.location - global_bbox_center

	def scale_to_world(self):

		global scaling

		objects = bpy.context.selected_objects
		mesh = objects[0]
		meshBound = mesh.matrix_world.to_quaternion() * mesh.dimensions

		x_ratio =  abs(meshBound.x)
		y_ratio =  abs(meshBound.y)
		z_ratio =  abs(meshBound.z)
		max_ratio = max(x_ratio, y_ratio, z_ratio)

		if max_ratio != 0:
			ratio = 1 / max_ratio
			# dont fill completely (leave little space around)
			scaling = 0.8 * ratio
			mesh.scale *= scaling
		else:
			scaling = 1

	def render(self, image_path, load_model=True, clear_model=True,
		resize_ratio=None, return_image=True):
		""" Render the object """
		if load_model:
			self.loadModel()

		# resize object
		self.selectModel()

		self.scale_to_world()
		self.origin_to_center()

		self.result_fn = image_path
		bpy.context.scene.render.filepath = image_path
		bpy.ops.render.render(write_still=True)  # save straight to file

		if clear_model:
			self.clearModel()


class ShapeNetRenderer(BaseRenderer):

	def __init__(self):
		super().__init__()
		self.setTransparency('TRANSPARENT')

	def _set_lighting(self):
		# Create new lamp datablock
		light_data = bpy.data.lamps.new(name="New Lamp", type='HEMI')

		# Create new object with our lamp datablock
		light_2 = bpy.data.objects.new(name="New Lamp", object_data=light_data)
		bpy.context.scene.objects.link(light_2)

		# put the light behind the camera. Reduce specular lighting
		self.light.location       = (0, -2, 2)
		self.light.rotation_mode  = 'ZXY'
		self.light.rotation_euler = (radians(45), 0, radians(90))
		self.light.data.energy = 0.2

		light_2.location       = (0, 2, 2)
		light_2.rotation_mode  = 'ZXY'
		light_2.rotation_euler = (-radians(45), 0, radians(90))
		light_2.data.energy = 0.2

def ignore_quarter_viewports(degrees):
	rnd = np.random.rand()
	if abs(degrees % 90) < 20:
		return degrees + 20 + (rnd * 15)
	elif abs(degrees % -90) < 20:
		return degrees - 20 + (rnd * -15)
	else:
		return degrees

def main():
	obj_path = args.obj

	renderer = ShapeNetRenderer()
	renderer.initialize(obj_path, 224, 224)

	model_path = os.path.split(os.path.split(os.path.split(obj_path)[0])[0])
	category_identifier = os.path.split(model_path[0])[1]
	model_identifier = model_path[1]

	output_folder = os.path.join(args.output_folder, category_identifier, model_identifier , "rendering/")
	render_list_file_path = os.path.join(output_folder, "rendering_metadata.txt")
	scale_list_file_path = os.path.join(output_folder, "scaling.txt")

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	image_id = 0

	f = open(render_list_file_path, "w")
	for i in range(0, args.views):

		#az, el, yaw, depth_ratio = list(*([360, 360, 0, 0.3] * np.random.rand(1, 4) + [0, 0, 0, 0.65]))

		#az = ignore_quarter_viewports(az)
		#el = ignore_quarter_viewports(el)
		#yaw = ignore_quarter_viewports(yaw)

		az, el, depth_ratio = list(*([360, 5, 0.3] * np.random.rand(1, 3) + [0, 25, 0.65]))

		output_file =  output_folder + '{0:02d}'.format(image_id)

		renderer.setViewpoint(az, el, 0, depth_ratio, 25)
		renderer.render(image_path=output_file, load_model=True, clear_model=True)

		f.write('{} {} 0 {} 25\n'.format(az, el, depth_ratio))
		image_id += 1

	with open(scale_list_file_path, "w") as scale_f:
		scale_f.write('{} {} {} {}\n'.format(global_bbox_center[0], global_bbox_center[1], global_bbox_center[2], scaling))

	scale_f.close()
	model_logging.info(obj_path)

def get_args():
	global args

	parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
	parser.add_argument('--views', type=int, default=8,
						help='number of views to be rendered')
	parser.add_argument('obj', type=str,
						help='Path to the obj file to be rendered.')
	parser.add_argument('--output_folder', type=str, default='./tmp/',
						help='The path the output will be dumped to.')
	parser.add_argument('--scale', type=float, default=1,
						help='Scaling factor applied to model. Depends on size of mesh.')
	parser.add_argument('--remove_doubles', type=bool, default=True,
						help='Remove double vertices to improve mesh quality.')
	parser.add_argument('--edge_split', type=bool, default=True,
						help='Adds edge split filter.')
	parser.add_argument('--depth_scale', type=float, default=1.4,
						help='Scaling that is applied to depth. Depends on size of mesh. '
							 'Try out various values until you get a good result. '
							 'Ignored if format is OPEN_EXR.')
	parser.add_argument('--color_depth', type=str, default='8',
						help='Number of bit per channel used for output. Either 8 or 16.')
	parser.add_argument('--format', type=str, default='PNG',
						help='Format of files generated. Either PNG or OPEN_EXR')

	argv = sys.argv[sys.argv.index("--") + 1:]
	args = parser.parse_args(argv)


if __name__ == "__main__":
	get_args()
	main()
