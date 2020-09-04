#!/usr/bin/env python3

import argparse, sys, os, time
import numpy as np
from math import radians
from tempfile import TemporaryFile
from contextlib import contextmanager
sys.path.append('/usr/local/lib/python3.7/site-packages')
from PIL import Image
#from PIL import Image

import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=30,
                    help='number of views to be rendered')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=1.4,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

@contextmanager
def stdout_redirected(new_stdout):
    save_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout

import bpy

model_identifier = os.path.split(os.path.split(args.obj)[0])[1]
fp = os.path.join(args.output_folder, model_identifier)

IMAGE_DIR = fp + '/rendering/'
BLENDER_TMP_DIR = fp + '/tmp/'
MAX_CAMERA_DIST = 0.8

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

        # set file name for storing rendering result
        self.result_fn = '%s/render_result_%d.png' % (IMAGE_DIR, os.getpid())
        bpy.context.scene.render.filepath = self.result_fn

        self.render_context = render_context
        self.org_obj = org_obj
        self.camera = camera
        self.light = light_1
        self._set_lighting()

    def initialize(self, viewport_size_x, viewport_size_y):
        self.render_context.resolution_x = viewport_size_x
        self.render_context.resolution_y = viewport_size_y

    def _set_lighting(self):
        pass

    def setViewpoint(self, azimuth, altitude, yaw, distance_ratio, fov):
        self.org_obj.rotation_euler = (0, 0, 0)
        self.light.location = (distance_ratio *
                               (MAX_CAMERA_DIST + 2), 0, 0)
        self.camera.location = (distance_ratio *
                                MAX_CAMERA_DIST, 0, 0)
        self.org_obj.rotation_euler = (radians(-yaw),
                                       radians(-altitude),
                                       radians(-azimuth))

    def setTransparency(self, transparency):
        """ transparency is either 'SKY', 'TRANSPARENT'
        If set 'SKY', render background using sky color."""
        self.render_context.alpha_mode = transparency

    def selectModel(self):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_pattern(pattern="RotCenter")
        bpy.ops.object.select_pattern(pattern="Lamp*")
        bpy.ops.object.select_pattern(pattern="Camera")
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

    def setModelIndex(self, model_idx):
        self.model_idx = model_idx

    def loadModel(self, file_path=None):
        bpy.ops.import_scene.obj(filepath=args.obj)



    def render(self, load_model=True, clear_model=True, resize_ratio=None,
               return_image=True, image_path=os.path.join(BLENDER_TMP_DIR, 'tmp.png')):
        """ Render the object """
        if load_model:
            self.loadModel()

        # resize object
        self.selectModel()
        if resize_ratio:
            bpy.ops.transform.resize(value=resize_ratio)

        self.result_fn = image_path
        bpy.context.scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)  # save straight to file

        if resize_ratio:
            bpy.ops.transform.resize(value=(1/resize_ratio[0],
                1/resize_ratio[1], 1/resize_ratio[2]))

        if clear_model:
            self.clearModel()

        if return_image:
            im = np.array(Image.open(self.result_fn))  # read the image

            # Last channel is the alpha channel (transparency)
            return im[:, :, :3], im[:, :, 3]


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
        self.light.data.energy = 0.7

        light_2.location       = (0, 2, 2)
        light_2.rotation_mode  = 'ZXY'
        light_2.rotation_euler = (-radians(45), 0, radians(90))
        light_2.data.energy = 0.7


def writeMetadata(y_rot, x_rot, dist):
    metastring = metastring + "{} {} {} {} {} \n" \
            .format(y_rot, x_rot, 0, dist, 25)


def main():
    """Test function"""
    # Modify the following file to visualize the model

    model_identifier = os.path.split(os.path.split(args.obj)[0])[1]
    fp = os.path.join(args.output_folder, model_identifier)
    obj_image_dir = fp + '/rendering/'

    model_id = [model_identifier]

    #model_id = [line.strip('\n') for line in open(dn + 'models.txt')]
    #file_paths = [os.path.join(dn, line, 'model.obj') for line in model_id]
    sum_time = 0
    renderer = ShapeNetRenderer()
    renderer.initialize(224, 224)
    for i, curr_model_id in enumerate(model_id):
        start = time.time()
        image_path = '%s/%s.png' % ('/tmp', curr_model_id[:-4])

        az, el, depth_ratio = list(
            *([360, 5, 0.3] * np.random.rand(1, 3) + [0, 25, 0.65]))

        renderer.setModelIndex(i)
        renderer.setViewpoint(30, 30, 0, 0.7, 25)

        #writeMetadata()

        with TemporaryFile('w') as f, stdout_redirected(f):
            rendering, alpha = renderer.render(load_model=True,
                clear_model=True, image_path=image_path)

        print('Saved at %s' % image_path)

        end = time.time()
        sum_time += end - start
        if i % 10 == 0:
            print(sum_time/(10))
            sum_time = 0

    #with open(obj_image_dir+"/rendering_metadata.txt", "w") as f:
    #    f.write(metastring)


main()
