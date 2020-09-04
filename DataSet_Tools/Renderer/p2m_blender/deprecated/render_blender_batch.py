# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# blender --background --python mytest.py -- --views 10 /path/to/my.obj
#

import sys, os
import argparse
import numpy as np
from math import radians
import io

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from log.logger import setup_formatted_logger, setup_simple_logger
logging = setup_formatted_logger('render_logger', 'log/render_blender.log')
model_logging = setup_simple_logger('model_logger', 'cache/rendered_models.txt')

#import logging
#logging.basicConfig(level = logging.INFO, filename = "log/render_blender.log")

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=30,
                    help='number of views to be rendered')
#parser.add_argument('obj', type=str,
#                    help='Path to the obj file to be rendered.')
#parser.add_argument('--output_folder', type=str, default='/tmp',
#                    help='The path the output will be dumped to.')
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
parser.add_argument('--input_list', nargs='+', type=str)

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

import bpy

length = len(args.input_list)
logging.info("generate images for " + str(length) + " objects")
for index, input in enumerate(args.input_list):
    logging.info("current image " + str(index + 1) + " / " + str(length))

    bpy.ops.wm.read_factory_settings()

    bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.user_preferences.addons['cycles'].preferences.devices[0].use = True

    # Set up rendering of depth map.
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Add passes for additionally dumping albedo and normals.
    bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
    bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
    bpy.context.scene.render.image_settings.file_format = args.format
    bpy.context.scene.render.image_settings.color_depth = args.color_depth

    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    if args.format == 'OPEN_EXR':
      links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
      # Remap as other types can not represent the full range of depth.
      map = tree.nodes.new(type="CompositorNodeMapValue")
      # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
      map.offset = [-0.7]
      map.size = [args.depth_scale]
      map.use_min = True
      map.min = [0]
      links.new(render_layers.outputs['Depth'], map.inputs[0])

      links.new(map.outputs[0], depth_file_output.inputs[0])

    scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    scale_normal.blend_type = 'MULTIPLY'
    # scale_normal.use_alpha = True
    scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])

    bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    bias_normal.blend_type = 'ADD'
    # bias_normal.use_alpha = True
    bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_normal.outputs[0], bias_normal.inputs[1])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    links.new(bias_normal.outputs[0], normal_file_output.inputs[0])

    albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = 'Albedo Output'
    links.new(render_layers.outputs['Color'], albedo_file_output.inputs[0])

    # Delete default cube
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete()

    target_obj = None
    bpy.ops.import_scene.obj(filepath=input)
    for object in bpy.context.scene.objects:
        if object.name in ['Camera', 'Lamp']:
            continue
        bpy.context.scene.objects.active = object
        target_obj = object
        if args.scale != 1:
            bpy.ops.transform.resize(value=(args.scale,args.scale,args.scale))
            bpy.ops.object.transform_apply(scale=True)
        if args.remove_doubles:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.remove_doubles()
            bpy.ops.object.mode_set(mode='OBJECT')
        if args.edge_split:
            bpy.ops.object.modifier_add(type='EDGE_SPLIT')
            bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
            bpy.ops.object.modifier_apply(apply_as='DATA', modifier="EdgeSplit")

    light_1  = bpy.data.objects['Lamp']
    light_1.data.type = 'HEMI'
    light = light_1

    light_data = bpy.data.lamps.new(name="New Lamp", type='HEMI')

    # Create new object with our lamp datablock
    light_2 = bpy.data.objects.new(name="New Lamp", object_data=light_data)
    bpy.context.scene.objects.link(light_2)

    # put the light behind the camera. Reduce specular lighting
    light.location       = (0, -2, 2)
    light.rotation_mode  = 'ZXY'
    light.rotation_euler = (radians(45), 0, radians(90))
    light.data.energy = 0.7

    light_2.location       = (0, 2, 2)
    light_2.rotation_mode  = 'ZXY'
    light_2.rotation_euler = (-radians(45), 0, radians(90))
    light_2.data.energy = 0.7

    def parent_obj_to_camera(b_camera):
        origin = (0, 0, 0)
        b_empty = bpy.data.objects.new("Empty", None)
        b_empty.location = origin
        b_camera.parent = b_empty  # setup parenting

        scn = bpy.context.scene
        scn.objects.link(b_empty)
        scn.objects.active = b_empty
        return b_empty


    def camera_info(param):
        theta = np.deg2rad(param[0])
        phi = np.deg2rad(param[1])
        # print(param[0],param[1], theta, phi, param[6])

        camY = param[3]*np.sin(phi) * param[6]
        temp = param[3]*np.cos(phi) * param[6]
        camX = temp * np.cos(theta)
        camZ = temp * np.sin(theta)
        cam_pos = np.array([camX, camY, camZ])

        axisZ = cam_pos.copy()
        axisY = np.array([0,1,0])
        axisX = np.cross(axisY, axisZ)
        # axisY = np.cross(axisZ, axisX)

        #print(camX, camY, camZ)
        return camX, -camZ, camY

    scene = bpy.context.scene
    scene.render.resolution_x = 224
    scene.render.resolution_y = 224
    scene.render.resolution_percentage = 100
    scene.render.alpha_mode = 'TRANSPARENT'
    world  = bpy.context.scene.world
    world.horizon_color = (1, 1, 1)  # set background color to be white
    cam = scene.objects['Camera']
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty

    # scene.render.image_settings.file_format = 'PNG'  # set output format to .png

    rotation_mode = 'XYZ'

    stepsize = 360 / args.views

    for output_node in [depth_file_output, normal_file_output, albedo_file_output]:
        output_node.base_path = ''
        output_node.format.file_format="PNG"

    #NOT USED CURRENTLY
    #model_identifier = os.path.split(os.path.split(input)[0])[1]

    # FOR ShapeNetDataset.v2
    fp = os.sep.join(input.split(os.sep)[:-2])
    #FOR P2M Dataset
    #fp = os.path.split(input)[0]

    obj_image_dir = fp + '/rendering/'

    target_obj.location = (0.0, 0.0, 0.0)

    current_rot_value = 0
    metastring = ""

    for i in range(args.views):

        current_rot_value += stepsize

        angle_rand = np.random.rand(3)
        y_rot = current_rot_value + angle_rand[0] * 10 - 5
        x_rot = 20 + angle_rand[1] * 10
        dist = 0.8
        param = [y_rot, x_rot, 0, dist, 35, 32, 3]
        camX, camY, camZ = camera_info(param)
        cam.location = (camX, camY, camZ)
        scene.render.filepath = obj_image_dir + '/{0:02d}'.format(i)

        # redirect output to log file
        '''
        logfile = 'blender_render.log'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)
        '''

        bpy.ops.render.render(write_still=True)  # render still

        # disable output redirection
        '''
        os.close(1)
        os.dup(old)
        os.close(old)
        '''

        metastring = metastring + "{} {} {} {} {} \n" \
                     .format(y_rot, x_rot, 0, dist, 25)

    with open(obj_image_dir+"/rendering_metadata.txt", "w") as f:
        f.write(metastring)

    model_logging.info(input)
