"""
Collada to Wavefront (.dae to .obj)
usage: blender -b -P dae_obj.py -- in.dae out.obj
"""
import sys
import bpy # Blender Python API
import math

def clear():
    """ Setup the scene """
    # Delete the default objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def toRadians(degree):
    return degree * math.pi / 180.

def usage():
    sys.stderr.write(__doc__)

def loadDAE(fdae):
    bpy.ops.wm.collada_import(filepath=fdae)

def loadOBJ(fobj):
    bpy.ops.import_scene.obj(filepath=fobj)


def save(fobj='out.obj'):
    bpy.ops.export_scene.obj(filepath=fobj, path_mode='RELATIVE')

#### Modifying
def join():
    for obj in bpy.data.objects:
        if 'mesh' in obj.name or 'Mesh' in obj.name or 'SketchUp' in obj.name:
            obj.select = True
            bpy.context.scene.objects.active = obj
            obj.name = 'mesh'
        else:
            obj.select = False
    try:

        bpy.ops.object.join()
        bpy.data.objects['mesh'].name = 'shape'

        return True
    except RuntimeError:
        return False

def scale():
    scale = 1
    coords = [0,0,0]
    rotation = [0,0,0]

    for obj in bpy.data.objects:
        for dim in range(3):
            if obj.dimensions[dim] > 1:
                orient(scale / obj.dimensions[dim], coords, rotation)

def orient(size, coords, rotation):
    obj = bpy.data.objects['shape']
    largest_dim = max(obj.dimensions)
    scale = size / largest_dim
    for dim in range(3):
        obj.scale[dim] = scale
        obj.location[dim] = coords[dim]
        obj.rotation_euler[dim] = toRadians(rotation[dim])

def main(argv=[]):
    args = []
    if '--' in argv:
        args = argv[argv.index('--')+1:]

    if len(args) < 2:
        usage()
        return 1

    fdae = args[0]
    fobj = args[1]

    # convert to obj
    clear()
    loadDAE(fdae)
    save(fobj)

    # scale to 1
    clear()
    loadOBJ(fobj)
    join()
    scale()
    save(fobj)

if __name__ == '__main__':
    main(sys.argv)
