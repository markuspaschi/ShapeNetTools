import pywavefront
import os.path
import ctypes
import pyglet
from pyglet.gl import *
from pyglet.window import key
from pywavefront import visualization
import pywavefront
import sys
import numpy as np

dir = "/Users/markuspaschke/Desktop/Master_backup/ShapeNetHandTools"
current_file = "./current_file.txt"
deleted_files = "./should_delete_files.txt"


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

models = listFiles(dir, "model.obj")


current_model_index = 1
current_scale = 1
rotation = 0
scene = pywavefront.Wavefront(models[current_model_index], create_materials=True)
window = pyglet.window.Window(width = 1024, height = 700)
lightfv = ctypes.c_float * 4



def scale():
    global current_scale
    # Iterate vertex data collected in each material
    for name, material in scene.materials.items():

        vertices = np.array([material.vertices])
        ordererd_vertices = np.reshape(vertices, (-1,3))

        x_mean = np.mean(ordererd_vertices[:,0])
        y_mean = np.mean(ordererd_vertices[:,1])
        z_mean = np.mean(ordererd_vertices[:,2])

        current_scale = 0.1 / y_mean
        ordererd_vertices = ordererd_vertices * current_scale
        scene.materials[name].vertices = np.reshape(ordererd_vertices, (-1,1))

def scale_by(scale):
    global current_scale
    current_scale = current_scale * scale
    # Iterate vertex data collected in each material
    for name, material in scene.materials.items():

        vertices = np.array([material.vertices])
        ordererd_vertices = np.reshape(vertices, (-1,3))

        ordererd_vertices = ordererd_vertices * current_scale
        scene.materials[name].vertices = np.reshape(ordererd_vertices, (-1,1))

@window.event
def on_resize(width, height):
    viewport_width, viewport_height = window.get_framebuffer_size()
    glViewport(0, 0, viewport_width, viewport_height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., float(width)/height, 1., 100.)
    glMatrixMode(GL_MODELVIEW)
    return True


@window.event
def on_draw():
    window.clear()
    glLoadIdentity()

    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-1.0, 1.0, 1.0, 0.0))
    glEnable(GL_LIGHT0)

    glTranslated(0.0, 0.0, -3.0)
    glRotatef(rotation, 0.0, 1.0, 0.0)
    glRotatef(-25.0, 1.0, 0.0, 0.0)
    glRotatef(45.0, 0.0, 0.0, 1.0)

    glEnable(GL_LIGHTING)

    visualization.draw(scene)


def update(dt):
    global rotation
    rotation += 90.0 * dt

    if rotation > 720.0:
        rotation = 0.0


@window.event
def on_key_press(symbol, modifiers):
    global current_model_index
    global scene

    if symbol == key.RIGHT:
        current_model_index = (current_model_index + 1 ) % len(models)
        scene = pywavefront.Wavefront(models[current_model_index])
        scale()
        np.savetxt(current_file, [models[current_model_index]], fmt='%s')

    elif symbol == key.LEFT:
        current_model_index = (current_model_index -1 ) % len(models)
        scene = pywavefront.Wavefront(models[current_model_index])
        scale()
        np.savetxt(current_file, [models[current_model_index]], fmt='%s')

    if symbol == key.H:
        current_model_index = (current_model_index + 100 ) % len(models)
        scene = pywavefront.Wavefront(models[current_model_index])
        scale()
        np.savetxt(current_file, [models[current_model_index]], fmt='%s')

    if symbol == key.F:
        current_model_index = (current_model_index + 50 ) % len(models)
        scene = pywavefront.Wavefront(models[current_model_index])
        scale()
        np.savetxt(current_file, [models[current_model_index]], fmt='%s')

    if symbol == key.T:
        current_model_index = (current_model_index + 20 ) % len(models)
        scene = pywavefront.Wavefront(models[current_model_index])
        scale()
        np.savetxt(current_file, [models[current_model_index]], fmt='%s')

    elif symbol == key.MINUS:
        scene = pywavefront.Wavefront(models[current_model_index])
        scale_by(0.5)

    elif symbol == key.PLUS:
        scene = pywavefront.Wavefront(models[current_model_index])
        scale_by(2)

    elif symbol == key.D:
        with open(deleted_files,'ab') as f:
            np.savetxt(f, [models[current_model_index]], fmt='%s')

    elif symbol == key.O:
        with open(deleted_files, "r") as f:
            lines = f.readlines()
        with open(deleted_files, "w") as f:
            for line in lines:
                if (models[current_model_index]) not in line:
                    f.write(line)


pyglet.clock.schedule(update)
pyglet.app.run()
