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

current_file_path = "./current_file.txt"
deleted_files_path = "./should_delete_files.txt"
time = 0
current_file_last = 0
deleted_files_last = 0

current_file = ""
deleted_files = np.array([])

dir = "/Users/markuspaschke/Desktop/Master_backup/ShapeNetHandTools"

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

current_index = 1;
model_size = len(models) + 1;

window = pyglet.window.Window(width = 300, height = 700)

def updateUI():
    global shouldDelete
    global counter
    global delete_counter

    try:
        current_file = np.genfromtxt(current_file_path,dtype='str')
        current_index = models.index(current_file)
    except:
        current_index = 0
        pass


    counter = pyglet.text.Label('Model # {}/{}'.format(current_index, model_size),
                              color=(0,100,0,255),
                              font_size=20,
                              x=10, y=window.height - 30)

    if(current_file in deleted_files):
        shouldDelete = pyglet.text.Label('Deleted',
            color=(255,0,0,255),font_size=36,
            x=window.width//2, y=window.height - 150, anchor_x='center')
    else:
        shouldDelete = pyglet.text.Label('OK',
            color=(0,100,0,255),font_size=36,
            x=window.width//2, y=window.height - 150, anchor_x='center')


    delete_counter = pyglet.text.Label('Deleted: {}'.format(deleted_files.size),
                              color=(0,100,0,255),
                              font_size=20,
                              x=10, y=window.height - 70)

def updateDeletedFiles():
    global deleted_files
    deleted_files = np.genfromtxt(deleted_files_path,dtype='str')

updateUI()
updateDeletedFiles()


def check_file():
    global current_file_last
    global deleted_files_last

    #update ui when current file changes
    current_file_current = os.path.getmtime(current_file_path)
    if current_file_current != current_file_last:
        updateUI()
    current_file_last = current_file_current

    # update array, if deleted items change
    deleted_files_current = os.path.getmtime(deleted_files_path)
    if deleted_files_current != deleted_files_last:
        updateDeletedFiles()
        updateUI()
    deleted_files_last = deleted_files_current

def update(dt):
    global time
    time += dt

    if(int(round(time)) % 3 == 0):
        check_file()


@window.event
def on_draw():
    window.clear()

    shouldDelete.draw()
    counter.draw()
    delete_counter.draw()

pyglet.clock.schedule(update)
pyglet.app.run()
