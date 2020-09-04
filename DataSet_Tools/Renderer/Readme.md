
# Render Script
Generate png's for .obj files with blender from different viewpoints.

We ship a prebuilt blender with **numpy** enabled for MacOS and Linux.  

>If you run a different configuration, you need to built it by source.
To install numpy [check out](https://blender.stackexchange.com/questions/5287/using-3rd-party-python-modules)

The corresponding files for Linux and MacOS can be found in:

> p2m_blender/blender_linux  
> p2m_blender/blender_macos

If blender / blenderplayer only exist in .tar.bz2 please unpack them in their corresponding directory.
GIT has a limitation on a max file size of 100MB.

> linux: ./p2m_blender/blender_linux/blender.tar.bz2  
> macOS: ./p2m_blender/blender_macos/blenderplayer.tar.bz2, ./p2m_blender/blender_macos/blender.app.tar.bz2

The different render scripts are found in *p2m_blender*.  
The currently active version is V13, older versions will be removed in future.
(Bad lighting and other problems)

Other implementations can be found in *p2m_blender/scripts*.

### Requirements

#### 1. Blender
Check if your blender configuration is working. Change directory for Linux.

`./p2m_blender/blender_macos/blender.app/Contents/MacOS/blender --background`


#### 2. DataSet
###### EXAMPLE FILES

Folder structure (ShapeNet and other authors use the same structure):

| Directories         | Path                                                           |
|---------------------|----------------------------------------------------------------|
| DataSet             | */DataSet/*                                                    |
| Category            | */DataSet/cars/*                                               |
| Models per Category | */DataSet/cars/1f93dbc9622d83de7a9f0bb7b1eb35a4/*              |
| Model               | */DataSet/cars/1f93dbc9622d83de7a9f0bb7b1eb35a4/models/*       |


| Files in Model Directory | Description               |
|--------------------------|---------------------------|
| model\.obj               | Wavefront object file     |
| model\.obj\.mtl          | Material file \(texture\) |


### Fill out

> DATASET_FILE_PATH = "MY_DATASET_PATH"  
OUTPUT_PATH = "MY_OUTPUT_PATH_FOR_GENERATED_PNGS"  
BLENDER_PATH = "p2m_blender/blender_linux/"  
RENDER_SCRIPT = "p2m_blender/render_blender_V13.py"  
N_JOBS_PARALLEL = 1


### Run

###### In foreground
`python render_handtools.py`

###### In background
`nohup python render_handtools.py &`

###### Check the current status if run in background with:

    wc -l rendered_models.txt  
    tail log/generation.log  
    tail log/render_blender.log
