# Download from ShapeNet by a specified Category

Blender is necessary for this downloader to convert `.dae` objects to `.obj` files.  
If you only want `.dae` files, you can uncomment the corresponding code part.

#### Requirements
* A working blender
  - use the prebuilt blender for MacOS or Linux or build it from source
* Python3


Change **downloader.py** for your needs!

* `PARALLEL_JOBS = 15` for GPU heavy workplaces
* `PARALLEL_JOBS = 1` for a normal Mac

* `BLENDER_DIR = ".."` adjust this to your blender path


#### Run the download

`python downloader.py`
