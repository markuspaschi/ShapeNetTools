# ShapeNetTools & Pixel2Mesh implementation

This repository contains some DataSet Generation and Evaluation Tools and an adapted Pixel2Mesh implementation for the following paper

[Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images (ECCV2018)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Nanyang_Wang_Pixel2Mesh_Generating_3D_ECCV_2018_paper.pdf)

Check [Pixel2Mesh Repository](https://github.com/nywang16/Pixel2Mesh) for more information on how to set up Pixel2Mesh.


## Overview

1. [Datset Downloader](DataSet_Downloader)
  - Initial Step: Download your desired .obj files from ShapeNet or Google 3D Warehouse
2. [Dataset Tools](DataSet_Tools)
  - Second Step: Prepare your DataSet for Pixel2Mesh or other Neural Networks.
    - Includes the **Renderer** to generate png's from different viewpoints.
    - Includes Occlusion (cropping holes in png's)
    - Generating Training and Testing Split
3. [Pixel2Mesh](P2M)
  - Run your desired Neural Network (in our case Pixel2Mesh) with different variants:
    - Pixel2Mesh with 2D (standard implementation)
    - Pixel2Mesh with 0.5D (only depth images)
    - Pixel2Mesh with 2.5D (rgbd images)
4. [Evalution Tools](Evaluation_Tools)
  - Some Tools for plotting losses
  - Losses per viewpoint analysis


## Dependencies

##### 1. Requirements for Pixel2Mesh

  * Python2.7+ with Numpy and scikit-image
  * Tensorflow (version 1.0+)
  * TFLearn


  * *Code has been tested with Python 2.7, TensorFlow 1.3.0, TFLearn 0.3.2, CUDA 8.0 on Ubuntu 14.04.*


##### 2. Requirements for Downloader (subject to change)

  * Python3
  * BeautifulSoup, joblib, pandas, requests, numpy

##### 3. Requirements for Renderer

  * Python3
  * Working blender (check Renderer Readme)
  * (Meshlab)
