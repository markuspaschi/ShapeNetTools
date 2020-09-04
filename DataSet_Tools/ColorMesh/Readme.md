# Color a mesh according to a groundtruth object

Color your mesh according to groundtruth object. The vertices have to be the same dimension.  
In this example, both objects have 2466 vertices.

![Colored Mesh](meshlab_info.png =250)
![Colored Hammer](colored_hammer.png =250)


### Run
```
python color_obj.py \
--gt_obj "ellipsoid_four_colors.obj" \
--objs_folder "input"
```
