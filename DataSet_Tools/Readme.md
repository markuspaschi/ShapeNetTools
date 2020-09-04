# Tools for the DataSet

Tools for the aquired DataSet with *.obj* files.

1. [Renderer](Renderer)
  * Used to generate png's for your dataset.
  * Utilizes Blender. Adjust and change the render script for your needs.
2. [MeshlabTools](MeshlabTools)
  * Re-Compute Vertex Normals, should be handled by blender already,
  but sometimes objects have weird transparency.
  * *use only if you have transparency problems*
3. [TrainingTestingSplit](TrainingTestingSplit)
  * Generate training and test list, with path to every single object.
4. [AddOcclusion](AddOcclusion)
  * Mimic a hidden part of an object. Crops out parts of an image.
5. [Color Mesh](ColorMesh)
  * Color a mesh according to a ground_truth colored object
5. [VisualizeModels](VisualizeModels)
  * Visualize object files and with the option to delete the corresponding model
6. [ContourCheck](ContourCheck)
  * Experimental - Check the amount of contours in an image. Images containing a single object should only contain a single contour. *use with caution!*
7. [DatToXyz](DatToXyz)
  * Convert .dat file to .xyz (simple pickle load and convert)
  * works only for point cloud .dat file
