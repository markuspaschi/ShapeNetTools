
import os, sys, pickle, cv2, trimesh, sklearn.preprocessing, subprocess, math, glob
import numpy as np

def generate_single_xyz_file(obj_path):

    mesh_list = trimesh.load_mesh(obj_path)

    if not isinstance(mesh_list, list):
        mesh_list = [mesh_list]
    area_sum = 0

    print(obj_path)
    for mesh in mesh_list:
        try:
            area_sum += np.sum(mesh.area_faces)
        except Exception:
            try:
                area_sum += np.sum(self.as_mesh(mesh).area_faces)
            except Exception:
                pass


    sample = np.zeros((0,3), dtype=np.float32)
    normal = np.zeros((0,3), dtype=np.float32)

    for mesh in mesh_list:
        try:
            number = int(round(16384*np.nan_to_num(np.sum(mesh.area_faces)/area_sum)))
        except Exception:
            try:
                number = int(round(16384*np.nan_to_num(np.sum(self.as_mesh(mesh).area_faces)/area_sum)))
            except Exception:
                number = 0
        if number < 1:
            continue

        try:
            points, index = trimesh.sample.sample_surface_even(mesh, number)
        except Exception:
            points, index = trimesh.sample.sample_surface_even(self.as_mesh(mesh), number)
        sample = np.append(sample, points, axis=0)

        triangles = mesh.triangles[index]
        pt1 = triangles[:,0,:]
        pt2 = triangles[:,1,:]
        pt3 = triangles[:,2,:]
        norm = np.cross(pt3-pt1, pt2-pt1)
        norm = sklearn.preprocessing.normalize(norm, axis=1)
        normal = np.append(normal, norm, axis=0)

    # bad meshes can happen in our dataset
    if len(sample) == 0 or len(normal) == 0:
        print("Bad MESH")
        return

    position = sample

    proj_points = np.hstack((position, normal))

    dat_path = obj_path.replace('obj','dat')
    #proj_points = np.loadtxt(xyz_path)
    serialized = pickle.dumps(proj_points, protocol=2)

    with open(dat_path,'wb') as f:
        f.write(serialized)

    np.savetxt(dat_path.replace(".dat", ".xyz"), proj_points, delimiter =' ')

    print("Saved to : ", dat_path)


obj_path = "/Users/markuspaschke/Documents/Master-Workbench/Jupiter/pixel2mesh_test/Data/examples/hammer/hammer_cuts/hammer_cut_right.obj"
generate_single_xyz_file(obj_path)
