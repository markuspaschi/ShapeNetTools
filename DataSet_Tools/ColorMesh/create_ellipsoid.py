import tensorflow as tf
import pickle
import numpy as np
import sys

# load faces (generated via https://github.com/nywang16/Pixel2Mesh/blob/master/GenerateData/4_make_auxiliary_dat_file.ipynb)

ellipsoid_path = '/Users/markuspaschke/Documents/Master-Workbench/Jupiter/pixel2mesh_test/Data/ellipsoid/info_ellipsoid.dat'
face_2_path = '/Users/markuspaschke/Documents/Master-Workbench/Jupiter/pixel2mesh_test/Data/ellipsoid/face3.obj'

pkl = pickle.load(open(ellipsoid_path, 'rb'))

param_x = tf.placeholder(tf.float32, shape=(None, 3))
param_y = tf.placeholder(tf.int32, shape=(None, 2))

#op_x_plus_y = tf.add(param_x, param_y)
new_coords = tf.concat([param_x, (1/2.0) * tf.reduce_sum(tf.gather(param_x, param_y), 1)], 0)


sess = tf.Session()

input = pkl[0].tolist()
pool1 = pkl[4][0].tolist()
pool2 = pkl[4][1].tolist()

coords_1 = sess.run(new_coords, feed_dict={param_x: input, param_y: pool1})
coords_2 = sess.run(new_coords, feed_dict={param_x: coords_1, param_y: pool2})

# print(len(coords_1))
# np.savetxt("./coords_1.txt", coords_1, fmt="%s", delimiter=" ")

# print(len(coords_2))
# np.savetxt("./coords_2.txt", coords_2, fmt="%s", delimiter=" ")

faces_2 = np.genfromtxt(face_2_path, delimiter=" ", dtype="|S6")
coords_2 = np.asarray(coords_2, dtype='|S6')
coords_2 = np.insert(coords_2, 0, "v ", axis=1)

obj = np.vstack([coords_2, faces_2])

np.savetxt("./ellipsoid_new.obj", obj, fmt="%s", delimiter=" ")
