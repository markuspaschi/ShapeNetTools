#!/usr/bin/env python
# We want one GPU for this script
#SBATCH --gres=gpu:1

# We want to submit this job to the partition named 'long'.
#SBATCH -p long

# We will need two hours of compute time
# IMPORTANT: If you exceed this timelimit your job will NOT be canceled. BUT for
# SLURM to be able to schedule efficiently a reasonable estimate is needed.
#___SBATCH -t 1:00:00
#SBATCH -t 60:00:00

# We want to be notified by email when the jobs starts / ends
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=markus.paschke@tu-ilmenau.de

#  Copyright (C) 2019 Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, Yu-Gang Jiang, Fudan University
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import sys, os
sys.path.append(os.getcwd())

import argparse
import tensorflow as tf
from p2m.models import GCN
from p2m.fetcher import *
from p2m.chamfer import nn_distance
sys.path.append('external')
from tf_approxmatch import approx_match, match_cost

def get_args():
	parser = argparse.ArgumentParser(description='Enter additional args')
	parser.add_argument('--testing_data', type=str, required=True, help='Testing list (corresponds to folder in Data/train_test_lists/).')
	parser.add_argument('--checkpoints', type=str, required=True, help='Checkpoints folder (epoch_1 / epoch_2...) (corresponds to folder in outputs).')
	parser.add_argument('--eval_name', type=str, required=True, help='Name for evaluation')
	parser.add_argument('--show_every',help='Frequency of displaying loss', type=int, default=100)
	parser.add_argument('--restrict_epochs', '--list', nargs='*', type=int, help='Epochs to evaluate, keep empty for all epochs')
	args = parser.parse_args()

	if not os.path.exists(args.testing_data):
		sys.exit("INVALID --testing_data")

	if not os.path.exists(args.checkpoints):
		sys.exit("INVALID --checkpoints")

	if os.path.exists(os.path.join(args.checkpoints, args.eval_name)):
		sys.exit("Eval name does already exist!")

	return args


def run(testing_data, checkpoints, eval_name, show_every=10):
	# Settings
	flags = tf.app.flags
	FLAGS = flags.FLAGS
	flags.DEFINE_string('data_list', testing_data, 'Data list path.')
	flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rate.')
	flags.DEFINE_integer('hidden', 256, 'Number of units in  hidden layer.')
	flags.DEFINE_integer('feat_dim', 963, 'Number of units in perceptual feature layer.')
	flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.')
	flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')

	# Define placeholders(dict) and model
	num_blocks = 3
	num_supports = 2
	placeholders = {
		'features': tf.placeholder(tf.float32, shape=(None, 3)), # initial 3D coordinates
		'img_inp': tf.placeholder(tf.float32, shape=(224, 224, 3)), # input image to network
        'depth_inp': tf.placeholder(tf.float32, shape=(224, 224, 1)),
		'labels': tf.placeholder(tf.float32, shape=(None, 6)), # ground truth (point cloud with vertex normal)
		'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)], # graph structure in the first block
		'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)], # graph structure in the second block
		'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)], # graph structure in the third block
		'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)], # helper for face loss (not used)
		'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)], # helper for normal loss
		'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)], # helper for laplacian regularization
		'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks-1)] # helper for graph unpooling
	}
	model = GCN(placeholders, logging=True)

	# Construct feed dictionary
	def construct_feed_dict(pkl, placeholders):
		coord = pkl[0]
		pool_idx = pkl[4]
		faces = pkl[5]
		lape_idx = pkl[7]
		edges = []
		for i in range(1,4):
			adj = pkl[i][1]
			edges.append(adj[0])

		feed_dict = dict()
		feed_dict.update({placeholders['features']: coord})
		feed_dict.update({placeholders['edges'][i]: edges[i] for i in range(len(edges))})
		feed_dict.update({placeholders['faces'][i]: faces[i] for i in range(len(faces))})
		feed_dict.update({placeholders['pool_idx'][i]: pool_idx[i] for i in range(len(pool_idx))})
		feed_dict.update({placeholders['lape_idx'][i]: lape_idx[i] for i in range(len(lape_idx))})
		feed_dict.update({placeholders['support1'][i]: pkl[1][i] for i in range(len(pkl[1]))})
		feed_dict.update({placeholders['support2'][i]: pkl[2][i] for i in range(len(pkl[2]))})
		feed_dict.update({placeholders['support3'][i]: pkl[3][i] for i in range(len(pkl[3]))})
		return feed_dict

	def f_score(label, predict, dist_label, dist_pred, threshold):
		num_label = label.shape[0]
		num_predict = predict.shape[0]

		f_scores = []
		for i in range(len(threshold)):
			num = len(np.where(dist_label <= threshold[i])[0])
			recall = 100.0 * num / num_label
			num = len(np.where(dist_pred <= threshold[i])[0])
			precision = 100.0 * num / num_predict

			f_scores.append((2*precision*recall)/(precision+recall+1e-8))
		return np.array(f_scores)

	checkpoints_folders = [os.path.join(checkpoints, f) for f in os.listdir(checkpoints) if f.startswith("epoch_")]

	# log to file in training output folder
	eval_dir = os.path.join(checkpoints, eval_name)
	if not os.path.exists(eval_dir):
		os.makedirs(eval_dir)

	log = open('%s/record_evaluation.txt'%(eval_dir), 'a')


	for index, checkpoint_folder in enumerate(checkpoints_folders):

		# Check if some epochs should be skipped in evaluation
		if args.restrict_epochs != None and (index + 1) not in args.restrict_epochs:
			continue

		print("folder {}".format(checkpoint_folder))

		# Load data
		data = DataFetcher(FLAGS.data_list, include_depth=True)
		data.setDaemon(True) ####
		data.start()
		train_number = data.number

		# Initialize session
		# xyz1:dataset_points * 3, xyz2:query_points * 3
		xyz1=tf.placeholder(tf.float32,shape=(None, 3))
		xyz2=tf.placeholder(tf.float32,shape=(None, 3))
		# chamfer distance
		dist1,idx1,dist2,idx2 = nn_distance(xyz1, xyz2)
		# earth mover distance, notice that emd_dist return the sum of all distance
		match = approx_match(xyz1, xyz2)
		emd_dist = match_cost(xyz1, xyz2, match)

		config=tf.ConfigProto()
		config.gpu_options.allow_growth=True
		config.allow_soft_placement=True
		sess = tf.Session(config=config)
		sess.run(tf.global_variables_initializer())
		model.load(checkpoint_folder, sess)

		# Construct feed dictionary
		pkl = pickle.load(open('Data/ellipsoid/info_ellipsoid.dat', 'rb'))
		feed_dict = construct_feed_dict(pkl, placeholders)

		###
		model_number = {}
		sum_f = {}
		sum_cd = {}
		sum_emd = {}

		eval = {}

		for iters in range(train_number):
			# Fetch training data
			img_inp, depth_inp, label, class_name, object_dat_path = data.fetch()

			feed_dict.update({placeholders['img_inp']: img_inp})
			feed_dict.update({placeholders['labels']: label})
			feed_dict.update({placeholders['depth_inp']: depth_inp})

			# Training step
			# predict = sess.run(model.output3, feed_dict=feed_dict)

			predict, meshloss1, meshloss2, meshloss3, laplaceloss1, laplaceloss2,laplaceloss3, l2losses, regularizationloss, full_loss = sess.run(
				[model.output3, model.meshloss1, model.meshloss2, model.meshloss3, model.laplaceloss1, model.laplaceloss2, model.laplaceloss3, model.l2losses, model.regularizationloss, model.full_loss ],
				feed_dict=feed_dict
			)


			label = label[:, :3]
			d1,i1,d2,i2,emd = sess.run([dist1,idx1,dist2,idx2, emd_dist], feed_dict={xyz1:label,xyz2:predict})
			cd = np.mean(d1) + np.mean(d2)
			f_sc = f_score(label,predict,d1,d2,[0.0001, 0.0002])

			if label.shape[0] == 0:
				continue

			# full evaluation (per object)
			object_name = object_dat_path.split('/')[-3]
			viewport = object_dat_path.split('/')[-1].replace(".dat","")
			identifier = class_name + "/" + object_name
			eval.setdefault(identifier, {})
			eval[identifier][viewport] = {
					"cd": cd * 1000,
					"emd": emd[0] * 0.01,
					"f_score": f_sc.tolist(),
					"meshloss1": meshloss1,
					"meshloss2": meshloss2,
					"meshloss3": meshloss3,
					"laplaceloss1": laplaceloss1,
					"laplaceloss2": laplaceloss2,
					"laplaceloss3": laplaceloss3,
					"l2losses": l2losses,
					"regularizationloss": regularizationloss,
					"full_loss": full_loss
			}



			class_id = class_name#.split('_')[0]

			model_number.setdefault(class_id,0)
			sum_f.setdefault(class_id,0)
			sum_cd.setdefault(class_id,0)
			sum_emd.setdefault(class_id,0)

			model_number[class_id] += 1.0

			sum_f[class_id] += f_sc
			sum_cd[class_id] += cd # cd is the mean of all distance
			sum_emd[class_id] += emd[0] # emd is the sum of all distance
			if (iters+1) % show_every == 0:
				   print 'processed number', iters + 1


		print("Saving eval per object to pickle")
		with open(os.path.join(eval_dir, "epoch_{}_full_eval.dat".format(index + 1)), 'wb') as f:
			pickle.dump(eval, f, protocol=pickle.HIGHEST_PROTOCOL)

		print >> log, "Epoch: ", index + 1
		for item in model_number:
			number = model_number[item] + 1e-8
			f = sum_f[item] / number
			cd = (sum_cd[item] / number) * 1000 #cd is the mean of all distance, cd is L2
			emd = (sum_emd[item] / number) * 0.01 #emd is the sum of all distance, emd is L1
			print item, int(number), f, cd, emd
			print >> log, item, int(number), f, cd, emd
		sess.close()
		data.shutdown()
	log.close()
	print 'Testing Finished!'

if __name__ == "__main__":
	args = get_args()
	run(args.testing_data, args.checkpoints, args.eval_name, args.show_every)
