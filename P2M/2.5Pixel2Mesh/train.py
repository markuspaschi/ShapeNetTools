#!/usr/bin/env python
# We want one GPU for this script
#SBATCH --gres=gpu:1

# We want to submit this job to the partition named 'long'.
#___SBATCH -p short
#SBATCH -p long

# We will need two hours of compute time
# IMPORTANT: If you exceed this timelimit your job will NOT be canceled. BUT for
# SLURM to be able to schedule efficiently a reasonable estimate is needed.
#___SBATCH -t 0:40:00
#SBATCH -t 5:00:00


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
import subprocess
from datetime import datetime
import tensorflow as tf
from p2m.utils import *
from p2m.models import GCN
from p2m.fetcher import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Arguments for training the vgg - p2m network')

parser.add_argument('--training_data',
                  help='Training data.',
                  type=str,
                  default='data/training_data/trainer_res.txt')
parser.add_argument('--testing_data',
                  help='Testing data.',
                  type=str,
                  default='data/testing_data/test_list.txt')
parser.add_argument('--learning_rate',
                  help='Learning rate.',
                  type=float,
                  default=1e-5)
parser.add_argument('--show_every',
                  help='Frequency of displaying loss',
                  type=int,
                  default=10)
parser.add_argument('--epochs',
                  help='Number of epochs to train.',
                  type=int,
                  default=10)
parser.add_argument('--checkpoint',
                  help='Checkpoint to use.',
                  type=str,
                  default=None#'data/checkpoints/last_checkpoint_res.pt'
                  )  # rechanged #changed


args = parser.parse_args()


if not os.path.exists(args.training_data):
    sys.exit("INVALID --training_data")

if args.checkpoint is not None and not os.path.exists(args.checkpoint):
    sys.exit("INVALID --checkpoint")

if not os.path.exists(args.testing_data):
    sys.exit("INVALID --testing_data")

# Set random seed
seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_list', args.training_data, 'Data list.') # training data list
flags.DEFINE_float('learning_rate', args.learning_rate, 'Initial learning rate.')
flags.DEFINE_integer('epochs', args.epochs, 'Number of epochs to train.')
flags.DEFINE_integer('hidden', 256, 'Number of units in hidden layer.') # gcn hidden layer channel
flags.DEFINE_integer('feat_dim', 963, 'Number of units in feature layer.') # image feature dim
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')

# Define placeholders(dict) and model
num_blocks = 3
num_supports = 2
placeholders = {
    'features': tf.placeholder(tf.float32, shape=(None, 3)),
    'img_inp': tf.placeholder(tf.float32, shape=(224, 224, 3)),
    'depth_inp': tf.placeholder(tf.float32, shape=(224, 224, 1)),
    'labels': tf.placeholder(tf.float32, shape=(None, 6)),
    'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],  #for face loss, not used.
    'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)],
    'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)], #for laplace term
    'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks-1)] #for unpooling
}

model = GCN(placeholders, logging=True)

# Load data, initialize session
data = DataFetcher(FLAGS.data_list, include_depth=True)
data.setDaemon(True) ####
data.start()
config=tf.ConfigProto()
#config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

if args.checkpoint is not None:
    model.load(args.checkpoint, sess)

pkl = pickle.load(open('Data/ellipsoid/info_ellipsoid.dat', 'rb'))
feed_dict = construct_feed_dict(pkl, placeholders)

train_number = data.number

print 'Total Epochs: %d, Total Iterations per Epoch %d'%(FLAGS.epochs, train_number)

start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
saving_path = os.path.join(os.getcwd(), 'outputs', start_time)
os.makedirs(saving_path)

print 'Saving into %s'%(saving_path)

train_loss = open('%s/record_train_loss.txt'%(saving_path), 'a')
train_loss.write('Start training, lr =  %f\n'%(FLAGS.learning_rate))

for epoch in range(FLAGS.epochs):
    all_loss = np.zeros(train_number,dtype='float32')
    for iters in range(train_number):
        # Fetch training data
        img_inp, depth_img, y_train, data_id, pkl_path = data.fetch()

        feed_dict.update({placeholders['img_inp']: img_inp})
        feed_dict.update({placeholders['depth_inp']: depth_img})
        feed_dict.update({placeholders['labels']: y_train})

        # Training step
        _, dists,out1,out2,out3 = sess.run([model.opt_op,model.loss,model.output1,model.output2,model.output3], feed_dict=feed_dict)

        all_loss[iters] = dists
        mean_loss = np.mean(all_loss[np.where(all_loss)])


        if (iters+1) % args.show_every == 0:
            print 'Epoch %d, Iteration %d'%(epoch + 1,iters + 1)
            print 'Mean loss = %f, iter loss = %f, %d'%(mean_loss,dists,data.queue.qsize())


    epoch_dir = saving_path + '/epoch_{}'.format(epoch + 1)
    os.makedirs(epoch_dir)
    os.makedirs(epoch_dir + '/outputs')
    print('-------- Folder created : {}'.format(epoch_dir))

    # Save model
    model.save(epoch_dir, sess)
    train_loss.write('Epoch %d, loss %f\n'%(epoch+1, mean_loss))
    train_loss.flush()

data.shutdown()
print 'Training Finished!'
print 'Now testing'

# from eval_testset_handset import run
# run(args.testing_data, saving_path)
