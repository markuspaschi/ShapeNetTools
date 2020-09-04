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
import numpy as np
import cPickle as pickle
import threading
import Queue
import sys
from skimage import io,transform

class DataFetcher(threading.Thread):
    def __init__(self, file_list, include_depth=False):
        super(DataFetcher, self).__init__()
        self.stopped = False
        self.queue = Queue.Queue(64)

        self.pkl_list = []
        self.include_depth = include_depth
        with open(file_list, 'r') as f:
            while(True):
                line = f.readline().strip()
                if not line:
                    break
                self.pkl_list.append(line)
        self.index = 0
        self.number = len(self.pkl_list)
        np.random.shuffle(self.pkl_list)

    def work(self, idx):
        pkl_path = self.pkl_list[idx]
        label = pickle.load(open(pkl_path, 'rb'))

        depth_img_path = pkl_path.replace('.dat', '.png')
        depth_img_path = depth_img_path[:-4] + '_depth' + depth_img_path[-4:]
        depth_img = io.imread(depth_img_path)
        depth_img = transform.resize(depth_img, (224,224))
        depth_img = depth_img[:,:,:1].astype('float32')
        return depth_img, label, pkl_path.split('/')[-4], pkl_path


    def run(self):
        while self.index < 90000000 and not self.stopped:
            self.queue.put(self.work(self.index % self.number))
            self.index += 1
            if self.index % self.number == 0:
                np.random.shuffle(self.pkl_list)

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()

if __name__ == '__main__':
    file_list = sys.argv[1]
    data = DataFetcher(file_list)
    data.start()

    image,point,cat = data.fetch()
    print image.shape
    print point.shape
    print cat
    data.stopped = True
