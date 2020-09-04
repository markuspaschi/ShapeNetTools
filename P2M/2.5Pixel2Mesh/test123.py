import tensorflow as tf

feed_dict = dict()

placeholders = {
    'img_inp': tf.placeholder(tf.float32, shape=(5, 5, 3))
    }


img_inp = [[[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
     [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
     [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
     [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
     [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]]

depth_inp = [[[1],[1],[1],[1],[1]],
     [[1],[1],[1],[1],[1]],
     [[1],[1],[1],[1],[1]],
     [[1],[1],[1],[1],[1]],
     [[1],[1],[1],[1],[1]]]


feed_dict.update({placeholders['img_inp']: img_inp})

x=placeholders['img_inp']
print(x.shape)
x= tf.concat([img_inp, depth_inp], 2)
x=tf.expand_dims(x, 0)

print(x.shape)

print(x)
