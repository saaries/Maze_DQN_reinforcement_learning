"""
DQN Class
reference by https://github.com/hunkim/ReinforcementZeroToAll/ 
It is also reference by here. 
DQN(NIPS-2013)
"Playing Atari with Deep Reinforcement Learning"
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
DQN(Nature-2015)
"Human-level control through deep reinforcement learning"
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
"""
# # 17x17x1（该结构 maze_size = 8/9都行）
# conv1 = tf.layers.conv2d(
#     inputs = observation, filters = 32, kernel_size = [2,2],
#     padding = "valid", activation = tf.nn.relu)
# # 16x16x64
#
# # 16x16x64
# pool1 = tf.layers.max_pooling2d(
#     inputs = conv1, pool_size = [2,2], strides = 2)
# # 8x8x64
#
# # 8x8x64
# conv2 = tf.layers.conv2d(
#     inputs = pool1, filters = 64, kernel_size = [2,2],
#     padding = "same", activation = tf.nn.relu)
# # 8x8x64
#
# # 8x8x64
# pool2 = tf.layers.max_pooling2d(
#     inputs = conv2, pool_size = [2,2], strides = 2)
# # 4x4x64
#
# # 4x4x64
# conv3 = tf.layers.conv2d(
#     inputs = pool2, filters = 128, kernel_size = [2,2],
#     padding = "same", activation = tf.nn.relu)
# # 4x4x128
#
# # 4x4x128
# pool3 = tf.layers.max_pooling2d(
#     inputs = conv3, pool_size = [2,2], strides = 2)
# # 2x2x128
#
# # 2x2x128
# pool3_flat3 = tf.reshape(pool3, [-1, 2*2*128])
# # reshape to 1x512
#
# # 1x512
# fc5 = tf.layers.dense(
#     inputs = pool3_flat3, units = self.output_size)
# # 1x4

# 17x17x1（该结构 maze_size = 8/9都行）

# 17x17x1

# 17x17x1（该结构 maze_size = 8/9都行）

import numpy as np
import tensorflow as tf

class DQN:
    def __init__(self, session:tf.compat.v1.Session, input_array:np.ndarray, output_size:int,
                 name:str = 'main'):
        self.session = session
        self.output_size = output_size
        self.net_name = name
        self.data_x = input_array[0]
        self.data_y = input_array[1]
        self.network()

        tf.compat.v1.summary.FileWriter("logs1/", self.session.graph)

    def network(self, learning_rate = 1e-3):
        print('learning_rate:',learning_rate)
        with tf.compat.v1.variable_scope(self.net_name):
            self.X = tf.compat.v1.placeholder(tf.float32, [None, self.data_x, self.data_y, 1])
            observation = self.X
            self.Y = tf.compat.v1.placeholder(tf.float32, [None, self.output_size])

            # 17x17x1
            conv1 = tf.compat.v1.layers.conv2d(
                inputs = observation, filters = 32, kernel_size = [2,2],
                padding = "valid", activation = tf.nn.relu)
            # 16x16x32

            pool1 = tf.compat.v1.layers.max_pooling2d(
                inputs = conv1, pool_size = [2,2], strides = 2)
            # 8x8x32

            conv2 = tf.compat.v1.layers.conv2d(
                inputs = pool1, filters = 64, kernel_size = [2,2],
                padding = "same", activation = tf.nn.relu)
            # 8x8x32

            pool2 = tf.compat.v1.layers.max_pooling2d(
                inputs = conv2, pool_size = [2,2], strides = 2)
            # 4x4x64

            conv3 = tf.compat.v1.layers.conv2d(
                inputs = pool2, filters = 128, kernel_size = [2,2],
                padding = "same", activation = tf.nn.relu)
            # 4x4x128

            pool3 = tf.compat.v1.layers.max_pooling2d(
                inputs = conv3, pool_size = [2,2], strides = 2)

            # 2x2x128
            flat = tf.reshape(pool3, [-1, 2*2*128])
            # reshape to 1x512

            fc = tf.compat.v1.layers.dense(
                inputs = flat, units = self.output_size)
            # 1x4

            self.Qpred = fc
            self.loss = tf.losses.mean_squared_error(self.Y, self.Qpred)

            train = tf.compat.v1.train.AdamOptimizer(learning_rate)
            self.train_op = train.minimize(self.loss)

    def predict(self, state):
        state = np.reshape(state, [-1, self.data_x, self.data_y, 1])
        rs = self.session.run(self.Qpred, feed_dict={self.X: state})

        return rs

    def update(self, x_stack, y_stack):
        x_stack = np.reshape(x_stack, [-1, self.data_x, self.data_y, 1])
        return self.session.run([self.loss, self.train_op],
                                feed_dict = {self.X:x_stack, self.Y:y_stack})





