from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE

from flip_gradient import flip_gradient
from utils import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Process MNIST
mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

# Load MNIST-M
mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
mnistm_train = mnistm['train']
mnistm_test = mnistm['test']
mnistm_valid = mnistm['valid']

# Compute pixel mean for normalizing data
pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))

# Create a mixed dataset for TSNE visualization
num_test = 500
combined_test_imgs = np.vstack([mnist_test[:num_test], mnistm_test[:num_test]])
combined_test_labels = np.vstack([mnist.test.labels[:num_test], mnist.test.labels[:num_test]])
combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]), np.tile([0., 1.], [num_test, 1])])

batch_size = 64

initializer = tf.truncated_normal_initializer(mean=0, stddev=0.02);#tf.contrib.layers.xavier_initializer()


def leaky_relu(x):
	return tf.nn.leaky_relu(x)

X = tf.placeholder(tf.uint8, [None, 28, 28, 3])
y = tf.placeholder(tf.float32, [None, 10])
domain = tf.placeholder(tf.float32, [None, 2])
l = tf.placeholder(tf.float32, [])
train = tf.placeholder(tf.bool, [])
source_labels = tf.placeholder(tf.float32, [None, 10])

X_input = (tf.cast(self.X, tf.float32) - pixel_mean) / 255.


def inv_fe(x):
	with tf.variable_scope("inv_fe"):
		h_conv0 = tf.layers.conv2d(x, 32, 5, strides = 1, padding="SAME", activation=leaky_relu);
		h_pool0 = max_pool_2x2(h_conv0)

		h_conv1 = tf.layers.conv2d(h_pool0, 48, 5, strides = 1, padding="SAME", activation=leaky_relu);
		h_pool1 = max_pool_2x2(h_conv1)

		return tf.reshape(h_pool1, [-1, 7*7*48])

def clf(x):
	with tf.variable_scope("classifier"):
		h_fc0 = tf.layers.dense(x, units=100, activation = leaky_relu)
		h_fc1 = tf.layers.dense(h_fc0, units=100, activation = leaky_relu)
		logits = tf.layers.dense(h_fc1, units=10, activation = None)
		return logits


def disc(feat, l):
	with tf.variable_scope("discriminator"):
		feat = flip_gradient(feat, l)
		d_h_fc0 = tf.layers.dense(feat, units=100, activation = leaky_relu)
		d_logits = tf.layers.dense(d_h_fc0, units=2, activation = None)
		return d_logits

learning_rate = tf.placeholder(tf.float32, [])
classify_labels = tf.cond(train, source_labels, all_labels)
inv_feats = inv_fe(X_input)
cls_logits = clf(inv_feats)
disc_logits = disc(inv_feats, l)
