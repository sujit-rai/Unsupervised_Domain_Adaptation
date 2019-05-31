from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE

from flip_gradient import flip_gradient
from utils import *

from scipy.misc import imsave

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)




inpt = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name="input")


initializer = tf.contrib.layers.xavier_initializer()

def leaky_relu(x):
	return tf.nn.leaky_relu(x)


def encoder(x,isTrainable=True,reuse=False):
    with tf.variable_scope("encoder1", reuse=reuse):
        #32x32x3
        conv1 = tf.layers.conv2d(x, 64, 5, strides = 2, padding="SAME", 
            kernel_initializer=initializer, activation=leaky_relu,trainable=isTrainable,reuse=reuse,name='conv1_layer');
        conv1 = tf.layers.batch_normalization(conv1,name='conv1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        #16x16x64
        conv2 = tf.layers.conv2d(conv1, 128, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=leaky_relu,trainable=isTrainable,reuse=reuse,name='conv2_layer')
        conv2 = tf.layers.batch_normalization(conv2,name='conv2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        #8x8x128 
        conv3 = tf.layers.conv2d(conv2, 256, 5, strides = 1, padding="SAME",
            kernel_initializer=initializer, activation=leaky_relu,trainable=isTrainable,reuse=reuse,name='conv3_layer')
        conv3 = tf.layers.batch_normalization(conv3,name='conv3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        #4x4x256
        
        fc = tf.contrib.layers.flatten(conv3)
        fc1 = tf.layers.dense(fc, units = 128, 
            kernel_initializer= initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse,name='fc_layer')

        return fc1




def decoder(z,isTrainable=True,reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        fc = tf.layers.dense(z, units = 7*7*512, 
            kernel_initializer = initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse)
        fc = tf.layers.batch_normalization(fc,name='fc_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        fl = tf.reshape(fc, [-1, 7, 7, 512]);

        #4x4x512
        deconv1 = tf.layers.conv2d_transpose(fl, 256, 5, strides=2, padding="SAME", 
            kernel_initializer=initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse)
        deconv1 = tf.layers.batch_normalization(deconv1,name='deconv1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        #8x8x256
        deconv2 = tf.layers.conv2d_transpose(deconv1, 128, 5, strides=1, padding="SAME", 
            kernel_initializer=initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse)
        deconv2 = tf.layers.batch_normalization(deconv2,name='deconv2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);


        #16x16x128
        deconv3 = tf.layers.conv2d_transpose(deconv2, 3, 5, strides=2, padding="SAME", 
            kernel_initializer=initializer, activation = tf.nn.sigmoid ,trainable=isTrainable,reuse=reuse)
        

        # deconv3 = tf.layers.batch_normalization(deconv3,name='deconv3_layer_batchnorm',
        #     trainable=isTrainable,reuse=reuse);

        print(deconv3)
        # #32x32x64
        # deconv4 = tf.layers.conv2d_transpose(deconv3, 3, 5, strides=1, padding="SAME", 
        #     kernel_initializer=initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse)
        ##Now the o/p is 32x32x3

        #deconv4 = tf.layers.batch_normalization(deconv4,name='deconv4_layer_batchnorm',
            #trainable=isTrainable,reuse=reuse);

        #32x32x3
        #deconv5 = tf.layers.conv2d_transpose(deconv4, 5, 1, strides=1, padding="SAME", 
         #   kernel_initializer=initializer, activation = tf.nn.tanh,trainable=isTrainable,reuse=reuse,name='')

        return deconv3



ls = encoder(inpt)

recon = decoder(ls)

recon_loss = tf.reduce_mean(tf.square(inpt - recon))

recon_op = tf.train.AdamOptimizer(0.0001).minimize(recon_loss)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

batch_size = 65

with tf.Session() as sess:
	sess.run(init)


	# Process MNIST
	mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8)
	mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
	mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8)
	mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

	mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
	mnistm_train = mnistm['train'] / 255.0
	mnistm_test = mnistm['test'] / 255.0
	mnistm_valid = mnistm['valid'] / 255.0

	print(np.amin(mnist_train))
	print(np.amax(mnist_train))

	ri = []
	for i in range(100):
		rls = []
		sin = []
		for j in range(int(55000/65)):
			i_choice = np.random.choice(len(mnistm_train), batch_size)
			sin = mnistm_train[i_choice]
			_, rl, ri = sess.run([recon_op, recon_loss, recon], feed_dict={inpt: sin})
			rls.append(rl)
		print(np.mean(rls))
		imsave("recons.jpg", ri[0])
		plt.imshow(sin[0,:,:,:]*0.99, origin="upper",interpolation='nearest');
		plt.savefig('og.png');