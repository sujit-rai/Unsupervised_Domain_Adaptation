# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE
#from flip_gradient import flip_gradient
#from utils import *
from scipy.misc import imsave
import sys
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

tf.set_random_seed(0);
random.seed(0);

inpt = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name="input")
domain = tf.placeholder(tf.float32, shape=[None, 1], name="domain")
src_labels = tf.placeholder(tf.float32,shape=[None,10]);
target_labels = tf.placeholder(tf.float32,shape=[None,10]);

#DCGAN 
initializer = tf.truncated_normal_initializer(mean=0, stddev=0.02);#tf.contrib.layers.xavier_initializer()

#tuning knobs
z_dim = 128;
enc1_z = enc2_z = int(z_dim/2.0);


def load_MNIST_data():
    # Process MNIST
    mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)/255.0
    mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)/255.0
    # print("MNIST meta data");
    # print("mnist_train : ",mnist_train.shape);
    # print("mnist_test : ",mnist_test.shape);
    return mnist_train, mnist_test;

def load_m_mnist_data():
    # Load MNIST-M
    mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
    mnistm_train = mnistm['train']/255.0
    mnistm_test = mnistm['test']/255.0
    mnistm_valid = mnistm['valid']/255.0

    # print("MNIST-M meta data");
    # print("mnist_m_train : ",mnistm_train.shape);
    # print("mnist_m_test : ",mnistm_test.shape);
    # print("mnist_m_val : ",mnistm_valid.shape);
    return mnistm_train, mnistm_test, mnistm_valid;

def leaky_relu(x):
    return tf.nn.tf.nn.leaky_relu(x)

def invar_encoder(x,isTrainable=True,reuse=False):
    with tf.variable_scope("invariant_encoder") as scope:
        if reuse:
            scope.reuse_variables();

        #28x28x3
        conv1 = tf.layers.conv2d(x, 64, 5, strides = 2, padding="SAME", 
            kernel_initializer=initializer, activation=tf.nn.relu,trainable=isTrainable,reuse=reuse,name='conv1_layer');
        conv1 = tf.layers.batch_normalization(conv1,name='conv1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        #14x14x64
        conv2 = tf.layers.conv2d(conv1, 128, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=tf.nn.relu,trainable=isTrainable,reuse=reuse,name='conv2_layer')
        conv2 = tf.layers.batch_normalization(conv2,name='conv2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        #7x7xx128 
        conv3 = tf.layers.conv2d(conv2, 256, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=tf.nn.relu,trainable=isTrainable,reuse=reuse,name='conv3_layer')
        conv3 = tf.layers.batch_normalization(conv3,name='conv3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        #4x4x256
        fc = tf.contrib.layers.flatten(conv3)
        fc1 = tf.layers.dense(fc, units = enc1_z, 
            kernel_initializer= initializer, activation = tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc_layer')

        return fc1

def var_encoder(x,isTrainable=True,reuse=False):
    with tf.variable_scope("variant_encoder") as scope:
        if reuse:
            scope.reuse_variables();

        #28x28x3
        conv1 = tf.layers.conv2d(x, 64, 5, strides = 2, padding="SAME", 
            kernel_initializer=initializer, activation=tf.nn.relu,trainable=isTrainable,reuse=reuse,name='conv1_layer');
        conv1 = tf.layers.batch_normalization(conv1,name='conv1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        #14x14x64
        conv2 = tf.layers.conv2d(conv1, 128, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=tf.nn.relu,trainable=isTrainable,reuse=reuse,name='conv2_layer')
        conv2 = tf.layers.batch_normalization(conv2,name='conv2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        #7x7x128
        conv3 = tf.layers.conv2d(conv2, 256, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=tf.nn.relu,trainable=isTrainable,reuse=reuse,name='conv3_layer')
        conv3 = tf.layers.batch_normalization(conv3,name='conv3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        #4x4x256

        print('conv3');
        print(conv3);
        
        fc = tf.contrib.layers.flatten(conv3)
        fc1 = tf.layers.dense(fc, units = enc2_z, 
            kernel_initializer= initializer, activation = tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc_layer')

        return fc1

def decoder(z,isTrainable=True,reuse=False):
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables();

        fc = tf.layers.dense(z, units = 4*4*256, 
            kernel_initializer = initializer, activation = tf.nn.relu,trainable=isTrainable,reuse=reuse)
        fc = tf.layers.batch_normalization(fc,name='fc_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        fl = tf.reshape(fc, [-1, 4, 4, 256]);

        #4x4x256
        deconv1 = tf.layers.conv2d_transpose(fl, 128, 5, strides=2, padding="SAME", 
            kernel_initializer=initializer, activation = tf.nn.relu,trainable=isTrainable,reuse=reuse,name='deconv1_layer')
        deconv1 = tf.layers.batch_normalization(deconv1,name='deconv1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        #print("dec_deconv1.shape : ",deconv1.shape);

        #8x8x128
        deconv2 = tf.layers.conv2d_transpose(deconv1, 64, 5, strides=2, padding="SAME", 
            kernel_initializer=initializer, activation = tf.nn.relu,trainable=isTrainable,reuse=reuse,name='deconv2_layer')
        deconv2 = tf.layers.batch_normalization(deconv2,name='deconv2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);


        #16x16x64
        deconv3 = tf.layers.conv2d_transpose(deconv2, 64, 5, strides=2, padding="SAME", 
            kernel_initializer=initializer, activation = tf.nn.relu,trainable=isTrainable,reuse=reuse,name='deconv3_layer')
        deconv3 = tf.layers.batch_normalization(deconv3,name='deconv3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);


        #32x32x3
        deconv4 = tf.layers.conv2d_transpose(deconv3, 3, 5, strides=1, padding="SAME", 
            kernel_initializer=initializer, activation = tf.nn.relu,trainable=isTrainable,reuse=reuse,name='deconv4_layer')

        deconv4 = tf.layers.batch_normalization(deconv4,name='deconv4_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        deconv4 = tf.layers.flatten(deconv4);
        deconv_fc = tf.layers.dense(deconv4,28*28*3,activation=tf.nn.sigmoid,trainable=isTrainable,reuse=reuse,name='dec_dense_fc_last_layer');

        deconv_fc_reshaped = tf.reshape(deconv_fc,[-1,28,28,3]); 

        return deconv_fc_reshaped;

def classifier(enc_ft,isTrainable=True,reuse=False):
    with tf.variable_scope("classifier") as scope:
        if reuse:
            scope.reuse_variables();

        fc1 = tf.layers.dense(enc_ft, units = 64,
            kernel_initializer = initializer, activation = tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc1_layer')
        fc1 = tf.layers.batch_normalization(fc1,name='classifier_fc1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        fc2 = tf.layers.dense(fc1, units = 32,
            kernel_initializer = initializer, activation = tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc2_layer')
        fc2 = tf.layers.batch_normalization(fc2,name='classifier_fc2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        fc3 = tf.layers.dense(fc2, units = 16,
            kernel_initializer = initializer, activation = tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc3_layer')
        fc3 = tf.layers.batch_normalization(fc3,name='classifier_fc3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
            
        fc4 = tf.layers.dense(fc3, units = 10,
            kernel_initializer = initializer, activation = None,trainable=isTrainable,reuse=reuse,name='classifier_fc3_layer') 

        return fc4;

def discriminator(x,isTrainable=True,reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables();

        # disc1 = tf.layers.conv2d(x, 256, 4, strides=2, padding="SAME", 
        #   kernel_initializer=initializer, activation = tf.nn.relu)
        # disc2 = tf.layers.conv2d(disc1, 256, 4, strides=2, padding="SAME", 
        #   kernel_initializer=initializer, activation = tf.nn.relu)
        # disc3 = tf.layers.conv2d(disc2, 256, 4, strides=2, padding="SAME", 
        #   kernel_initializer=initializer, activation = tf.nn.relu)
        # fc_disc = tf.contrib.layers.flatten(disc3)
        fc1 = tf.layers.dense(x, units = 64,
            kernel_initializer = initializer, activation = tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc1_layer')
        fc1 = tf.layers.batch_normalization(fc1,name='fc1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        fc2 = tf.layers.dense(fc1, units = 32,
            kernel_initializer = initializer, activation = tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc2_layer')
        fc2 = tf.layers.batch_normalization(fc2,name='fc2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        fc3 = tf.layers.dense(fc2, units = 16,
            kernel_initializer = initializer, activation = tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc3_layer')
        fc3 = tf.layers.batch_normalization(fc3,name='fc3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        fc4 = tf.layers.dense(fc3, units = 4,
            kernel_initializer = initializer, activation = tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc4_layer')
        fc4 = tf.layers.batch_normalization(fc4,name='fc4_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        logits = tf.layers.dense(fc4, units = 1,
            kernel_initializer = initializer, activation = None,trainable=isTrainable,reuse=reuse,name='fc_logits_layer')

        return logits



invar_features = invar_encoder(inpt);
var_features = var_encoder(inpt);

invar_disc = discriminator(invar_features)
var_disc = discriminator(var_features,reuse=True)



z_concat = tf.concat([var_features, invar_features],axis=1);


recons_dec = decoder(z_concat);

logits_clf = classifier(invar_features);

# #test_logits_clf = classifier(enc1_test,isTrainable=False,reuse=True);
test_logits_clf = classifier(z_concat_test,isTrainable=False,reuse=True);

clf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_clf,labels=src_labels));

# #test_clf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=test_logits_clf,labels=target_labels));

target_class = tf.argmax(target_labels,1);
predicted_class = tf.argmax(tf.nn.sigmoid(logits_clf),1);
train_class = tf.argmax(src_labels,1);


train_accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_class,train_class),tf.float32)) 
test_accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_class,target_class),tf.float32))


disc_invariant_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=invar_disc, labels=domain))
disc_variant_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=var_disc, labels=domain))


enc1_invariant_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=invar_disc, labels=0.5*tf.ones_like(domain)))

recon_loss = tf.reduce_mean(tf.square(inpt - recons_dec))
#mnist_recon_loss = tf.reduce_mean(tf.square(inpt - recons_dec))
#invar_recon_loss = tf.reduce_mean(tf.square(inpt - dec_invariant))
#var_recon_loss = tf.reduce_mean(tf.square(inpt - dec_variant))

#combining the disc for both var and invae enc via beta
beta = 0.1;
var_encoder_loss = beta*disc_variant_loss + (1-beta)*recon_loss;



discriminator_loss = disc_invariant_loss + disc_variant_loss;

#combined fwdkl nd revkl via alpha
alpha = 0.1;
inv_encoder_loss = alpha*enc1_invariant_loss + (1-alpha)*recon_loss;

variables = tf.trainable_variables()

decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='decoder');
invariant_enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='invariant_encoder');
variant_enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='variant_encoder');
discriminator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator');
classifier_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='classifier');

# #recon_vars = [var for var in variables if ("discriminator" not in var.name)]

# #disc_vars = [var for var in variables if ("discriminator" in var.name)]

# #invariant_enc_vars = [var for var in variables if ("encoder1" in var.name)]

# #variant_vars = [var for var in variables if ("encoder2" in var.name or "discriminator" in var.name)]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS);