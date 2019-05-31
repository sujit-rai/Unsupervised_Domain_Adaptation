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
# enc2 = encoder2(inpt)

invar_disc = discriminator(invar_features)
var_disc = discriminator(var_features,reuse=True)

# fol = tf.nn.sigmoid(disc1)
# n_fol = tf.nn.sigmoid(disc2)



# #z_concat = tf.concat([enc1,enc2],axis=1)

# dec_invariant = decoder(invar_features);
# dec_variant = decoder(var_features,reuse=True);


# #print('z_concat.shape : ',z_concat.shape);

# ######## OPS FOR INFERENCING
var_enc_test = var_encoder(inpt,isTrainable=False,reuse=True);
invar_enc_test = invar_encoder(inpt,isTrainable=False,reuse=True);

z_concat_test = tf.concat([var_enc_test,invar_enc_test],axis=1);
z_concat = tf.concat([var_enc_test,invar_enc_test],axis=1);

# # print('z_concat_test.shape : ',z_concat_test.shape);

# # dec_test = decoder(z_concat_test,isTrainable=False,reuse=True);

#var_dec_test = decoder(var_enc_test,isTrainable=False,reuse=True);
#invar_dec_test = decoder(invar_enc_test,isTrainable=False,reuse=True);
recons_dec = decoder(z_concat);
recons_dec_test = decoder(z_concat_test,isTrainable=False,reuse=True);

# #logits_clf = classifier(enc1_test);
logits_clf = classifier(z_concat_test);

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
with tf.control_dependencies(update_ops):

    #mnist_recon_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(mnist_recon_loss, var_list = decoder_params+invariant_enc_params+variant_enc_params);
    recon_op = tf.train.AdamOptimizer(learning_rate = 0.0005).minimize(recon_loss, var_list = decoder_params+invariant_enc_params+variant_enc_params);
    #invar_recon_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(invar_recon_loss, var_list = decoder_params+invariant_enc_params);
    #var_recon_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(var_recon_loss, var_list = decoder_params+variant_enc_params);

    #inv_disc : for inv encoder the discriminator
    disc_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(discriminator_loss, var_list = discriminator_params)
    enc_inv_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(inv_encoder_loss, var_list = invariant_enc_params)
    enc_var_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(var_encoder_loss, var_list = variant_enc_params)
    clf_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(clf_loss, var_list = classifier_params);




#tf.summary.scalar("invar_recon_loss", invar_recon_loss)
#tf.summary.scalar("var_recon_loss", var_recon_loss)
tf.summary.scalar("reconstruction_loss", recon_loss)
tf.summary.scalar("discriminator_loss_for_variant_enc", disc_variant_loss)
tf.summary.scalar("discriminator_loss_for_invariant_enc", disc_invariant_loss)
tf.summary.scalar("discriminator loss",discriminator_loss);
#tf.summary.scalar("variant_loss", disc_variant_loss)
# tf.summary.scalar("enc_inv_loss -- wants to confuse discriminator", inv_encoder_loss)
# tf.summary.scalar("enc_var_loss -- wants to help discriminator", var_encoder_loss)


merged_all = tf.summary.merge_all()

log_dir = "feature_extractr_3"

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

clf_train_epochs =100;
epochs = 100;
batch_size = 128

checkpoint_path = "./feature_extractr_3"

mode = 'test'

if mode == 'train':

    with tf.Session() as sess:
        sess.run(init)
        
        # source = np.load("dataset/mnist.npy")
        # target = np.load("dataset/svhn.npy");

        source_train,source_test = load_MNIST_data();
        target_train,target_test,target_val = load_m_mnist_data();

        # print(np.amin(target_train),' , ',np.amax(target_train));
        # print(np.amin(target_test),' , ',np.amax(target_test));
        # print(np.amin(target_val),' , ',np.amax(target_val));

        # sys.exit(0);

        print('source_train.shape : ',source_train.shape);
        print('target_train.shape : ',target_train.shape);

        recon = []

        target_len = len(target_train);
        source_len = len(source_train);

        source_len_test = len(source_test);
        target_len_val = len(target_val);

        print('-'*80);
        print('target_len : ',target_len);
        print('source_len : ',source_len);
        print('-'*80);

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        iterations = 0;

        _stin = [];

        auto_encoder_epochs = 100;
        #############TRAIN DECODER AND ENCODER FIRST###########

        for epoch in range(auto_encoder_epochs):
            sin = source_train[np.random.choice(source_len, batch_size)]
            tin = target_train[np.random.choice(target_len, batch_size)]
            #sin = tin;
            sd = np.zeros((batch_size, 1)) #SOURCE IS Labelled 0
            td = np.ones((batch_size, 1)) #TARGET IS Labelled 1

            _stin = stin = np.concatenate((sin, tin), axis = 0)
            std = np.concatenate((sd, td), axis = 0)
            
            fd = {inpt:stin,domain:std}; 
            _,_recon_loss = sess.run([recon_op,recon_loss],feed_dict=fd);

            # fd = {inpt:stin[:batch_size],domain:std[:batch_size]}; 
            # _,_s_recon_loss = sess.run([mnist_recon_op,mnist_recon_loss],feed_dict=fd);

            

            if(epoch%20==0):
                #print("At epoch #",epoch,"(source)mnist reconstruction loss : ",_s_recon_loss," target reconstruction loss : ",_recon_loss);
                print("At epoch #",epoch," reconstruction loss : ",_recon_loss);

        print("-"*80);
        print("AE Training complete");
        print("-"*80);

        n = 5;
        random_source_batch = source_test[np.random.choice(source_len_test, n*n)];
        random_target_batch = target_val[np.random.choice(target_len_val, n*n)];
        
        source_fd = {inpt:random_source_batch};
        target_fd = {inpt:random_target_batch};

        recon_source = sess.run(recons_dec_test ,feed_dict=source_fd);

        # print(recon_source.shape);

        # recon_source[:,:,:,1] = recon_source[:,:,:,0];
        # recon_source[:,:,:,2] = recon_source[:,:,:,0]; 

        #sys.exit(0);

        # mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
        # mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

        #recon_source = recon_source.reshape(10000, 28, 28, 1).astype(np.uint8) * 255
        #recon_source = np.concatenate([recon_source[:,:,:,0], recon_source[:,:,:,0], recon_source[:,:,:,0]], 3)

        recon_target = sess.run(recons_dec_test ,feed_dict=target_fd);
        
        reconstructed_source = np.empty((28*n,28*n,3));
        original_source = np.empty((28*n,28*n,3));
        reconstructed_target = np.empty((28*n,28*n,3));
        original_target = np.empty((28*n,28*n,3));
        
        for i in range(n):
            for j in range(n):
                original_source[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = random_source_batch[i*n+j].reshape([28, 28,3]);
                reconstructed_source[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = recon_source[i*n+j].reshape([28, 28,3]);

                original_target[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = random_target_batch[i*n+j].reshape([28, 28,3]);
                reconstructed_target[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = recon_target[i*n+j].reshape([28, 28,3]);
            
        print("Original_source Images");
        plt.figure(figsize=(n, n));
        plt.imshow(original_source, origin="upper",interpolation='nearest', cmap="gray");
        plt.savefig('op/ae-orig-img-source.png');
        plt.close();

        print("Reconstructed_source Images");
        plt.figure(figsize=(n, n));
        plt.imshow(reconstructed_source, origin="upper",interpolation='nearest', cmap="gray");
        plt.savefig('op/ae-recons-img-source.png');
        plt.close();

        print("Original_target Images");
        plt.figure(figsize=(n, n));
        plt.imshow(original_target, origin="upper",interpolation='nearest', cmap="gray");
        plt.savefig('op/ae-orig-img-target.png');
        plt.close();

        print("Reconstructed_target Images");
        plt.figure(figsize=(n, n));
        plt.imshow(reconstructed_target, origin="upper",interpolation='nearest', cmap="gray");
        plt.savefig('op/ae-recons-img-target.png');
        plt.close();

        # #############MAIN TRAINING################

        for epoch in range(epochs):

            n_batches = int(55000.0/batch_size);
            print("For batch_size : ",batch_size,' n_batches : ',n_batches);

            #n_batches = 1;
            for batch in range(n_batches):
                iterations += 1;
                #picks randomly training instances
                #sin = source[np.random.choice(len(target), batch_size)]

                ### SCHEDULE ###



                #regularize the encoders
                n=1;
                for p in range(n):
                    sin = source_train[np.random.choice(source_len, batch_size)]
                    tin = target_train[np.random.choice(target_len, batch_size)]
                    sd = np.zeros((batch_size, 1)) #SOURCE IS Labelled 0
                    td = np.ones((batch_size, 1)) #TARGET IS Labelled 1

                    _stin = stin = np.concatenate((sin, tin), axis = 0)
                    std = np.concatenate((sd, td), axis = 0)
                    
                    fd = {inpt:stin,domain:std}; 
                    _,_recon_loss = sess.run([recon_op,recon_loss],feed_dict=fd);
                

                # o=2;
                # for p in range(o):
                #     sin = source_train[np.random.choice(source_len, batch_size)]
                #     tin = target_train[np.random.choice(target_len, batch_size)]
                #     sd = np.zeros((batch_size, 1)) #SOURCE IS Labelled 0
                #     td = np.ones((batch_size, 1)) #TARGET IS Labelled 1

                #     _stin = stin = np.concatenate((sin, tin), axis = 0)
                #     std = np.concatenate((sd, td), axis = 0)
                    
                #     fd = {inpt:stin,domain:std}; 
                #     _,_invar_recon_loss= sess.run([invar_recon_op,invar_recon_loss],feed_dict=fd);

                #learn domain discriminator
                k=1;
                for p in range(k):
                    sin = source_train[np.random.choice(source_len, batch_size)]
                    tin = target_train[np.random.choice(target_len, batch_size)]

                    sd = np.zeros((batch_size, 1)) #SOURCE IS Labelled 0
                    td = np.ones((batch_size, 1)) #TARGET IS Labelled 1

                    _stin = stin = np.concatenate((sin, tin), axis = 0)
                    std = np.concatenate((sd, td), axis = 0)

                    # print('_stin.shape : ',_stin.shape);
                    # print('std.shape : ',std.shape);

                    # plt.imshow(_stin[0], origin="upper",interpolation='nearest', cmap="gray");
                    # plt.show(); #Source data point
                    # plt.imshow(_stin[-1], origin="upper",interpolation='nearest', cmap="gray");
                    # plt.show(); #Target data point
                    
                    fd = {inpt:stin,domain:std}; 
                    _,_disc_loss = sess.run([disc_op,discriminator_loss],feed_dict=fd);
                
                #confuse domain discriminator
                l=1;
                for p in range(l):
                    sin = source_train[np.random.choice(source_len, batch_size)]
                    tin = target_train[np.random.choice(target_len, batch_size)]
                    sd = np.zeros((batch_size, 1)) #SOURCE IS Labelled 0
                    td = np.ones((batch_size, 1)) #TARGET IS Labelled 1

                    _stin = stin = np.concatenate((sin, tin), axis = 0)
                    std = np.concatenate((sd, td), axis = 0)
                    
                    fd = {inpt:stin,domain:std}; 
                    _,_enc_inv_loss = sess.run([enc_inv_op,inv_encoder_loss],feed_dict=fd);

                #help domain discriminator
                m=1;
                for p in range(m):
                    sin = source_train[np.random.choice(source_len, batch_size)]
                    tin = target_train[np.random.choice(target_len, batch_size)]
                    sd = np.zeros((batch_size, 1)) #SOURCE IS Labelled 0
                    td = np.ones((batch_size, 1)) #TARGET IS Labelled 1

                    _stin = stin = np.concatenate((sin, tin), axis = 0)
                    std = np.concatenate((sd, td), axis = 0)
                    
                    fd = {inpt:stin,domain:std}; 
                    _,_enc_var_loss,_merged_all = sess.run([enc_var_op,var_encoder_loss,merged_all],feed_dict=fd);

                if(iterations%20==0):
                    writer.add_summary(_merged_all,iterations);

                if(batch%100==0):
                    print("Batch #",batch," Done !");

            print('-'*80);
            print("Epoch #",epoch," completed !!");
            print('-'*80);

            if(epoch%5==0):
                #_stin = _stin.reverse();
                n = 5;
                random_source_batch = source_test[np.random.choice(source_len_test, n*n)];
                random_target_batch = target_val[np.random.choice(target_len_val, n*n)];
                
                source_fd = {inpt:random_source_batch};
                target_fd = {inpt:random_target_batch};

                # var_recon_source = sess.run(var_dec_test ,feed_dict=source_fd);
                # var_recon_target = sess.run(var_dec_test ,feed_dict=target_fd);

                # invar_recon_source = sess.run(invar_dec_test ,feed_dict=source_fd);
                # invar_recon_target = sess.run(invar_dec_test ,feed_dict=target_fd);

                recon_source = sess.run(recons_dec_test ,feed_dict=source_fd);
                recon_target = sess.run(recons_dec_test ,feed_dict=target_fd);
                
                # var_reconstructed_source = np.empty((28*n,28*n,3));
                # invar_reconstructed_source = np.empty((28*n,28*n,3));
                reconstructed_source = np.empty((28*n,28*n,3));
                original_source = np.empty((28*n,28*n,3));
                reconstructed_target = np.empty((28*n,28*n,3));
                # var_reconstructed_target = np.empty((28*n,28*n,3));
                # invar_reconstructed_target = np.empty((28*n,28*n,3));
                original_target = np.empty((28*n,28*n,3));
                for i in range(n):
                    for j in range(n):
                        original_source[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = random_source_batch[i*n+j].reshape([28, 28,3]);
                        # var_reconstructed_source[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = var_recon_source[i*n+j].reshape([28, 28,3]);
                        # invar_reconstructed_source[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = invar_recon_source[i*n+j].reshape([28, 28,3]);
                        reconstructed_source[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = recon_source[i*n+j].reshape([28, 28,3]);

                        original_target[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = random_target_batch[i*n+j].reshape([28, 28,3]);
                        # var_reconstructed_target[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = var_recon_target[i*n+j].reshape([28, 28, 3]);
                        # invar_reconstructed_target[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = invar_recon_target[i*n+j].reshape([28, 28,3]);
                        reconstructed_target[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = recon_target[i*n+j].reshape([28, 28,3]);
                    
                print("Original_source Images");
                plt.figure(figsize=(n, n));
                plt.imshow(original_source, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig('op/orig-img-'+str(epoch)+'-source.png');
                plt.close();

                print("Reconstructed_source Images");
                plt.figure(figsize=(n, n));
                plt.imshow(reconstructed_source, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig('op/recons-img-'+str(epoch)+'-source.png');
                plt.close();

                # print("Variant Reconstructed_source Images");
                # plt.figure(figsize=(n, n));
                # plt.imshow(var_reconstructed_source, origin="upper",interpolation='nearest', cmap="gray");
                # plt.savefig('op/var-recons-img-'+str(epoch)+'-source.png');
                # plt.close();

                # print("Invariant Reconstructed_source Images");
                # plt.figure(figsize=(n, n));
                # plt.imshow(invar_reconstructed_source, origin="upper",interpolation='nearest', cmap="gray");
                # plt.savefig('op/invar-recons-img-'+str(epoch)+'-source.png');
                # plt.close();

                print("Original_target Images");
                plt.figure(figsize=(n, n));
                plt.imshow(original_target, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig('op/orig-img-'+str(epoch)+'-target.png');
                plt.close();

                print("Reconstructed_target Images");
                plt.figure(figsize=(n, n));
                plt.imshow(reconstructed_target, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig('op/recons-img-'+str(epoch)+'-target.png');
                plt.close();

                # print("Variant Reconstructed_target Images");
                # plt.figure(figsize=(n, n));
                # plt.imshow(var_reconstructed_target, origin="upper",interpolation='nearest', cmap="gray");
                # plt.savefig('op/var-recons-img-'+str(epoch)+'-target.png');
                # plt.close();
                
                # print("Invariant Reconstructed_target Images");
                # plt.figure(figsize=(n, n));
                # plt.imshow(invar_reconstructed_target, origin="upper",interpolation='nearest', cmap="gray");
                # plt.savefig('op/invar-recons-img-'+str(epoch)+'-target.png');
                # plt.close();
                
                # imsave("op/s_orig_"+str(epoch)+".jpg",_stin[0,:,:,0]);
                # imsave("op/s_rec_"+str(epoch)+".jpg", recon[0,:,:,0]);
                # imsave("op/t_orig_"+str(epoch)+".jpg",_stin[-1,:,:,0]);
                # imsave("op/t_rec_"+str(epoch)+".jpg", recon[-1,:,:,0]);
                saver.save(sess, checkpoint_path)

else:
    print("Inference mode activated :)");
    clf_train_epochs = 20;
    clf_checkpoint_path = './model_with_classifier'
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());

        #params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES);
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='invariant_encoder');
        saver = tf.train.Saver(var_list=params);

        path_for_weights = checkpoint_path;

        try:
            saver.restore(sess, path_for_weights);
        except:
            #print("Previous weights not found of invariant encoder !"); 
            print("Previous weights not found of model"); 
            sys.exit(0);

        print ("Full model loaded successfully :)");

        source_train,source_test = load_MNIST_data();
        target_train,target_test,target_val = load_m_mnist_data();
        
        source_len = len(source_train);
        source_labels = mnist.train.labels;

        target_len = len(target_test);
        tgt_labels = mnist.test.labels;

        plt.imshow(source_train[1], origin="upper",interpolation='nearest', cmap="gray");
        plt.show()
        print(np.argmax(source_labels[1]))

        plt.imshow(target_test[1], origin="upper",interpolation='nearest', cmap="gray");
        plt.show()
        print(np.argmax(tgt_labels[1]))


        for epoch in range(clf_train_epochs):
            
            clf_batch_size = 128;
            n_batches = int(1.0*source_len/clf_batch_size);
            print('n_batches : ',n_batches,' for batch_size : ',clf_batch_size);

            _train_accuracy = 0.0;
            ittt=0;
            for batch in range(n_batches):
                ittt +=1;
                random_indexes = np.random.choice(source_len, clf_batch_size);
                src_X = source_train[random_indexes];
                src_Y = source_labels[random_indexes];
                
                src_fd = {inpt:src_X,src_labels:src_Y};
                _,_src_clf_loss,_temp_train_accuracy = sess.run([clf_op,clf_loss,train_accuracy],feed_dict=src_fd);
                _train_accuracy += _temp_train_accuracy;

                if(batch%100==0):
                    print("Batch #",batch," Done!");
                #writer.add_summary(_accuracy_logistic,ittt);

            
            print("Epoch #",epoch," Done with train accuracy : ,",(_train_accuracy/n_batches)*100," !!"); 
            #writer.add_summary(_merged_all,iterations);
            if(epoch%2==0):

                target_random_indexes = np.random.choice(target_len, target_len);

                target_X = target_test[target_random_indexes];
                target_Y = tgt_labels[target_random_indexes];
        
                target_fd = {inpt:target_X,target_labels:target_Y};
                
                _test_accuracy = sess.run(test_accuracy,feed_dict=target_fd);
                
                
                #_src_clf_loss is loss corresponding to last epoch
                print(" Test accuracy : ",_test_accuracy*100);
                saver.save(sess, clf_checkpoint_path)
