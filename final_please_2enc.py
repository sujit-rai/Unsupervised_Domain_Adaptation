import tensorflow as tf 
import numpy as np 
import pickle as pkl
from sklearn.manifold import TSNE
from scipy.misc import imsave
import sys
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

tf.set_random_seed(0);
random.seed(0);

src_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name="src_input")
tar_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name="tar_input")
X = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name="input")
X_target = tf.placeholder(tf.float32,[None,28,28,3]);
domain = tf.placeholder(tf.float32, shape=[None, 1], name="domain")
src_labels = tf.placeholder(tf.float32,shape=[None,10]);
target_labels = tf.placeholder(tf.float32,shape=[None,10]);
epochs = 100;
batch_size = 128

#DCGAN 
#initializer = tf.truncated_normal_initializer(mean=0, stddev=0.02);#
initializer = tf.contrib.layers.xavier_initializer()

#tuning knobs
z_dim = 128;
enc1_z = enc2_z = int(z_dim/2.0);
enc_dec_lr = 0.0005;
disc_lr = 0.0001;

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

def src_encoder(x,isTrainable=True,reuse=False):
    with tf.variable_scope("src_encoder") as scope:
        if reuse:
            scope.reuse_variables();

        #28x28x3
        conv1 = tf.layers.conv2d(x, 64, 5, strides = 2, padding="SAME", 
            kernel_initializer=initializer, activation=None,trainable=isTrainable,reuse=reuse,name='conv1_layer');
        conv1 = tf.layers.batch_normalization(conv1,name='conv1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        conv1 = tf.nn.relu(conv1);

        #14x14x64
        conv2 = tf.layers.conv2d(conv1, 128, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=None,trainable=isTrainable,reuse=reuse,name='conv2_layer')
        conv2 = tf.layers.batch_normalization(conv2,name='conv2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        conv2 = tf.nn.relu(conv2);
        #7x7xx128 
        conv3 = tf.layers.conv2d(conv2, 256, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=None,trainable=isTrainable,reuse=reuse,name='conv3_layer')
        conv3 = tf.layers.batch_normalization(conv3,name='conv3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        conv3 = tf.nn.relu(conv3);

        conv4 = tf.layers.conv2d(conv3, 512, 5, strides = 1, padding="SAME",
            kernel_initializer=initializer, activation=None,trainable=isTrainable,reuse=reuse,name='conv4_layer')
        conv4 = tf.layers.batch_normalization(conv4,name='conv4_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        conv4 = tf.nn.relu(conv4);

        #4x4x256
        fc = tf.contrib.layers.flatten(conv4)
        fc1 = tf.layers.dense(fc, units = enc1_z, 
            kernel_initializer= initializer, activation = tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc_layer')

        return fc1

def tar_encoder(x,isTrainable=True,reuse=False):
    with tf.variable_scope("tar_encoder") as scope:
        if reuse:
            scope.reuse_variables();

        #28x28x3
        conv1 = tf.layers.conv2d(x, 64, 5, strides = 2, padding="SAME", 
            kernel_initializer=initializer, activation=None,trainable=isTrainable,reuse=reuse,name='conv1_layer');
        conv1 = tf.layers.batch_normalization(conv1,name='conv1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        conv1 = tf.nn.relu(conv1);

        #14x14x64
        conv2 = tf.layers.conv2d(conv1, 128, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=None,trainable=isTrainable,reuse=reuse,name='conv2_layer')
        conv2 = tf.layers.batch_normalization(conv2,name='conv2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        conv2 = tf.nn.relu(conv2);
        #7x7xx128 
        conv3 = tf.layers.conv2d(conv2, 256, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=None,trainable=isTrainable,reuse=reuse,name='conv3_layer')
        conv3 = tf.layers.batch_normalization(conv3,name='conv3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        conv3 = tf.nn.relu(conv3);

        conv4 = tf.layers.conv2d(conv3, 512, 5, strides = 1, padding="SAME",
            kernel_initializer=initializer, activation=None,trainable=isTrainable,reuse=reuse,name='conv4_layer')
        conv4 = tf.layers.batch_normalization(conv4,name='conv4_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        conv4 = tf.nn.relu(conv4);

        #4x4x256
        fc = tf.contrib.layers.flatten(conv4)
        fc1 = tf.layers.dense(fc, units = enc1_z, 
            kernel_initializer= initializer, activation = tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc_layer')

        return fc1

# def classifier(z,isTrainable=True,reuse=False):
#   with tf.variable_scope("classifier") as scope:
#       if reuse:
#           scope.reuse_variables();

#         clf_logits = tf.layers.dense(z, units = 10,
#             kernel_initializer = initializer, activation = None,trainable=isTrainable,reuse=reuse,name='classifier_fc3_layer') 

#         return clf_logits;


def decoder(z,isTrainable=True,reuse=False):
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables();

        fc = tf.layers.dense(z, units = 4*4*512, 
            kernel_initializer = initializer, activation = None,trainable=isTrainable,reuse=reuse)
        fc = tf.layers.batch_normalization(fc,name='fc_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        fc = tf.nn.relu(fc);
        fl = tf.reshape(fc, [-1, 4, 4, 512]);

        #4x4x512
        deconv1 = tf.layers.conv2d_transpose(fl, 256, 5, strides=2, padding="SAME", 
            kernel_initializer=initializer, activation = None,trainable=isTrainable,reuse=reuse,name='deconv1_layer')
        deconv1 = tf.layers.batch_normalization(deconv1,name='deconv1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        deconv1 = tf.nn.relu(deconv1);

        #print("dec_deconv1.shape : ",deconv1.shape);

        #8x8x256
        deconv2 = tf.layers.conv2d_transpose(deconv1, 128, 5, strides=2, padding="SAME", 
            kernel_initializer=initializer, activation = None,trainable=isTrainable,reuse=reuse,name='deconv2_layer')
        deconv2 = tf.layers.batch_normalization(deconv2,name='deconv2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        deconv2 = tf.nn.relu(deconv2);


        #16x16x128
        deconv3 = tf.layers.conv2d_transpose(deconv2, 64, 5, strides=2, padding="SAME", 
            kernel_initializer=initializer, activation = None,trainable=isTrainable,reuse=reuse,name='deconv3_layer')
        deconv3 = tf.layers.batch_normalization(deconv3,name='deconv3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        deconv3 = tf.nn.relu(deconv3);


        #32x32x64
        deconv4 = tf.layers.conv2d_transpose(deconv3, 16, 5, strides=1, padding="SAME", 
            kernel_initializer=initializer, activation = None,trainable=isTrainable,reuse=reuse,name='deconv4_layer')
        deconv4 = tf.layers.batch_normalization(deconv4,name='deconv4_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        deconv4 = tf.nn.relu(deconv4);

        #32x32x16
        deconv5 = tf.layers.conv2d_transpose(deconv4, 3, 5, strides=1, padding="SAME", 
            kernel_initializer=initializer, activation = None,trainable=isTrainable,reuse=reuse,name='deconv5_layer')
        deconv5 = tf.layers.batch_normalization(deconv5,name='deconv5_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        deconv5 = tf.nn.relu(deconv5);
        #32x32x3
        deconv5 = tf.layers.flatten(deconv5);
        deconv_fc = tf.layers.dense(deconv5,28*28*3,activation=tf.nn.sigmoid,trainable=isTrainable,reuse=reuse,name='dec_dense_fc_last_layer');

        deconv_fc_reshaped = tf.reshape(deconv_fc,[-1,28,28,3]); 

        return deconv_fc_reshaped;

def discriminator(x,isTrainable=True,reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables();

        #28x28x3
        conv1 = tf.layers.conv2d(x, 64, 5, strides = 2, padding="SAME", 
            kernel_initializer=initializer, activation=None,trainable=isTrainable,reuse=reuse,name='conv1_layer');
        conv1 = tf.layers.batch_normalization(conv1,name='conv1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        conv1 = tf.nn.relu(conv1);

        #14x14x64
        conv2 = tf.layers.conv2d(conv1, 128, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=None,trainable=isTrainable,reuse=reuse,name='conv2_layer')
        conv2 = tf.layers.batch_normalization(conv2,name='conv2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        conv2 = tf.nn.relu(conv2);
        #7x7xx128 
        conv3 = tf.layers.conv2d(conv2, 256, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=None,trainable=isTrainable,reuse=reuse,name='conv3_layer')
        conv3 = tf.layers.batch_normalization(conv3,name='conv3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        conv3 = tf.nn.relu(conv3);

        #4x4x256
        conv3_flt = tf.contrib.layers.flatten(conv3)
        disc_logits = tf.layers.dense(conv3_flt, units = 1, 
            kernel_initializer= initializer, activation = None,trainable=isTrainable,reuse=reuse,name='fc_layer')

        return conv3_flt,disc_logits;

latent_code_src = src_encoder(src_x);
latent_code_tar = tar_encoder(tar_x)

src_tilde = decoder(latent_code_src);
tar_tilde = decoder(latent_code_tar, reuse = True)

latent_code_test = src_encoder(X,isTrainable=False,reuse=True);
X_tilde_test_src = decoder(latent_code_test,isTrainable=False,reuse=True);
X_tilde_test_tar = decoder(latent_code_test, isTrainable=False, reuse = True)

recon_og = tf.concat([src_x, tar_x], axis=0)
recon = tf.concat([src_tilde, tar_tilde], axis = 0)

recon_loss = tf.reduce_mean(tf.square(recon - recon_og));

target_X_representation,real_X_disc_logits = discriminator(tar_x);
fake_X_representation,fake_X_tilde_logits = discriminator(recon,reuse=True);
source_X_representation,fake_X_src_logits = discriminator(src_x,reuse=True);

#Say target is labelled as 1
disc_loss_X_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_X_disc_logits,labels=tf.ones_like(real_X_disc_logits)));
disc_loss_X_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_X_tilde_logits,labels=tf.zeros_like(fake_X_tilde_logits)));
disc_loss_X_src_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_X_src_logits,labels=tf.zeros_like(fake_X_src_logits)));

gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_X_tilde_logits,labels=tf.ones_like(fake_X_tilde_logits)));

#disc_feature_loss_style = tf.reduce_mean(tf.square(target_X_representation - fake_X_representation));
#disc_feature_loss_content = tf.reduce_mean(tf.square(fake_X_representation - source_X_representation));

alpha = 0.2;
beta = 0.5;
enc_dec_loss = alpha*recon_loss + 0.1*gan_loss;

#enc_dec_loss = gan_loss + alpha*disc_feature_loss_style;# + beta*recon_loss;

disc_loss = disc_loss_X_fake + disc_loss_X_real + disc_loss_X_src_fake;

src_enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='src_encoder');
tar_enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='tar_encoder')
dec_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='decoder');
enc_dec_params = src_enc_params + tar_enc_params + dec_params;
disc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator');


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS);
with tf.control_dependencies(update_ops):
    enc_dec_op = tf.train.AdamOptimizer(learning_rate = enc_dec_lr).minimize(enc_dec_loss, var_list = enc_dec_params);
    disc_op = tf.train.AdamOptimizer(learning_rate = disc_lr).minimize(disc_loss, var_list = disc_params);

tf.summary.scalar("reconstruction_loss", alpha*recon_loss)
#tf.summary.scalar("disc_feature_loss_style",alpha*disc_feature_loss_style);
#tf.summary.scalar("recon_loss_content",beta*recon_loss);
tf.summary.scalar("gan_loss", gan_loss)
tf.summary.scalar("discriminator_loss", disc_loss);
#tf.summary.scalar("disc_feature_loss", beta*disc_feature_loss);

merged_all = tf.summary.merge_all()

log_dir = "feature_extractr_3"

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


checkpoint_path = "./feature_extractr_3"

def train():
    with tf.Session() as sess:
        sess.run(init);
        source_train,source_test = load_MNIST_data();
        target_train,target_test,target_val = load_m_mnist_data();

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

        n_batches = int(1.0*source_len/batch_size);
        print('n_batches : ',n_batches);
        iterations = 0;
        for epoch in range(epochs):
            #n_batches = 0;
            for batch in range(n_batches):
                iterations += 1;
                #Train discriminiator, k times
                k=1;
                for i in range(k):
                    sin = source_train[np.random.choice(source_len, batch_size)]
                    tin = target_train[np.random.choice(target_len, batch_size)]
                    fd = {src_x: sin, tar_x: tin};                                
                    _,_disc_loss = sess.run([disc_op, disc_loss],feed_dict=fd);

                #Train enc_dec jointly, m times
                m=1;
                for i in range(1):
                    sin = source_train[np.random.choice(source_len, batch_size)]
                    tin = target_train[np.random.choice(target_len, batch_size)]
                    
                    fd = {src_x: sin, tar_x: tin};                                
                    #fd = {X:sin, X_target:tin};
                    _,_enc_dec_loss,_merged_all = sess.run([enc_dec_op, enc_dec_loss, merged_all],feed_dict=fd);                                

                if(iterations%20==0):
                    writer.add_summary(_merged_all,iterations);

                if(batch%100==0):
                    print("Batch #",batch," Done !");

            print('-'*80);
            print("Epoch #",epoch," completed !!");
            print('-'*80);

            if(epoch%2==0):

                n = 5;
                random_source_batch = source_test[np.random.choice(source_len_test, n*n)];
                random_target_batch = target_val[np.random.choice(target_len_val, n*n)];

                source_fd = {X:random_source_batch};
                target_fd = {X:random_target_batch};

                recon_source = sess.run(X_tilde_test_src ,feed_dict=source_fd);
                recon_target = sess.run(X_tilde_test_tar ,feed_dict=target_fd);
                
                # var_reconstructed_source = np.empty((28*n,28*n,3));
                # invar_reconstructed_source = np.empty((28*n,28*n,3));
                reconstructed_source = np.empty((28*n,28*n,3));
                original_source = np.empty((28*n,28*n,3));
                for i in range(n):
                    for j in range(n):
                        original_source[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = random_source_batch[i*n+j].reshape([28, 28,3]);
                        reconstructed_source[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = recon_source[i*n+j].reshape([28, 28,3]);
                    
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

                reconstructed_target = np.empty((28*n,28*n,3));
                original_target = np.empty((28*n,28*n,3));
                for i in range(n):
                    for j in range(n):
                        original_target[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = random_target_batch[i*n+j].reshape([28, 28,3]);
                        reconstructed_target[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28,:] = recon_target[i*n+j].reshape([28, 28,3]);
                    
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

                saver.save(sess, checkpoint_path)

train();