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

X = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name="input")
X_target = tf.placeholder(tf.float32,[None,28,28,3]);
domain = tf.placeholder(tf.float32, shape=[None, 1], name="domain")
labels = tf.placeholder(tf.float32,shape=[None,10]);
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

def inv_encoder(x,isTrainable=True,reuse=False):
    with tf.variable_scope("inv_encoder") as scope:
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
        fc1 = tf.layers.dense(fc, units = enc1_z/2, 
            kernel_initializer= initializer, activation = tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc_layer')

        return fc1

def v_encoder(x,isTrainable=True,reuse=False):
    with tf.variable_scope("v_encoder") as scope:
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
        fc1 = tf.layers.dense(fc, units = enc1_z/2, 
            kernel_initializer= initializer, activation = tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc_layer')

        return fc1


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



def inv_dom_classifier(x,isTrainable=True,reuse=False):
    with tf.variable_scope("inv_dom_classifier") as scope:
        if reuse:
            scope.reuse_variables();

        fc1 = tf.layers.dense(x, 32,activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc1');
        fc2 = tf.layers.dense(fc1, 16,activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc2');
        fc3 = tf.layers.dense(fc2, 8,activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc3');
    
        disc_logits = tf.layers.dense(fc3, units = 1, 
            kernel_initializer= initializer, activation = None,trainable=isTrainable,reuse=reuse,name='fc_layer')

        return disc_logits;

def v_dom_classifier(x,isTrainable=True,reuse=False):
    with tf.variable_scope("v_dom_classifier") as scope:
        if reuse:
            scope.reuse_variables();

        fc1 = tf.layers.dense(x, 32,activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc1');
        fc2 = tf.layers.dense(fc1, 16,activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc2');
        fc3 = tf.layers.dense(fc2, 8,activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc3');
    
        disc_logits = tf.layers.dense(fc3, units = 1, 
            kernel_initializer= initializer, activation = None,trainable=isTrainable,reuse=reuse,name='fc_layer')

        return disc_logits;

def inv_cls_classifier(x,isTrainable=True,reuse=False):
    with tf.variable_scope("inv_cls_classifier") as scope:
        if reuse:
            scope.reuse_variables();

        fc1 = tf.layers.dense(x, 32,activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc1');
        fc2 = tf.layers.dense(fc1, 16,activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc2');
        fc3 = tf.layers.dense(fc2, 16,activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc3');
    
        disc_logits = tf.layers.dense(fc3, units = 10, 
            kernel_initializer= initializer, activation = None,trainable=isTrainable,reuse=reuse,name='fc_layer')

        return disc_logits;

def v_cls_classifier(x,isTrainable=True,reuse=False):
    with tf.variable_scope("v_cls_classifier") as scope:
        if reuse:
            scope.reuse_variables();

        fc1 = tf.layers.dense(x, 32,activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc1');
        fc2 = tf.layers.dense(fc1, 16,activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc2');
        fc3 = tf.layers.dense(fc2, 16,activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc3');
    
        disc_logits = tf.layers.dense(fc3, units = 10, 
            kernel_initializer= initializer, activation = None,trainable=isTrainable,reuse=reuse,name='fc_layer')

        return disc_logits;




inv_latent_code = inv_encoder(X)
inv_dom_logits = inv_dom_classifier(inv_latent_code)
inv_cls_logits = inv_cls_classifier(inv_latent_code)


v_latent_code = v_encoder(X)
v_dom_logits = v_dom_classifier(v_latent_code)
v_cls_logits = v_cls_classifier(v_latent_code)


latent_code = tf.concat([inv_latent_code, v_latent_code], axis=1)
x_tilde = decoder(latent_code)

inv_pred = tf.argmax(tf.nn.softmax(inv_cls_logits), axis=1)
inv_lab = tf.argmax(labels, axis=1)

inv_acc = tf.reduce_mean(tf.to_float(tf.equal(inv_pred, inv_lab)))


v_pred = tf.argmax(tf.nn.softmax(v_cls_logits), axis=1)
v_lab = tf.argmax(labels, axis=1)

v_acc = tf.reduce_mean(tf.to_float(tf.equal(v_pred, v_lab)))



recon_loss = tf.reduce_mean(tf.square(X - x_tilde))
inv_d_disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=inv_dom_logits, labels=domain))
inv_e_disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=inv_dom_logits, labels=0.5*tf.ones_like(inv_dom_logits)))
v_disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=v_dom_logits, labels=domain))
inv_cls_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=inv_cls_logits, onehot_labels=labels))
v_e_cls_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=v_cls_logits, onehot_labels=0.1*tf.ones_like(labels)))
v_d_cls_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=v_cls_logits, onehot_labels=labels))


inv_enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="inv_encoder")
v_enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="v_encoder")
inv_disc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="inv_dom_classifier")
v_disc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="v_dom_classifier")
inv_cls_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="inv_cls_classifier")
v_cls_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="v_cls_classifier")
dec_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")




recon_lr = 0.0001
inv_cls_lr = 0.0001
v_e_cls_lr = 0.0001
v_d_cls_lr = 0.0001
inv_e_d_lr = 0.0001
v_e_d_lr = 0.0001
inv_d_d_lr = 0.0001


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    recon_op = tf.train.AdamOptimizer(learning_rate = recon_lr).minimize(recon_loss,var_list=inv_enc_params + v_enc_params + dec_params)
    inv_cls_op = tf.train.AdamOptimizer(learning_rate = inv_cls_lr).minimize(inv_cls_loss,var_list=inv_enc_params + inv_cls_params)
    v_e_cls_op = tf.train.AdamOptimizer(learning_rate = v_e_cls_lr).minimize(v_e_cls_loss,var_list=v_enc_params)
    v_d_cls_op = tf.train.AdamOptimizer(learning_rate = v_d_cls_lr).minimize(v_d_cls_loss,var_list=v_cls_params)
    
    inv_e_d_op = tf.train.AdamOptimizer(learning_rate = inv_e_d_lr).minimize(inv_e_disc_loss,var_list=inv_enc_params)
    v_d_op = tf.train.AdamOptimizer(learning_rate = v_e_d_lr).minimize(v_disc_loss,var_list=v_enc_params + v_disc_params)
    inv_d_d_op = tf.train.AdamOptimizer(learning_rate = inv_d_d_lr).minimize(inv_d_disc_loss,var_list=inv_disc_params)
    


tf.summary.scalar("recon_loss",recon_loss)
tf.summary.scalar("inv_cls_loss",inv_cls_loss)
tf.summary.scalar("v_d_cls_loss",v_d_cls_loss)
tf.summary.scalar("v_e_cls_loss", v_e_cls_loss)
tf.summary.scalar("inv_e_disc_loss",inv_e_disc_loss)
tf.summary.scalar("v_disc_loss",v_disc_loss)
tf.summary.scalar("inv_d_disc_loss",inv_d_disc_loss)
tf.summary.scalar("inv_acc", inv_acc)
tf.summary.scalar("v_acc", v_acc)

merged_all = tf.summary.merge_all()

log_dir = "var_invar_2dc"

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

checkpoint_path = "./var_invar_2dc"

source_train, source_test = load_MNIST_data()
target_train, target_test, target_val = load_m_mnist_data()

source_train_labels = mnist.train.labels
source_test_labels = mnist.test.labels

target_train_labels = mnist.train.labels
target_test_labels = mnist.test.labels

target_len = len(target_train)
source_len = len(source_train)

source_len_test = len(source_test)
target_len_test = len(target_test)
target_len_val = len(target_val)
n_batches = int(1.0*source_len/batch_size)

def produce_batch(mode):
    s_choice = np.random.choice(source_len, batch_size)
    t_choice = np.random.choice(target_len, batch_size)
    sin = source_train[s_choice]
    tin = target_train[t_choice]
    slab = source_train_labels[s_choice]
    tlab = target_train_labels[t_choice]

    s_choice = np.random.choice(source_len_test, batch_size)
    t_choice = np.random.choice(target_len_test, batch_size)
    sin_t = source_test[s_choice]
    tin_t = target_test[t_choice]
    slab_t = source_test_labels[s_choice]
    tlab_t = target_test_labels[t_choice]

    
    if mode == "both":
        stin = np.concatenate((sin, tin), axis=0)
        dom = np.expand_dims(np.concatenate((np.ones(batch_size), np.zeros(batch_size)), axis=0),axis=-1)
        stlab = np.concatenate((slab, tlab), axis=0)
        return {X:stin, labels:stlab, domain: dom}

    elif mode == "single":
        return {X:sin, labels:slab, domain: np.expand_dims(np.ones(batch_size), axis=-1)}

    elif mode == "test_acc":
        return {X:tin, labels:tlab}

    else:
        stin_t = np.concatenate((sin_t, tin_t), axis=0)
        dom_t = np.expand_dims(np.concatenate((np.ones(batch_size), np.zeros(batch_size)), axis=0),axis=-1)
        stlab_t = np.concatenate((slab_t, tlab_t), axis=0)
        return {X:stin_t, labels:stlab_t, domain: dom_t}


def train():
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        iterations = 0

        for epoch in range(epochs):
            for batch in range(n_batches):
                iterations += 1
                _,_v_e_cls_loss,_v_acc = sess.run([v_e_cls_op, v_e_cls_loss, v_acc], feed_dict=produce_batch("both"))
                _,_v_d_cls_loss = sess.run([v_d_cls_op, v_d_cls_loss], feed_dict=produce_batch("single"))
                _,_v_disc_loss = sess.run([v_d_op, v_disc_loss], feed_dict=produce_batch("both"))
                _,_inv_e_disc_loss = sess.run([inv_e_d_op, inv_e_disc_loss], feed_dict=produce_batch("both"))
                _,_inv_d_disc_loss = sess.run([inv_d_d_op, inv_d_disc_loss], feed_dict=produce_batch("both"))
                _,_inv_cls_loss,_inv_acc = sess.run([inv_cls_op, inv_cls_loss, inv_acc], feed_dict=produce_batch("single"))
                _,_recon_loss,_merged_all = sess.run([recon_op, recon_loss, merged_all], feed_dict=produce_batch("both"))
                
                if(iterations%20 == 0):
                    writer.add_summary(_merged_all, iterations)

                if(batch%100 == 0):
                    print("batch #", batch," Done")

            print("-"*80)
            print("epoch #", epoch," completed")
            print("-"*80)

            if(epoch%1==0):                
                _latent_code_test, _inv_latent_code_test, _v_latent_code_test, _domain, _labels = sess.run([latent_code,inv_latent_code, v_latent_code, domain, labels],feed_dict=produce_batch("test"));
                _inv_acc = sess.run(inv_acc, feed_dict=produce_batch('test_acc'))
                print("Test acc : ", _inv_acc)
                np.savetxt('vectors.tsv', _latent_code_test, fmt="%i", delimiter="\t")
                np.savetxt('inv_vectors.tsv', _inv_latent_code_test, fmt="%i", delimiter="\t")
                np.savetxt('v_vectors.tsv', _v_latent_code_test, fmt="%i", delimiter="\t")
                np.savetxt('domain.tsv', _domain, fmt="%i", delimiter="\t")
                np.savetxt('labels.tsv', np.argmax(_labels, axis=1), fmt="%i", delimiter="\t")

            if(epoch%1==0):

                n = 5;
                random_source_batch = source_test[np.random.choice(source_len_test, n*n)];
                random_target_batch = target_val[np.random.choice(target_len_val, n*n)];

                source_fd = {X:random_source_batch};
                target_fd = {X:random_target_batch};

                recon_source = sess.run(x_tilde ,feed_dict=source_fd);
                recon_target = sess.run(x_tilde ,feed_dict=target_fd);
                
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
train()