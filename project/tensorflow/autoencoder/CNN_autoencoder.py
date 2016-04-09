import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

import numpy as np
import matplotlib.pyplot as plt

import pickle
import time
import sys

sess = tf.Session()

#helper methods
def weight_variable(shape):
	initial = tf.truncated_normal(shape=shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#conv2d_transpose is the gradient fo the conv


#strides:
# a list of ints that has length>=4. The stride of the sliding window 
#for each dimension of the input tensor
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],
	                      padding='SAME')


n_train      = mnist.train.num_examples
n_validation = mnist.validation.num_examples
n_test       = mnist.test.num_examples

#use initialization recommendation:
# Paper: X. Glorot and Y. Bengio, 
# “Understanding the difficulty of training deep feedforward neural networks,” in International conference on artificial intelligence and statistics, 2010, pp. 249–256.
#http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
def xavier_uniform_initialization(fan_in, fan_out, gain = np.sqrt(2.0)):
	""" Xavier initialization of network weights """
	low = -gain*np.sqrt(6.0/(fan_in + fan_out))
	#6.0 is here because the 
	high = gain*np.sqrt(6.0/(fan_in + fan_out))
	return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

def xavier_gaussian_initialization(fan_in, fan_out, gain = np.sqrt(2.0)):
	""" Xavier initialization of network weights """
	#1.0 for linear and sigmoid
	#sqrt(2.0) for ReLU
	#sqrt(2/(1+alpha**2)) for leaky ReLU with leakiness 'alpha'
	#sigma
	sigma = gain * np.sqrt(2.0 / (fan_in + fan_out))
	#f.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
	return tf.random_normal((fan_in,fan_out), 
	                        mean=0.0, stddev=sigma,
	                        dtype=tf.float32)

def weight_xavier_variable(shape):
	initial = xavier_gaussian_initialization(shape[0], shape[1], 1.0)
	return tf.Variable(initial)



#helper methods
def weight_variable(shape):
	initial = tf.truncated_normal(shape=shape, stddev=0.1)
	return tf.Variable(initial)

def gaussian_variable(shape):
	initial = tf.random_normal(shape)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

latent_size = 20

#image_placeholder
x  = tf.placeholder(tf.float32, shape=[None,784])
x_image = tf.reshape(x, shape=[-1, 28, 28, 1])

#class space
y_ = tf.placeholder(tf.float32, shape=[None, 10])
#y_ is the dimensionality of the classification space

#generated image placeholder
x_gen = tf.placeholder(tf.float32, shape=[784,None])

W_conv1 = weight_variable([3,3,1,128])
#f_x,f_y,depth, number of filters
b_conv1 = bias_variable([128])
cnn_layer_1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

W_conv2 = weight_variable([3,3,128,128])
b_conv2 = bias_variable([128])
cnn_layer_2 = tf.nn.relu(conv2d(cnn_layer_1, W_conv2) + b_conv2)

W_conv3 = weight_variable([3,3,128,128])
b_conv3 = bias_variable([128])
cnn_layer_3 = tf.nn.relu(conv2d(cnn_layer_2, W_conv3) + b_conv3)

#convert it to a flat distribution
cnn_layer_3_flat = tf.reshape(cnn_layer_3, [-1, 28*28*128])
W_flat_to_latent = weight_xavier_variable((28*28*128,latent_size))
b_flat_to_latent = bias_variable([latent_size])

h_latent = tf.matmul(cnn_layer_3_flat, W_flat_to_latent) + b_flat_to_latent

W_latent_to_decoder = weight_xavier_variable((latent_size,14*14*128))
b_latent_to_decoder = bias_variable([14*14*128])

h_flat_decoder1 = tf.nn.relu(tf.matmul(h_latent, W_latent_to_decoder) + b_latent_to_decoder)

W_decoder_1_to_2 = weight_xavier_variable((14*14*128,28*28))
b_decoder_1_to_2 = bias_variable([28*28])
x_gen            = tf.nn.relu(tf.matmul(h_flat_decoder1, W_decoder_1_to_2) + b_decoder_1_to_2)

x_gen_image = tf.reshape(x_gen, shape=[-1, 28, 28, 1])

input_image   = tf.image_summary("input image", x_image, max_images=2)
encoded_image = tf.image_summary("autoencoder image", x_gen_image, max_images=2)

loss = tf.reduce_mean(tf.square(x - x_gen), name="l2_loss")

summary_loss = tf.scalar_summary(loss.op.name, loss)
train_step   = tf.train.AdamOptimizer(1e-4).minimize(loss)

#input_image   = tf.image_summary("Validation Batch (image)",             x_image, max_images=3)
#encoded_image = tf.image_summary("Validation Batch (autoencoder image)", x_gen_image, max_images=3)

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("MNIST_CNN_autoencoder", sess.graph_def)

sess.run(tf.initialize_all_variables())

start = time.time()
loop_time = start

print("number of test       = ", n_test)      # 10000
print("number of train      = ", n_train)     # 55000
print("number_of validation = ", n_validation)# 5000
#print(dir(mnist.test))
chunk_size = 1000
batch_Xtest_list = [mnist.test.images[i:i+chunk_size] for i in range(0, n_test, chunk_size)]
batch_ytest_list = [mnist.test.labels[i:i+chunk_size] for i in range(0, n_test, chunk_size)]

print("Done splitting up test data set;")
print("Starting training loop.")

#train for 10 epochs 55000 * 10 / 100 
for i in range(5500):
	if i % 550 == 0:
		print("Epoch: ", i//550)
	batch = mnist.train.next_batch(100)
	if i % 10 == 0:
		validation_batch = mnist.validation.next_batch(100)
		feed_dict_val = {x: validation_batch[0], 
		                 y_ : validation_batch[1]}
		validation_results = sess.run([merged, loss, x_gen_image], feed_dict=feed_dict_val)
		print("STEP %d, validation %g" % (i, validation_results[1]))
		writer.add_summary(validation_results[0], i)
		# plt.subplot(2,1,1)
		# #copy_validation = validation_batch[0][0].reshape((28,28)).copy()
		# plt.imshow(validation_batch[0][0].reshape((28,28)))
		# plt.title("Input (top), AutoEncoder Reconstruction (bottom)")
		# plt.subplot(2,1,2)
		# plt.imshow(validation_results[2][0].reshape((28,28)))
		# plt.show()
		# plt.savefig('foo.png', bbox_inches='tight')
	train_step.run(session=sess, feed_dict={x:batch[0], y_: batch[1]})

#numpy_Wc1 =  W_conv1.eval(sess)

loss_estimate = 0.
for i in range(len(batch_Xtest_list)):
	loss_estimate += loss.eval(session=sess, feed_dict={x:batch_Xtest_list[i], y_: batch_ytest_list[i]})
print("final loss %g" % (loss_estimate/len(batch_Xtest_list)))
print("run time = %d" % (time.time() - start))

with open('Wc1.pkl','wb') as f:
	pickle.dump(W_conv1.eval(sess), f)

with open('W_conv2.pkl','wb') as f:
	pickle.dump(W_conv2.eval(sess), f)

with open('W_conv3.pkl','wb') as f:
	pickle.dump(W_conv3.eval(sess), f)

with open('W_flat_to_latent.pkl','wb') as f:
	pickle.dump(W_flat_to_latent.eval(sess), f)

with open('W_latent_to_decoder.pkl', 'wb') as f:
	pickle.dump(W_latent_to_decoder.eval(sess), f)

with open('W_decoder_1_to_2.pkl', 'wb') as f:
	pickle.dump(W_decoder_1_to_2.eval(sess), f)

print("Done!")
