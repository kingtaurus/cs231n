import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
#mnist.train
#mnist.validation
#mnist.test

import matplotlib.pyplot as plt

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

#images_placeholder
x  = tf.placeholder(tf.float32, shape=[None, 784])

#labels_placeholder
y_ = tf.placeholder(tf.float32, shape=[None,10])

W_conv1 = weight_variable([3,3,1,64])
#f_x,f_y,depth, number of filters
b_conv1 = bias_variable([64])

x_image = tf.reshape(x, [-1, 28, 28, 1])
#reshape the input to have the correct shape (28,28,1)

#hidden layer
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# at this stage:
#RULE, (X_1 - F_x) / S_x + 1 = X_2
# 1,28,28,64 -> 
# 28 -> (28 - F)/ S + 1 = 14
#Step 2
# 14 -> (14 - F)/S + 1 = 7

W_conv2 = weight_variable([3,3,64,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([3,3,64,64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_conv3_flat = tf.reshape(h_conv3, [-1, 7*7*64])
#downsampled

W_fc1 = weight_variable([7*7*64,512])
b_fc1 = weight_variable([512])

h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

#adding dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#output layer
W_fc2 = weight_variable([512,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy =  tf.reduce_sum(-1 * y_ * tf.log(y_conv), name='cross_entropy_sum')
loss          =  tf.reduce_mean(y_ * tf.log(y_conv), name='cross_entropy_mean')


w_hist       = tf.histogram_summary("weights_conv1", W_conv1)

w_image      = tf.image_summary("filter output",h_conv1[:,:,:,0:1])
#second_image = tf.image_summary("first filter", tf.reshape(W_conv1, [64,3,3,1]))
second_image = tf.image_summary("first filter", tf.transpose(W_conv1, perm=[3,0,1,2]), max_images=1)
#original shape (3,3,1,64) -> (64,3,3,1)

summ_loss    = tf.scalar_summary(loss.op.name, loss)
summ_cross   = tf.scalar_summary(cross_entropy.op.name, cross_entropy)
#summ_spars1  = tf.scalar_summary('sparsity_W_conv1', tf.nn.zero_fraction(W_conv1))
#summ_spars2  = tf.scalar_summary('sparsity_W_conv2', tf.nn.zero_fraction(W_conv2))
#summ_spars3  = tf.scalar_summary('sparsity_W_conv3', tf.nn.zero_fraction(W_conv3))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("MNIST_test_summary", sess.graph_def)

sess.run(tf.initialize_all_variables())

for i in range(10000):
	batch = mnist.train.next_batch(50)
	if i % 100 == 0:
		#get a new validation batch
		validation_batch = mnist.validation.next_batch(500)
		feed_dict_val = {x: validation_batch[0], 
		                 y_ : validation_batch[1], 
		                 keep_prob:1.0}
		validation_result = sess.run([merged, accuracy], feed_dict=feed_dict_val)
		train_results     = sess.run([merged, accuracy], feed_dict={x:batch[0], y_: batch[1], keep_prob:1.0})
		# train_accuracy      = accuracy.eval(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob:1.0})
		# validation_accuracy = accuracy.eval(session=sess, feed_dict=feed_dict_val)
		print("STEP %d, training_accuracy %g validation_accuracy %g" % (i, train_results[1], validation_result[1]))
		#writer.add_summary(summ_loss.eval(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob:1.0} ))
		writer.add_summary(validation_result[0], i)
		#writer.add_summary(train_results[0], i)
	train_step.run(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob:0.5})
	#sess.run([train_op, loss], feed_dict={x:batch[0], y_: batch[1], keep_prob:0.5})

train_accuracy = accuracy.eval(session=sess, feed_dict={x:mnist.test.images, y_: mnist.test.labels, keep_prob:1.0})
print("training_accuracy %g" % (train_accuracy))

#a = W_conv1.eval(sess)
#gives a numpy array of the weights for the current session

