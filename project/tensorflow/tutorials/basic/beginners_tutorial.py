import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
#create a place for data input
y_ = tf.placeholder(tf.float32, [None, 10])
#create a place for data class

W1 = tf.Variable(tf.truncated_normal([784, 784], stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([784], stddev=0.1))

W2 = tf.Variable(tf.truncated_normal([784,10], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([10], stddev=0.1))

y  = tf.nn.softmax(tf.matmul(tf.matmul(x,W1) + b1,W2) + b2)

cross_entropy = - tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.00005).minimize(cross_entropy)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(200)
  #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  if i % 25 == 0:
    print("%.3f" % sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

train_step = tf.train.GradientDescentOptimizer(0.000015).minimize(cross_entropy)
for i in range(6000):
  batch_xs, batch_ys = mnist.train.next_batch(500)
  #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#basic workflow is 
# (1) create netowrk;
# (2) get data (format data); 
# (3) construct loss mechanism;
# (4) construct training mechanism (i.e. gradient desecent, stocastic gradient descent, etc.)
# (5) initialize weights;
# (6) feed data into the session;
# (7) evaluate accuracy;