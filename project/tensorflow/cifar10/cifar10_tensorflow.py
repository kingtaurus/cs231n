import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from data_utils import get_CIFAR10_data

import seaborn as sns
sns.set_style("darkgrid")

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from sklearn.metrics import confusion_matrix

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

import tensorflow as tf

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=None):
    if labels is None:
        labels = list(range(len(cm)))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.grid(b='off')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def activation_summaries(activation, name):
  with tf.name_scope("activation_summaries"):
    mean = tf.reduce_mean(activation)
    tf.histogram_summary(name + '/activations', activation)
    tf.scalar_summary(name + '/sparsity', tf.nn.zero_fraction(activation))
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(activation - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(activation))
    tf.scalar_summary('min/' + name, tf.reduce_min(activation))

def variable_summaries(variable, name):
  with tf.name_scope("variable_summaries"):
    mean = tf.reduce_mean(variable)
    tf.histogram_summary(name + '/variable_hist', variable)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(variable - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(variable))
    tf.scalar_summary('min/' + name, tf.reduce_min(variable))

LOSSES_COLLECTION  = 'regularizer_losses'
DEFAULT_REG_WEIGHT =  1e-4

#reg_placeholder = tf.placeholder(dtype=tf.float32, shape=[1])
## may want to add this to the inputs for rcl (and inference methods)
# with tf.op_scope([tensor], scope, 'L2Loss'):
#     weight = tf.convert_to_tensor(weight,
#                               dtype=tensor.dtype.base_dtype,
#                               name='loss_weight')
#     loss = tf.mul(weight, tf.nn.l2_loss(tensor), name='value')
#     tf.add_to_collection(LOSSES_COLLECTION, loss)
#     return loss

def conv_relu(layer_in, kernel_shape, bias_shape, name):
  with tf.variable_scope(name) as scope:
    kernel = tf.get_variable("W",
                             shape=kernel_shape,
                             initializer=tf.contrib.layers.xavier_initializer_conv2d())
    bias = tf.get_variable("b", shape=bias_shape, initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv2d(layer_in, kernel, strides=[1,1,1,1], padding='SAME')
    layer = tf.nn.relu(conv + bias)
    #variable_summaries(bias, bias.name)
    variable_summaries(kernel, kernel.name)
    activation_summaries(layer, layer.name)
    # layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')
  return layer

def fcl_relu(layer_in, output_size, name):
  with tf.variable_scope(name) as scope:
    dim = np.prod(layer_in.get_shape().as_list()[1:])
    reshape = tf.reshape(layer_in, [-1, dim])
    weights = tf.get_variable("W_fcl",
                              shape=[dim, output_size],
                              initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable("b_fcl",
                             shape=[output_size],
                             initializer=tf.constant_initializer(0.))
    layer = tf.nn.relu(tf.matmul(reshape, weights) + bias, name=scope.name + "_activation")
    variable_summaries(weights, weights.name)
    #variable_summaries(bias, bias.name)
    activation_summaries(layer, layer.name)
    regularizer_loss = tf.mul(DEFAULT_REG_WEIGHT, tf.nn.l2_loss(weights))
    tf.add_to_collection(LOSSES_COLLECTION, regularizer_loss)
  return layer

NUM_CLASSES = 10

def inference(images, classes = NUM_CLASSES):
  layer = conv_relu(images, [3,3,3,64], [64], "conv_1")
  layer = conv_relu(layer,  [3,3,64,64], [64], "conv_2")
  layer = conv_relu(layer,  [3,3,64,128], [128], "conv_3")
  # layer = conv_relu(layer,  [3,3,128,128], [128], "conv_4")
  # layer = conv_relu(layer,  [3,3,128,128], [128], "conv_5")
  # layer = conv_relu(layer,  [3,3,128,128], [128], "conv_6")
  layer = fcl_relu(layer, 32, "fcl_1")

  with tf.variable_scope('pre_softmax_linear') as scope:
    weights = tf.get_variable('weights',
                              shape=[32, classes],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', 
                             shape=[classes],
                             initializer=tf.constant_initializer(0.))
    pre_softmax_linear = tf.add(tf.matmul(layer, weights), biases, name=scope.name)
    variable_summaries(weights, weights.name)
    #variable_summaries(biases, biases.name)
    #activation_summaries(pre_softmax_linear, pre_softmax_linear.name)
    regularizer_loss = tf.mul(DEFAULT_REG_WEIGHT, tf.nn.l2_loss(weights))
    tf.add_to_collection(LOSSES_COLLECTION, regularizer_loss)
  return pre_softmax_linear

def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

  # The total loss is defined as the cross entropy loss
  return cross_entropy_mean

INITIAL_LEARNING_RATE = 0.005
LEARNING_RATE_DECAY_FACTOR = 0.90
BATCH_SIZE = 256
MAX_STEPS = 100000

DECAY_STEPS = 12 * BATCH_SIZE

def train(total_loss, global_step):
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  DECAY_STEPS,#number of steps required for it to decay
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)

  tf.scalar_summary('learning_rate', lr)

  with tf.control_dependencies([total_loss]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # #apply the gradients
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  for grad, var in grads:
    if grad is not None:
      print("Found gradients for: ", var.op.name)
      tf.histogram_summary(var.op.name + "/gradients", grad)

  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name="train")

  #opt = tf.train.GradientDescentOptimizer(lr).minimize(total_loss, global_step=global_step)
  
  # grads = opt.compute_gradients(total_loss)

  return train_op

def main():
  data = get_CIFAR10_data()
  for k, v in data.items():
      print('%s: '%(k), v.shape)

  #PLACEHOLDER VARIABLES
  keep_prob = tf.placeholder(dtype=tf.float32, shape=[1])
  learning_rate = tf.placeholder(dtype=tf.float32, shape=[1])
  regularizer_weight = tf.placeholder(dtype=tf.float32, shape=[1])
  #Not used --- ^ (currently)

  X_image = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
  y_label = tf.placeholder(dtype=tf.int64, shape=[None])

  #MODEL related operations and values
  global_step = tf.Variable(0, trainable=False)
  #MODEL construction
  logits  = inference(X_image)
  loss_op = loss(logits, y_label)

  reg_loss = tf.reduce_sum(tf.get_collection(LOSSES_COLLECTION))
  total_loss = loss_op + reg_loss

  accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1), y_label), tf.float32))
  train_op = train(loss_op, global_step)
  saver = tf.train.Saver(tf.all_variables())

  #Summary operation
  tf.image_summary('images', X_image)
  summary_op = tf.merge_all_summaries()

  acc_summary        = tf.scalar_summary('accuracy', accuracy_op)
  cross_entropy_loss = tf.scalar_summary('loss_raw', loss_op)
  reg_loss_summary   = tf.scalar_summary('regularization_loss', reg_loss)
  total_loss_summary = tf.scalar_summary('total_loss', total_loss)

  acc_val_summary = tf.scalar_summary('accuracy_validation', accuracy_op)

  #SESSION Construction
  init = tf.initialize_all_variables()
  sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=False))
  sess.run(init)

  train_dir = "cifar10_results"
  summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

  print("Starting Training.")
  for step in range(MAX_STEPS):
    num_train = data['X_train'].shape[0]
    if BATCH_SIZE * (step - 1) // num_train < BATCH_SIZE * (step) // num_train and step > 0:
      print("Completed Epoch: %d" % (BATCH_SIZE * (step) // num_train ))

    batch_mask = np.random.choice(num_train, BATCH_SIZE)
    X_batch = data['X_train'][batch_mask]
    y_batch = data['y_train'][batch_mask]
    start_time = time.time()
    feed_dict = { X_image : X_batch, y_label : y_batch}
    _, loss_value, accuracy, acc_str, xentropy_str, reg_loss_str = sess.run([train_op, total_loss, accuracy_op, acc_summary, cross_entropy_loss, reg_loss_summary], feed_dict=feed_dict)
    summary_writer.add_summary(acc_str, step)
    summary_writer.add_summary(xentropy_str, step)
    summary_writer.add_summary(reg_loss_str, step)
    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
    if step % 100 == 0:
      summary_str = sess.run(summary_op, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, step)
    if step % 10 == 0:
      num_valid = data['X_val'].shape[0]
      batch_valid_mask = np.random.choice(num_valid, BATCH_SIZE)
      X_val_batch = data['X_val'][batch_valid_mask]
      y_val_batch = data['y_val'][batch_valid_mask]
      valid_dict = { X_image : X_val_batch, y_label : y_val_batch}
      format_str = ('{0}: step {1:>5d}, loss = {2:2.3f}, accuracy = {3:>3.2f}, accuracy (validation) = {4:>3.2f}')
      print( format_str.format(datetime.now(), step, loss_value, accuracy*100, 100*accuracy_op.eval(feed_dict=valid_dict, session=sess)))
    if (step % 1000 == 0 and step > 0) or (step + 1) == MAX_STEPS:
      checkpoint_path = os.path.join(train_dir, 'model.ckpt')
      saver.save(sess, checkpoint_path, global_step=step)
  # rng_state = np.random.get_state()
  # X_train = np.random.permutation(X_train)
  # np.random.set_state(rng_state)
  # y_train = np.random.permutation(y_train)


if __name__ == '__main__':
    main()
