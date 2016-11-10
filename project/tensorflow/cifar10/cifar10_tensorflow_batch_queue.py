import io

import gzip
import os
import re
import sys
import tarfile

from datetime import datetime, date
import time
import urllib.request

import calendar

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.models.image.cifar10 import cifar10_input

import seaborn as sns
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from sklearn.metrics import confusion_matrix
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# plt.get_cmap('gray')
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=None):
    if labels is None:
        labels = list(range(len(cm)))
    fig = plt.figure()
    plt_img = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(plt_img)
    plt.title(title)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.grid(b='off')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.colorbar()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

IMAGE_SIZE = 32
NUM_CLASSES = 10
BATCH_SIZE = 256

LOSSES_COLLECTION  = 'regularizer_losses'
DEFAULT_REG_WEIGHT =  1e-2

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def maybe_download_and_extract(data_dir):
  """Download and extract the tarball from Alex's website."""
  dest_directory = data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

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

def get_cifar10_filenames(data_dir):
  data_dir += "/cifar-10-batches-bin"
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in range(1, 6)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  return filenames

def get_image(filename_queue):
  #CIFAR10Record is a 'C struct' bundling tensorflow input data
  class CIFAR10Record(object):
    pass
  #
  result = CIFAR10Record()
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                           [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  return result

def generate_batch(image, label, min_queue_examples, batch_size=BATCH_SIZE, shuffle=True):
  num_preprocess_threads = 4
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def read_cifar10(data_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
  filenames = get_cifar10_filenames(data_dir)
  filename_queue = tf.train.string_input_producer(filenames)
  read_input = get_image(filename_queue)

  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  distorted_image = tf.image.random_flip_left_right(reshaped_image)
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  float_image = tf.image.per_image_whitening(distorted_image)

  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print('Filling queue with %d CIFAR images before starting to train. '
        'This will take a few minutes.' % min_queue_examples)
  return generate_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)


#initializer=tf.contrib.layers.xavier_initializer()
#tf.contrib.layers.xavier_initializer_conv2d
def model(images):
  pass


#conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]), name="conv1_weights")
#alternate way of doing it

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

def fcl_relu(layer_in, output_size, name,
             regularizer_weight=None, keep_prob = None,
             loss_collection=LOSSES_COLLECTION):
  with tf.variable_scope(name) as scope:
    batch_size = layer_in.get_shape().as_list()[0]
    reshape = tf.reshape(layer_in, [batch_size, -1])
    dim = reshape.get_shape()[1].value
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
  return layer

def inference(images,
              classes = NUM_CLASSES,
              keep_prob=None,
              regularizer_weight=None,
              loss_collection=LOSSES_COLLECTION,
              model_name="model_1"):
  with tf.variable_scope(model_name) as model_scope:
    layer = conv_relu(images, [5,5,3,64], [64], "conv_1")
    layer = conv_relu(layer,  [5,5,64,64], [64], "conv_2")
    layer = conv_relu(layer,  [5,5,64,128], [128], "conv_3")
    # layer = conv_relu(layer,  [5,5,64,64], [64], "conv_4")
    # layer = conv_relu(layer,  [3,3,128,128], [128], "conv_5")
    # layer = conv_relu(layer,  [3,3,128,128], [128], "conv_6")
    layer = fcl_relu(layer, 64, "fcl_1", keep_prob=keep_prob)

    with tf.variable_scope('pre_softmax_linear') as scope:
      weights = tf.get_variable('weights',
                                shape=[64, classes],
                                initializer=tf.contrib.layers.xavier_initializer())
      biases = tf.get_variable('biases',
                               shape=[classes],
                               initializer=tf.constant_initializer(0.))
      pre_softmax_linear = tf.add(tf.matmul(layer, weights), biases, name=scope.name)
      if keep_prob is None:
        keep_prob = 1.
      pre_softmax_linear = tf.nn.dropout(pre_softmax_linear, keep_prob)
      variable_summaries(weights, weights.name)
      #variable_summaries(biases, biases.name)
      #activation_summaries(pre_softmax_linear, pre_softmax_linear.name)
  return pre_softmax_linear

def predict(logits):
  return tf.argmax(logits, dimension=1)

def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

  # The total loss is defined as the cross entropy loss
  return cross_entropy_mean

INITIAL_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY_FACTOR = 0.95
NUM_EPOCHS_PER_DECAY = 5
MAX_STEPS = 100000

def train(total_loss, global_step, batch_size=BATCH_SIZE):
  number_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size

  decay_steps = int(number_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)

  tf.scalar_summary('learning_rate', lr)

  # with tf.control_dependencies([total_loss]):
  #   opt = tf.train.AdamOptimizer(lr)
  #   grads = opt.compute_gradients(total_loss)

  # #apply the gradients
  # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # for grad, var in grads:
  #   if grad is not None:
  #     tf.histogram_summary(var.op.name + "/gradients", grad)

  # with tf.control_dependencies([apply_gradient_op]):
  #   train_op = tf.no_op(name="train")

  opt = tf.train.GradientDescentOptimizer(lr).minimize(total_loss, global_step=global_step)
  # grads = opt.compute_gradients(total_loss)

  return opt

def main():
  print("Loading Data (downloading if archive hasn't been downloaded already);")

  data_dir = "cifar10_images"
  train_dir = "cifar10_results/batch/"
  maybe_download_and_extract(data_dir=data_dir)
  if tf.gfile.Exists(train_dir):
    tf.gfile.DeleteRecursively(train_dir)
  tf.gfile.MakeDirs(train_dir)
  images, labels = read_cifar10(data_dir=data_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
  #print(images,labels)

  #PLACEHOLDER VARIABLES
  keep_prob = tf.placeholder(dtype=tf.float32, shape=())
  learning_rate = tf.placeholder(dtype=tf.float32, shape=())
  regularizer_weight = tf.placeholder(dtype=tf.float32, shape=())
  #Not used --- ^ (currently)

  #MODEL related operations and values
  global_step = tf.Variable(0, trainable=False)
  #MODEL construction
  logits = inference(images)
  loss_op = loss(logits, labels)
  accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1), tf.cast(labels, tf.int64)), tf.float32))
  train_op = train(loss_op, global_step, batch_size=BATCH_SIZE)

  reg_loss = tf.reduce_sum(tf.get_collection(LOSSES_COLLECTION))
  total_loss = loss_op + reg_loss

  saver = tf.train.Saver(tf.all_variables())

  summary_op = tf.merge_all_summaries()
  acc_summary = tf.scalar_summary('Training_accuracy (Batch)', accuracy_op)
  cross_entropy_loss = tf.scalar_summary('loss_raw', loss_op)
  reg_loss_summary   = tf.scalar_summary('regularization_loss', reg_loss)
  total_loss_summary = tf.scalar_summary('total_loss', total_loss)

  init = tf.initialize_all_variables()
  sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=False))
  sess.run(init)
  tf.train.start_queue_runners(sess=sess)

  #today = date.today()
  current_time = datetime.now()
  # LR_%f, INITIAL_LEARNING_RATE
  # REG_%f, DEFAULT_REG_WEIGHT
  # add details, relating per epoch results (and mean filtered loss etc.)
  train_dir = "cifar10_results/LR_" + str(INITIAL_LEARNING_RATE) + "/" + "REG_" + str(DEFAULT_REG_WEIGHT) + "/" + current_time.strftime("%B") + "_" + str(current_time.day) + "_" + str(current_time.year) + "-h" + str(current_time.hour) + "m" + str(current_time.minute)
  print("Writing summary data to :  ",train_dir)
  summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

  nbatches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
  for step in range(MAX_STEPS):
    start_time = time.time()
    _, loss_value, accuracy, acc_str, xentropy_str = sess.run([train_op, loss_op, accuracy_op, acc_summary, cross_entropy_loss])
    summary_writer.add_summary(acc_str, step)
    summary_writer.add_summary(xentropy_str, step)
    duration = time.time() - start_time

    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
    if step % 10 == 0:
      num_examples_per_step = BATCH_SIZE
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)

      format_str = ('%s: step %d, loss = %.2f, accuracy = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
      print(format_str % (datetime.now(), step, loss_value, (accuracy*100),
                           examples_per_sec, sec_per_batch), flush=True)
    if step != 0 and ((step - 1 ) * BATCH_SIZE) // NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN != ((step) * BATCH_SIZE) // NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN:
      print("Starting New Epoch; Epoch %d" % (((step) * BATCH_SIZE) // NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN + 1))
    if step % 100 == 0:
      summary_str = sess.run(summary_op)
      summary_writer.add_summary(summary_str, step)
    if step % 1000 == 0 or (step + 1) == MAX_STEPS:
      checkpoint_path = os.path.join(train_dir, 'model.ckpt')
      saver.save(sess, checkpoint_path, global_step=step)

  return 0


if __name__ == '__main__':
  main()
