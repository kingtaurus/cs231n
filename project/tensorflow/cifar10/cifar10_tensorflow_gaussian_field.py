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

from scipy import ndimage

import tensorflow as tf
from data_utils import get_CIFAR10_data

import argparse
from collections import namedtuple

from tqdm import tqdm, trange

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
  test_image = np.array(ndimage.imread(buf))
  plt.close()
  return test_image[np.newaxis,:]

IMAGE_SIZE = 32
NUM_CLASSES = 10
BATCH_SIZE = 512

LOSSES_COLLECTION  = 'regularizer_losses'
DEFAULT_REG_WEIGHT =  1e-1

def activation_summaries(activation, name):
  #might want to specify the activation type (since min will always be 0 for ReLU)
  with tf.name_scope("activation_summaries"):
    mean = tf.reduce_mean(activation)
    tf.summary.histogram(name + '/activations', activation)
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(activation))
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(activation - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(activation))
    tf.summary.scalar('min/' + name, tf.reduce_min(activation))

def variable_summaries(variable, name):
  with tf.name_scope("variable_summaries"):
    mean = tf.reduce_mean(variable)
    tf.summary.histogram(name + '/variable_hist', variable)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(variable - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(variable))
    tf.summary.scalar('min/' + name, tf.reduce_min(variable))

#reg_placeholder = tf.placeholder(dtype=tf.float32, shape=[1])
## may want to add this to the inputs for rcl (and inference methods)
# with tf.op_scope([tensor], scope, 'L2Loss'):
#     weight = tf.convert_to_tensor(weight,
#                               dtype=tensor.dtype.base_dtype,
#                               name='loss_weight')
#     loss = tf.mul(weight, tf.nn.l2_loss(tensor), name='value')
#     tf.add_to_collection(LOSSES_COLLECTION, loss)
#     return loss


# IDEA: construct validation network (reuses parameters)
#       construct train network
#       construct visualization tool
#       construct weight reduction test

#probably need to change for validation
# so: it should be train_layer
# validation_layer = tf.get_variable("W", resuse=True)


def weight_decay(layer_weights, wd=0.99):
  layer_weights = tf.mul(wd, layer_weights)
  return layer_weights

unit_value =  np.zeros(shape=(5,5))
unit_value[2,2] = 1
a_s = ndimage.filters.gaussian_filter(unit_value, sigma=1.0, order=0)
max_a = np.max(a_s)
a_s = 0.95 * a_s / max_a
a_s = a_s[:,:, np.newaxis, np.newaxis]
gaussian_field = tf.constant(a_s, dtype=tf.float32)
field_modified = tf.random_uniform(shape=kernel_shape, minval=0, maxval=1.0, dtype=tf.float32, name='random')
sub = tf.greater(field_modified, gaussian_field)
# a_s = np.array([[0,0,1,0,0],
#                 [0,1,1,1,0],
#                 [1,1,1,1,1],
#                 [0,1,1,1,0],
#                 [0,0,1,0,0]])
print(name,"\n", a_s)
# a_s = a_s[:,:, np.newaxis, np.newaxis]
# sub = tf.constant(a_s, dtype=tf.float32)
sub = tf.cast(sub, dtype=tf.float32)

def conv_relu(layer_in, kernel_shape, bias_shape, name, is_training=True):
  with tf.variable_scope(name) as scope:
    kernel = tf.get_variable("W",
                             shape=kernel_shape,
                             initializer=tf.contrib.layers.xavier_initializer_conv2d())
    bias = tf.get_variable("b", shape=bias_shape, initializer=tf.constant_initializer(0.))

    weight_decayed_op = weight_decay(kernel)

    if name != "conv_1":
      # a_s = np.random.binomial(1, p=0.8, size=kernel_shape[0:3])
      # #np.random.randint(low=0, high=2, size=kernel_shape[0:3])
      # print(name,"\n",a_s[:,:,0])
      # a_s = a_s[:,:,:,np.newaxis]

      # sub = tf.constant(a_s, dtype=tf.float32)
      # kernel = kernel * sub
      #Above is similar to drop connect(?)
      print("Not altering the 'kernels' for later layers")
    else:
      kernel = kernel * sub

    conv = tf.nn.conv2d(layer_in, kernel, strides=[1,1,1,1], padding='SAME')
    layer = tf.nn.relu(conv + bias)
    #, is_training=False
    layer = tf.contrib.layers.batch_norm(inputs=layer, decay=0.999, center=True, scale=True, data_format="NHWC", is_training=is_training, reuse=False, scope=scope, updates_collections=None)
    scope.reuse_variables()
    bn_mean = tf.get_variable("beta")
    bn_std = tf.get_variable("gamma")
    variable_summaries(bn_mean, name + "_bn_mean")
    variable_summaries(bn_std, name + "_bn_std")

    #variable_summaries(bias, bias.name)
    variable_summaries(kernel, name + "_kernel")
    activation_summaries(layer, name + "_activation")
    # layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')
  return layer, weight_decayed_op

def fcl_relu(layer_in, output_size, name,
             regularizer_weight=None, keep_prob=None,
             loss_collection=LOSSES_COLLECTION):
  with tf.variable_scope(name) as scope:
    #batch_size = layer_in.get_shape().as_list()[0]
    dim = np.prod(layer_in.get_shape().as_list()[1:])
    reshape = tf.reshape(layer_in, [-1, dim])
    weights = tf.get_variable("W_fcl",
                              shape=[dim, output_size],
                              initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable("b_fcl",
                           shape=[output_size],
                           initializer=tf.constant_initializer(0.))
    if keep_prob is None:
      keep_prob = 1.
    layer = tf.nn.relu(tf.matmul(reshape, weights) + bias, name=scope.name)
    layer = tf.nn.dropout(layer, keep_prob)
    variable_summaries(weights, weights.name)
    #variable_summaries(bias, bias.name)
    activation_summaries(layer, layer.name)
    if regularizer_weight is None:
      regularizer_weight = DEFAULT_REG_WEIGHT
    regularizer_loss = tf.mul(regularizer_weight, tf.nn.l2_loss(weights))
    tf.add_to_collection(loss_collection, regularizer_loss)
  return layer


def inference(images,
              classes = NUM_CLASSES,
              keep_prob=None,
              regularizer_weight=None,
              loss_collection=LOSSES_COLLECTION,
              is_training=True,
              model_name="model_1"):
  with tf.variable_scope(model_name) as model_scope:
    layer = conv_relu(images, [5,5,3,128], [128], "conv_1", is_training=is_training)
    layer = conv_relu(layer,  [3,3,128,128], [128], "conv_2", is_training=is_training)
    layer = conv_relu(layer,  [3,3,128,128], [128], "conv_3", is_training=is_training)
    layer = conv_relu(layer,  [3,3,128,128], [128], "conv_4", is_training=is_training)
    layer = conv_relu(layer,  [3,3,128,128], [128], "conv_5", is_training=is_training)
    layer = conv_relu(layer,  [3,3,128,128], [128], "conv_6", is_training=is_training)
    layer = conv_relu(layer,  [3,3,128,128], [128], "conv_7", is_training=is_training)
    # layer = conv_relu(layer,  [5,5,64,64], [64], "conv_4")
    # layer = conv_relu(layer,  [3,3,128,128], [128], "conv_5")
    # layer = conv_relu(layer,  [3,3,128,128], [128], "conv_6")
    last_conv_layer = layer
    print(last_conv_layer.get_shape())
    layer = fcl_relu(layer, 128, "fcl_1", keep_prob=keep_prob)

    with tf.variable_scope('pre_softmax_linear') as scope:
      weights = tf.get_variable('weights',
                                shape=[128, classes],
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
      if regularizer_weight is None:
        regularizer_weight = DEFAULT_REG_WEIGHT
      regularizer_loss = tf.mul(regularizer_weight, tf.nn.l2_loss(weights))
      tf.add_to_collection(loss_collection, regularizer_loss)
  grad_image_placeholder = tf.placeholder(dtype=tf.float32, shape=last_conv_layer.get_shape())
  grad_image = tf.gradients(last_conv_layer, [images], grad_image_placeholder)
  print(grad_image[0].get_shape())
  return pre_softmax_linear, (wd_0, wd_1, wd_2), grad_image[0], grad_image_placeholder, last_conv_layer

def predict(logits):
  return tf.argmax(logits, dimension=1)

def accuracy(logits, y_label):
  return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1), y_label), tf.float32))


def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

  # The total loss is defined as the cross entropy loss
  return cross_entropy_mean

INITIAL_LEARNING_RATE = 0.03
LEARNING_RATE_DECAY_FACTOR = 0.80
DROPOUT_KEEPPROB = 0.9
NUM_EPOCHS_PER_DECAY = 20
MAX_STEPS = 100000

DECAY_STEPS = NUM_EPOCHS_PER_DECAY * 95
#150 is roughly the number of batches per epoch
#40,000/256 ~ 150

parser = argparse.ArgumentParser(description='CIFAR-10 Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--learning_rate', default=INITIAL_LEARNING_RATE,
  type=float, nargs='?', help='Initial Learning Rate;')
parser.add_argument('--decay_rate', default=LEARNING_RATE_DECAY_FACTOR,
  type=float, nargs='?', help='Learning Rate Decay Factor;')#alternative %(default) in help string
parser.add_argument('--keep_prob', default=DROPOUT_KEEPPROB, type=float, nargs='?',
  help='Probablity to keep a neuron in the Full Connected Layers;')
parser.add_argument('--max_steps', type=int, default=MAX_STEPS, nargs='?',
  help='Maximum number of batches to run;')
parser.add_argument('--lr_decay_time', type=int, default=NUM_EPOCHS_PER_DECAY, nargs='?',
  help='Number of Epochs till LR decays;')
parser.add_argument('--regularization_weight', type=float, default=DEFAULT_REG_WEIGHT,
  nargs='?', help='Regularization weight (l2 regularization);')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
  nargs='?', help='Batch size;')
#parser.add_argument('--lr_momentum', type=float, default=0.95, nargs='?', help='SGD Momentum Parameter;')

mutex_group = parser.add_mutually_exclusive_group(required=False)
mutex_group.add_argument('--sgd',  action='store_true')
mutex_group.add_argument('--mom',  default=True, action='store_true')
mutex_group.add_argument('--adam', action='store_true')
mutex_group.add_argument('--rms',  action='store_true')

sgd_parser = argparse.ArgumentParser(prog='cifar10_tensorflow.py', usage='%(prog)s --sgd [options]', description='SGD optimizer parsing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

mom_parser = argparse.ArgumentParser(prog='cifar10_tensorflow.py', usage='%(prog)s --mom [options]', description='SGD momentum optimizer parsing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
mom_parser.add_argument('--momentum', type=float, default=0.95, nargs='?', help='SGD Momentum Parameter;')

rms_parser = argparse.ArgumentParser(prog='cifar10_tensorflow.py', usage='%(prog)s --rms [options]', description='SGD RMSProp optimizer parsing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
rms_parser.add_argument('--decay', type=float, default=0.9, nargs='?', help="Exponential decay rate for the history gradient;")
rms_parser.add_argument('--momentum', type=float, default=0.0, nargs='?', help="SGD Momentum Parameter;")

adam_parser = argparse.ArgumentParser(prog='cifar10_tensorflow.py', usage='%(prog)s --adam [options]', description='SGD ADAM optimizer parsing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
adam_parser.add_argument('--beta1', type=float, default=0.9, nargs='?', help="Exponential decay rate for the 1st moment estimates;")
adam_parser.add_argument('--beta2', type=float, default=0.999, nargs='?', help="Exponential decay rate for the 2nd moment estimates;")
#adam_parser.add_argument('--epsilon', type=float, default=1e-8, nargs='?', help="Numerical Stability;")



##
# Add device placement (0,1)?
# Add Seed?
##

##
# Add classwise scalars
##

GradOpt    = tf.train.GradientDescentOptimizer
AdagradOpt = tf.train.AdagradDAOptimizer
MomOpt     = tf.train.MomentumOptimizer
AdamOpt    = tf.train.AdamOptimizer
RMSOpt     = tf.train.RMSPropOptimizer

opt_to_name = { GradOpt : "grad", AdagradOpt : "Adagrad",
                MomOpt  : "momentum", AdamOpt : "ADAM",
                RMSOpt  : "RMSProp"
               }

#probably should pass in the optimizer to be used:
# tf.train.GradientDescentOptimizer
# tf.train.AdagradDAOptimizer
# tf.train.MomentumOptimizer
# tf.train.AdamOptimizer
# tf.train.FtrlOptimizer
# tf.train.ProximalGradientDescentOptimizer
# tf.train.ProximalAdagradOptimizer
# tf.train.RMSPropOptimizer

## BATCH NORM
# tf.contrib.layers.batch_norm

#probably should pass in an optimizer
def train(total_loss, global_step,
          learning_rate=INITIAL_LEARNING_RATE,
          decay_steps=DECAY_STEPS,
          lr_rate_decay_factor=LEARNING_RATE_DECAY_FACTOR):
  lr = tf.train.exponential_decay(learning_rate,
                                  global_step,
                                  decay_steps,#number of steps required for it to decay
                                  lr_rate_decay_factor,
                                  staircase=True)

  tf.summary.scalar('learning_rate', lr)

  #compute gradient step
  with tf.control_dependencies([total_loss]):
    opt = MomOpt(learning_rate=lr, momentum=0.95)
    grads = opt.compute_gradients(total_loss)

  #if we wanted to clip the gradients
  #would apply the operation here

  #apply the gradients
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  for grad, var in grads:
    if grad is not None:
      print("Found gradients for: ", var.op.name)
      tf.summary.histogram(var.op.name + "/gradients", grad)

  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name="train")

  #opt = tf.train.GradientDescentOptimizer(lr).minimize(total_loss, global_step=global_step)
  # grads = opt.compute_gradients(total_loss)

  return train_op

#REFACTOR IDEA:
# (*) get_args() [ should be in main ? ];
# (0) load_data [ ... ];
# (1) build [constructs the graph], including placeholders and variables
# (2) train [generates training op]
# (3) generate parameters for two runs (one on each GPU)
# (4) runs [feeds and runs ops]

def get_optimizer(args, remaining):
  lr = args.learning_rate
  
  optimizer = MomOpt(learning_rate=lr, momentum=0.95)
  opt_string = ("SGDmomentum_%f" % 0.95)
  sub_remains = []
  args_dict = vars(args)

  # if len(remaining) == 0:
  #   args_dict['momentum']  = 0.95
  #   #return early
  #   return optimizer, opt_string

  if args.sgd:
    args_sgd, sub_remains = sgd_parser.parse_known_args(remaining)
    if len(sub_remains) > 0:
      parser.print_help()
      sgd_parser.print_help()
    optimizer = GradOpt(learning_rate=lr)
    opt_string = "SGD"
  if args.mom:
    args_sgd, sub_remains = mom_parser.parse_known_args(remaining)
    if len(sub_remains) > 0:
      parser.print_help()
      mom_parser.print_help()
    optimizer = MomOpt(learning_rate=lr, momentum=args_sgd.momentum)
    opt_string = ("SGDmomentum_%.3f" % args_sgd.momentum)
  if args.rms:
    args_sgd, sub_remains = rms_parser.parse_known_args(remaining)
    if len(sub_remains) > 0:
      parser.print_help()
      rms_parser.print_help()
    optimizer = RMSOpt(learning_rate=lr, decay=args_sgd.decay)
    opt_string = ("RMSProp_decay_%.3f_momentum_%.3f" % (args_sgd.decay, args_sgd.momentum))
  if args.adam:
    args_sgd, sub_remains = adam_parser.parse_known_args(remaining)
    if len(sub_remains) > 0:
      parser.print_help()
      adam_parser.print_help()
    optimizer = AdamOpt(learning_rate=lr, beta1=args_sgd.beta1, beta2=args_sgd.beta2)
    opt_string = ("ADAM_beta1_%.3f_beta2_%.3f" % (args_sgd.beta1, args_sgd.beta2))

  if len(sub_remains) > 0:
    [print("Failed due to extra args: ", x) for x in sub_remains]
    sys.exit(1)

  #add the arguments to the dictionary (for args)
  args_sgd_dict  = vars(args_sgd)

  for k in args_sgd_dict.keys():
    args_dict[k] = args_sgd_dict[k]

  return optimizer, opt_string

def main():
  args, remaining = parser.parse_known_args()

  lr         = args.learning_rate#INITIAL_LEARNING_RATE
  reg_weight = args.regularization_weight
  kp         = args.keep_prob
  max_steps  = args.max_steps
  decay_rate = args.decay_rate
  lr_decay_time = args.lr_decay_time
  batch_size = args.batch_size

  optimizer, opt_string = get_optimizer(args, remaining)
  print(opt_string)
  #CURRENTLY NOT Used
  print("Arguments = ", args)

  print("Loading Data;")

  data = get_CIFAR10_data()
  train_size = len(data['y_train'])
  for k, v in data.items():
      print('%s: '%(k), v.shape)

  #PLACEHOLDER VARIABLES
  keep_prob = tf.placeholder(dtype=tf.float32, shape=())
  learning_rate = tf.placeholder(dtype=tf.float32, shape=())
  regularizer_weight = tf.placeholder(dtype=tf.float32, shape=())
  is_training = tf.placeholder(dtype=tf.bool, shape=())

  X_image = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
  y_label = tf.placeholder(dtype=tf.int64, shape=[None])

  # test = tf.equal(True, is_training)
  #only do distortions on training data
  X_image = tf.cond(is_training, lambda: tf.map_fn(lambda img: tf.image.random_flip_left_right(img), X_image), lambda: X_image)
  X_image = tf.cond(is_training, lambda: tf.map_fn(lambda img: tf.image.random_flip_up_down(img), X_image), lambda: X_image)
  X_image = tf.cond(is_training, lambda: tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=60), X_image), lambda: X_image)
  X_image = tf.cond(is_training, lambda: tf.map_fn(lambda img: tf.image.random_contrast(img, lower=0.2, upper=1.8), X_image), lambda: X_image)

  # X_image = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), X_image)
  # X_image = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), X_image)
  # X_image = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=60), X_image)
  # X_image = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=0.2, upper=1.8), X_image)
  # def image_distortions(image, distortions):
  #     distort_left_right_random = distortions[0]
  #     mirror = tf.less(tf.pack([1.0, distort_left_right_random, 1.0]), 0.5)
  #     image = tf.reverse(image, mirror)
  #     distort_up_down_random = distortions[1]
  #     mirror = tf.less(tf.pack([distort_up_down_random, 1.0, 1.0]), 0.5)
  #     image = tf.reverse(image, mirror)
  #     return image
  # distortions = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)
  # image = image_distortions(image, distortions)
  # tf.image.flip_up_down(image)
  # tf.image.flip_left_right(image)
  # tf.image.transpose_image(image)
  # tf.image.rot90(image, k=1, name=None)
  # tf.image.adjust_brightness
  # tf.image.adjust_contrast(images, contrast_factor)
  # tf.image.per_image_standardization(image)

  #MODEL related operations and values
  global_step = tf.Variable(0, trainable=False)
  b_norm_images  = tf.contrib.layers.batch_norm(inputs=X_image, center=True, scale=True, decay=0.95, data_format="NHWC", is_training=is_training, scope="input", updates_collections=None)
  #MODEL construction
  logits, (wd_0, wd_1, wd_2), grad_image, grad_image_placeholder, last_layer = inference(b_norm_images, keep_prob=keep_prob, regularizer_weight=regularizer_weight, is_training=is_training)
  prediction = predict(logits)
  loss_op = loss(logits, y_label)

  reg_loss = tf.reduce_sum(tf.get_collection(LOSSES_COLLECTION))
  total_loss = loss_op + reg_loss

  accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1), y_label), tf.float32))
  #print('decay_steps = ', lr_decay_time * (train_size // batch_size + 1))
  print("Number of batch steps till lr_decay = ", lr_decay_time * ((train_size //  batch_size) + 1))
  train_op = train(total_loss, global_step, learning_rate=lr, lr_rate_decay_factor=decay_rate, decay_steps=lr_decay_time * ((train_size //  batch_size) + 1))

  saver = tf.train.Saver(tf.global_variables())

  #Summary operation
  tf.summary.image('images', X_image)
  summary_op = tf.summary.merge_all()

  acc_summary        = tf.summary.scalar('Training_accuracy_batch', accuracy_op)
  validation_acc_summary = tf.summary.scalar('Validation_accuracy', accuracy_op)
  cross_entropy_loss = tf.summary.scalar('loss_raw', loss_op)
  reg_loss_summary   = tf.summary.scalar('regularization_loss', reg_loss)
  total_loss_summary = tf.summary.scalar('total_loss', total_loss)

  accuracy_batch = tf.placeholder(shape=(None), dtype=tf.float32)
  overfit_estimate = tf.placeholder(shape=(None), dtype=tf.float32)

  accuracy_100 = tf.reduce_mean(accuracy_batch)
  mean_summary = tf.summary.scalar('Training_accuracy_mean', accuracy_100)
  validation_mean_summary = tf.summary.scalar('Validation_accuracy_mean', accuracy_100)

  acc_summary_histogram = tf.summary.histogram('Training_accuracy_histogram', accuracy_batch)
  overfit_summary = tf.summary.scalar('overfit_estimate', overfit_estimate)

  #SESSION Construction
  init = tf.global_variables_initializer()

  config = tf.ConfigProto()
  # config.gpu_options.allow_growth = True
  # config.gpu_options.per_process_gpu_memory_fraction = 0.5
  config.log_device_placement=False

  sess = tf.Session(config=config)
  sess.run(init)
  # input_grad_image = np.zeros((1,32,32,16), dtype=np.float)
  # input_grad_image[0,15,15,:] = 1000.
  # back_image = sess.run(grad_image[0], feed_dict={X_image : 128 * np.ones((1,32,32,3)), regularizer_weight : 0., keep_prob : 1.0, grad_image_placeholder : input_grad_image})
  # print(back_image, np.max(back_image))
  # plt.figure()
  # max_value = np.max(back_image)
  # min_value = np.min(back_image)
  # print(back_image.shape)
  # plt.imshow(back_image[:,:,0], cmap=plt.get_cmap("seismic"), vmin=-1,
  #        vmax=1, interpolation="nearest")
  # plt.show()
  # sys.exit(0)

  # print("sub.shape = ", sub.get_shape())
  # print("sub = ", sess.run(sub))
  # print("sub = ", sess.run(sub))
  #today = date.today()
  current_time = datetime.now()
  # LR_%f, INITIAL_LEARNING_RATE
  # REG_%f, DEFAULT_REG_WEIGHT
  # add details, relating per epoch results (and mean filtered loss etc.)
  train_dir = "cifar10_results/gaussian_field/LR_" + str(lr) + "/" + "REG_" + str(reg_weight) + "/" + "KP_" + str(kp) + "/" + current_time.strftime("%B") + "_" + str(current_time.day) + "_" + str(current_time.year) + "-h" + str(current_time.hour) + "m" + str(current_time.minute)
  print("Writing summary data to :  ", train_dir)
  #probably should write parameters used to train the model to this directory
  #also pickle the named tuple
  # with open('train_dir' + '/model_parameters.txt', 'w') as outfile:
  #   #
  #should write the checkpoint files


  acc_list = []
  valid_acc_list = []

  cm_placeholder = tf.placeholder(shape=(1, None, None, 4), dtype=tf.uint8)
  confusion_summary = tf.summary.image('confusion_matrix', cm_placeholder)
  layer_output_placeholder = tf.placeholder(shape=(3,None,None,1), dtype=tf.uint8)
  layer_summary = tf.summary.image('layer_summary', layer_output_placeholder)
  print(last_layer.get_shape())
  summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

  print("Starting Training.")
  print("Training for %d batches (of size %d); initial learning rate %f" % (max_steps, batch_size, lr))
  # tqdm_format_str = ('{0}: step {1:>5d}, loss = {2:2.3f}, accuracy = {3:>3.2f}, accuracy (val) = {4:>3.2f}')
  # current_step = 0
  # tqdm_loss = np.inf
  # tqdm_acc  = 0.
  # tqdm_val  = 0.
  # t = tqdm(range(max_steps), desc="Epoch %d, step %d, loss %2.2f, acc %2.2f, acc (val) %2.2f"%(epoch, current_step, tqdm_loss, tqdm_acc, tqdm_val), leave=True)
  #t = trange(max_steps, desc="Epoch %d, step %d, loss %2.2f, acc %2.2f, acc (val) %2.2f"%(epoch, current_step, tqdm_loss, tqdm_acc, tqdm_val), leave=True)
  for step in range(max_steps):
    # current_step = step
    # t.set_description(desc="Epoch %d, step %d, loss %2.2f, acc %2.2f, acc (val) %2.2f"%(epoch, current_step, tqdm_loss, tqdm_acc, tqdm_val))
    # t.refresh()
    num_train = data['X_train'].shape[0]
    if batch_size * (step - 1) // num_train < batch_size * (step) // num_train and step > 0:
      print("Completed Epoch: %d (step=%d, max_steps=%d, percentage complete= %f)" % ((batch_size * (step) // num_train ), step, max_steps, step/max_steps * 100))
      epoch = (batch_size * (step) // num_train )
      sess.run([wd_0])

    batch_mask = np.random.choice(num_train, batch_size)
    X_batch = data['X_train'][batch_mask]
    y_batch = data['y_train'][batch_mask]
    start_time = time.time()
    feed_dict = { X_image : X_batch, y_label : y_batch, keep_prob : kp, regularizer_weight : reg_weight, is_training : True }

    loss_value, accuracy, acc_str, xentropy_str, reg_loss_str, predicted_class = sess.run([total_loss, accuracy_op, acc_summary, cross_entropy_loss, reg_loss_summary, prediction], feed_dict=feed_dict)
    #print(sess.run(prediction, feed_dict=feed_dict))
    # tqdm_loss = loss_value
    # tqdm_acc = accuracy
    sess.run(train_op, feed_dict=feed_dict)

    # if step > 0 and step % 50 == 0:
    #   last_layer_out, logits_out = sess.run([last_layer, logits], feed_dict=feed_dict)
    #   #print(logits[0])
    #   logits_out = np.exp(logits_out) / np.sum(np.exp(logits_out), axis=0)

    #   #print(layer_out.shape, layer_out.mean())
    #   sliced_layer = last_layer_out[0:1,:,:,0:9]
    #   sliced_layer = np.transpose(sliced_layer, (3,1,2,0))
    #   #print(sliced_layer.shape)
    #   split_layer = np.vsplit(sliced_layer, sliced_layer.shape[0])
    #   squeezed_ = [np.squeeze(x, axis=(0,3)) for x in split_layer]
    #   vstacked = np.vstack(squeezed_)
    #   #print(vstacked.shape)
    #   plt.figure()
    #   plt.subplot(211)
    #   plt.imshow(vstacked, vmin=0, vmax=np.max(vstacked), cmap=plt.cm.Blues)
    #   plt.grid(b='off')
    #   plt.subplot(212)
    #   bar_width = 0.1
    #   print(logits_out.shape)
    #   index = np.arange(len(logits_out[0]))
    #   colors = ['blue' for x in logits_out[0]]
    #   colors[np.argmax(logits_out[0])] = 'green'
    #   sns.barplot(classes, logits_out[0], palette=colors)
    #   plt.grid(b='off')
    #   plt.show()


    acc_list.append(accuracy)
    acc_list = acc_list[-100:]
    accuracy_100_str = sess.run(mean_summary, feed_dict={accuracy_batch : np.array(acc_list)})
    #print(sess.run([accuracy_100], feed_dict={accuracy_batch : np.array(acc_list[-100:])}))
    summary_writer.add_summary(acc_str, step)
    summary_writer.add_summary(xentropy_str, step)
    summary_writer.add_summary(reg_loss_str, step)
    summary_writer.add_summary(accuracy_100_str, step)
    #image = sess.run(grad_image, feed_dict=feed_dict)
    #summary_writer.add_summary('Training_accuracy (Mean)', np.mean(acc_list[-100:]), step)
    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
    if step % 100 == 0:
      summary_str = sess.run(summary_op, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, step)
      # plt.figure()
      # plt.imshow(image[0])
      # plt.grid(b=False)
      # plt.show()
    if step % 10 == 0:
      #print("max = %f; mean = %f" %(np.max(image), np.mean(image)))
      if step > 0:
        #print('creating image;')
        confusion_img = plot_confusion_matrix(confusion_matrix(y_batch, predicted_class),
                                              title='Confusion matrix',
                                              cmap=plt.cm.Blues,
                                              labels=classes)
        # print(img.get_shape())
        # print(img.dtype)

        summary_writer.add_summary(confusion_summary.eval(session=sess, feed_dict={cm_placeholder: confusion_img}), step)
        del confusion_img

        acc_summary_histogram_out = sess.run(acc_summary_histogram, feed_dict={accuracy_batch : np.array(acc_list[-100:])})
        summary_writer.add_summary(acc_summary_histogram_out, step)
        #print('done adding summary')
      num_valid = data['X_val'].shape[0]
      batch_valid_mask = np.random.choice(num_valid, batch_size)
      X_val_batch = data['X_val'][batch_valid_mask]
      y_val_batch = data['y_val'][batch_valid_mask]
      valid_dict = { X_image : X_val_batch, y_label : y_val_batch, keep_prob : 1.0, regularizer_weight : 0.00, is_training : False}
      format_str = ('{0}: step {1:>5d}, loss = {2:2.3f}, accuracy = {3:>3.2f}, accuracy (val) = {4:>3.2f}, loss = {5:2.3f}')
      valid_summary, valid_acc, valid_loss = sess.run([validation_acc_summary, accuracy_op, loss_op], feed_dict=valid_dict)
      valid_acc_list.append(valid_acc)
      #tqdm_val = valid_acc

      valid_acc_list = valid_acc_list[-100:]
      # Probably should change the slice size to be smaller (10 instead of 100)
      valid_accuracy_100_str = sess.run(validation_mean_summary, feed_dict={accuracy_batch : np.array(valid_acc_list)})
      print(format_str.format(datetime.now(), step, loss_value, 100*accuracy, 100*valid_acc, valid_loss))
      overfit_summary_str = sess.run(overfit_summary, feed_dict = {overfit_estimate : accuracy - valid_acc})
      summary_writer.add_summary(overfit_summary_str, step)
      summary_writer.add_summary(valid_summary, step)
      summary_writer.add_summary(valid_accuracy_100_str, step)

    if (step % 5000 == 0 and step > 0) or (step + 1) == max_steps:
      checkpoint_path = os.path.join(train_dir, current_time.strftime("%B") + "_" + str(current_time.day) + "_" + str(current_time.year) + "-h" + str(current_time.hour) + "m" + str(current_time.minute) + 'model.ckpt')
      print("Checkpoint path = ", checkpoint_path)
      saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=True)

  return 0

  # rng_state = np.random.get_state()
  # X_train = np.random.permutation(X_train)
  # np.random.set_state(rng_state)
  # y_train = np.random.permutation(y_train)

if __name__ == '__main__':
  main()
