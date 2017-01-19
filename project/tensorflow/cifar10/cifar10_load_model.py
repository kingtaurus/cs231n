import io
import os
import re
import sys
import tarfile

from datetime import datetime, date
import time

import calendar

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage

import tensorflow as tf
from data_utils import get_CIFAR10_data

from cifar10_tensorflow import inference

import argparse
from collections import namedtuple

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

#CUDA_VISIBLE_DEVICES=0,1
#CUDA_VISIBLE_DEVICES=0
#CUDA_VISIBLE_DEVICES=1
#CUDA_VISIBLE_DEVICES=""

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

with tf.device('/cpu:0'):
  sess = tf.Session(config=config)
  data = get_CIFAR10_data(num_training=49000,num_validation=1000, num_test=5000)
  print(len(data['y_test']))

  X_image = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
  y_label = tf.placeholder(dtype=tf.int64, shape=[None])

  logits, grad_image, grad_image_placeholder   = inference(X_image)
  top_k_op = tf.nn.in_top_k(predictions=logits, targets=y_label, k=1)

  ckpt = tf.train.get_checkpoint_state('./cifar10_results/LR_0.03/REG_0.11/KP_0.9/January_14_2017-h17m55/')
  saver = tf.train.Saver()
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    array = sess.run(top_k_op, feed_dict={X_image: data['X_test'][0:500],
                                        y_label : data['y_test'][0:500]})
    print(np.mean(array))
  else:
    print("No Checkpoint Found")
  # restore = tf.train.Saver(tf.all_variables())
  # restore.restore(sess, './cifar10_results/LR_0.03/REG_0.11/KP_0.9/January_14_2017-h17m55/checkpoint' )
  # print(sess.run(tf.all_Variables()))



