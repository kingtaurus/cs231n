
import sys
import os
import time

import numpy as np
import tensorflow as tf

from vgg_model import load_graph

import img_utils

# parameters to manage experiments
STYLE = 'guernica'
CONTENT = 'deadpool'
STYLE_IMAGE = 'styles/' + STYLE + '.jpg'
CONTENT_IMAGE = 'content/' + CONTENT + '.jpg'
IMAGE_HEIGHT = 600
IMAGE_WIDTH = 800
NOISE_RATIO = 0.6 # percentage of weight of the noise for intermixing with the content image


# Layers used for style features. You can change this.
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
#W_LAYER      = [0.5, 1.0, 1.5, 3.0, 4.0] # give more weights to deeper layers.
W_LAYER      = [2.**-1, 2.**0, 2.**1, 2.**2, 2.**3] # give more weights to deeper layers.

# Layer used for content features. You can change this.
CONTENT_LAYER = 'conv4_2'

ITERS = 1600

LR = 0.6

batch_shape = (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)

def create_content_loss(features, graph):
    content_loss = tf.constant(0, dtype=tf.float32)
    for key in features.keys():
        content_loss += tf.reduce_sum(tf.square(graph[key] - tf.constant(features[key])))
    return content_loss


def tf_gram_matrix(feature):
    feature_shape = feature.get_shape().as_list()
    M = feature_shape[-1]
    N = np.prod(feature_shape[:-1])
    F = tf.reshape(feature, shape=[-1, M], name="F")
    return tf.matmul(tf.transpose(F), F) / tf.reduce_prod(tf.cast(F.get_shape(), dtype=tf.float32))

def create_style_loss(features, graph):
    style_loss = tf.constant(0, dtype=tf.float32)
    for idx, key in enumerate(STYLE_LAYERS):
        style_loss += tf.reduce_sum(W_LAYER[idx] * tf.square((tf_gram_matrix(graph[key]) - tf.constant(features[key], dtype=tf.float32))))
    return style_loss

def create_content_features(content_image):
    content_features = {}
    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = img_utils.preprocess(X_content)
        content_np = np.array([content_image])
        content_net, _ = load_graph(data_path='vgg_weights/imagenet-vgg-verydeep-19.mat', input_image=X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER].eval(feed_dict={X_content : content_np})
    return content_features

def create_style_features(style_image):
    style_features = {}
    with tf.Graph().as_default(), tf.Session() as sess:
        X_style = tf.placeholder(tf.float32, shape=batch_shape, name='style_image')
        X_pre = img_utils.preprocess(X_style)
        net, _ = load_graph(data_path='vgg_weights/imagenet-vgg-verydeep-19.mat', input_image=X_pre)
        style_np = np.array([style_image])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={X_style: style_np})
            features = features.reshape((-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram
    return style_features

if __name__ == '__main__':
    #files to be used:
    stylized_file = 'samples/starry_night.jpg'
    content_file  = 'samples/yosemite_upperfalls_river.jpg'

    #need the basename (to split from extension)
    content_file_basename  = os.path.basename(content_file)
    stylized_file_basename = os.path.basename(stylized_file)

    #used to save the output:
    content_file_noext  = os.path.splitext(content_file_basename)[0]
    stylized_file_noext = os.path.splitext(stylized_file_basename)[0]

    output_file = "stylized/" + content_file_noext + "-style-" + stylized_file_noext + '_iter_{:d}.png'

    # (os.path.basename('samples/starry_night.jpg'))
    # os.path.splitext("path_to_file")[0]
    # sys.exit(0)
    style_image_file = img_utils.get_image_of_size(stylized_file, IMAGE_HEIGHT, IMAGE_WIDTH)
    style_features = create_style_features(style_image_file)
    #print(style_features.keys())

    content_image_file = img_utils.get_image_of_size(content_file, IMAGE_HEIGHT, IMAGE_WIDTH)
    content_features = create_content_features(content_image_file)
    #print(content_features.keys())

    with tf.variable_scope('input') as scope:
        input_image = tf.get_variable(name="image", trainable=True, dtype=tf.float32, shape=batch_shape, initializer=tf.constant_initializer(value=0., dtype=tf.float32))
    net, _ = load_graph(data_path='vgg_weights/imagenet-vgg-verydeep-19.mat', input_image=input_image)
    l_content = create_content_loss(content_features, net)
    l_style   = create_style_loss(style_features, net)

    total_loss = l_content + 512 * l_style

    import matplotlib.pyplot as plt

    content_image_file = img_utils.preprocess(content_image_file)

    initial_image = img_utils.generate_noise_image(content_image_file, IMAGE_HEIGHT, IMAGE_WIDTH, 0.1)
    #initial_image = img_utils.postprocess(initial_image)#undo the preprocessing;

    assign_op = input_image.assign(initial_image)
    opt = tf.train.AdamOptimizer(learning_rate=LR).minimize(total_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        an_image = sess.run(assign_op)
        plt.figure()
        plt.subplot(121)
        plt.imshow(img_utils.postprocess(np.squeeze(an_image, 0)).astype(np.uint8))
        plt.subplot(122)
        plt.imshow(img_utils.postprocess(np.squeeze(content_image_file)).astype(np.uint8))
        plt.show()
        plt.ioff()
        #log loss
        print(np.log(sess.run(total_loss)))
        for index in range(ITERS+1):
            print(index)
            start = time.time()
            sess.run(opt)
            print("optimization took : ", time.time() - start)
            print(np.log(sess.run(total_loss)))
            out_image = sess.run(input_image)
            if index > 0 and index % 100 == 0:
                plt.figure(figsize=(12,10), dpi=300)
                display = img_utils.postprocess(np.squeeze(out_image, 0))
                plt.imshow(np.clip(img_utils.postprocess(np.squeeze(out_image, 0)), 0., 255).astype(np.uint8), interpolation="none")
                plt.axis('off')
                plt.savefig(output_file.format(index), bbox_inches='tight', dpi=100)
                plt.show()



    print("Done")