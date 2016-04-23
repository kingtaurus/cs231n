import tensorflow as tf
import numpy as np
import math

def autoencoder(n_filters=[1,10,10,10], filter_size=[3,3,3,3]):
    input_shape = [None, 784]
    x = tf.placeholder(tf.float32, shape=input_shape)
    x_tensor = tf.reshape(x, shape=[-1, 28, 28, 1])

    current_input = x_tensor

    encoder = []
    shapes  = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_size[layer_i],
                filter_size[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.relu(
            tf.nn.conv2d(
                current_input, W, strides=[1, 2, 2, 1], padding='SAME') + b)
        #stride 1,2,2,1 collapses the dimensions
        # by 1/2 (roughly)
        #print(output.get_shape().as_list())
        current_input = output

    # %%
    # store the latent representation
    z = current_input
    #print(z.get_shape().as_list())
    encoder.reverse()
    shapes.reverse()

    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        #print("W.shape = ", W.get_shape().as_list())
        #print("output.shape = ",[tf.shape(x)[0], shape[1], shape[2], shape[3]])
        W_1 = tf.Variable(
            tf.random_uniform([
                W.get_shape().as_list()[0],
                W.get_shape().as_list()[1],
                W.get_shape().as_list()[2], W.get_shape().as_list()[3]],
                -1.0 / math.sqrt( W.get_shape().as_list()[2]),
                1.0 / math.sqrt( W.get_shape().as_list()[2])))
        output = tf.nn.relu( tf.nn.conv2d_transpose(
                current_input, W_1,
                tf.pack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME') +  b)
        current_input = output
        #print(current_input.get_shape().as_list())

    y = current_input
    cost = tf.reduce_mean(tf.square(y-x_tensor), name='l2_loss')
    return {'x': x, 'z': z, 'y': y, 'cost': cost}

def test_mnist():
    """Test the convolutional autoencder using MNIST."""
    # %%
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    ae = autoencoder()

    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    batch_size = 100
    n_epochs = 20
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: batch_xs}))

    n_examples = 10
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    print(recon.shape)
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (784,)) + mean_img,
                (28, 28)))

    fig.show()
    plt.draw()
    plt.savefig("simple_autoencoder.png")

if __name__ == "__main__":
    test_mnist()
