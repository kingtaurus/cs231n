def conv2d(input, nout, filter=[3,3], stride=1, name="conv", activation=tf.nn.relu):
    with tf.variable_scope(name) as scope:
        nin = input.get_shape().as_list()[-1]
        shape = [filter[0], filter[1], nin, nout]
        print name, input.get_shape().as_list(), "     Filter:", shape
        initial = tf.truncated_normal(shape, stddev=np.sqrt(2.0/(filter[0]*filter[1]*nin)))
        w = tf.Variable(initial, name="W_"+name)
        initial = tf.constant(0.1, shape=[nout])
        b = tf.Variable(initial, name="b_"+name)

        lrn = tf.nn.local_response_normalization(input)
        net = tf.nn.conv2d(lrn, w, [1, stride, stride, 1], padding="SAME")
        if activation is not None:
            net = activation(net + b)
    return net

#########################
##### Network
#########################
net = conv2d(x, 32)
net = conv2d(net, 32)
net = conv2d(net, 48)
net = conv2d(net, 48)
net = conv2d(net, 48)
net = conv2d(net, 48, stride=2)
#net = max_pool(net, [2,2])
net = conv2d(net, 80)
net = conv2d(net, 80)
net = conv2d(net, 80)
net = conv2d(net, 80)
net = conv2d(net, 80)
net = conv2d(net, 80, stride=2)
#net = max_pool(net, [2,2])
net = conv2d(net, 128)
net = conv2d(net, 128)
net = conv2d(net, 128)
net = conv2d(net, 128)
net = conv2d(net, 128)
net = max_pool(net, [8,8])
net = flatten(net)
net = fc(net, 500)
y = fc(net, 10, activation=tf.nn.softmax)
