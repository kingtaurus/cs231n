import tensorflow as tf
import numpy as np
import scipy.io

def conv_layer(in_layer, kernel, bias, strides=None):
    return tf.nn.conv2d(in_layer, tf.constant(kernel), strides=(1,1,1,1), padding="SAME") + bias

def pool_layer(in_layer, ksize=None, strides=None):
    return tf.nn.avg_pool(in_layer, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')

def relu_layer(in_layer):
    return tf.nn.relu(in_layer)

def load_graph(data_path, input_image=None):
    data = scipy.io.loadmat(data_path)
    vgg_layers = data['layers']
    weights = vgg_layers[0]
    net_weights = {}
    layers = []

    ##layer name
    ##weights[i][0][0][0]

    ##layer type
    ##weights[i][0][0][1]

    ##layer weights
    ##weights[i][0][0][2]

    ##layer shape
    ##weights[i][0][0][3]
    ## conv == [[ ky, ky, input_filters, output_filters ]]
    ## pool == [[ ky, kx ]]
    ## relu == []

    ##layer stride
    ##weights[i][0][0][4]
    ## conv == [[ sy, sx ]]
    ## pool == [[ sy, sx ]]
    ## relu == [[ 0 ]]

    net_graph = {}
    net_graph['input'] = input_image

    last = 'input'

    for i in range(vgg_layers.shape[1]):
        if "fc" in weights[i][0][0][0][0]:
            break
        #add layer name
        layer_name = weights[i][0][0][0][0]
        layer_type = weights[i][0][0][1][0]

        layers.append(layer_name)

        #layer is conv layer grab the weights
        if "conv" == layer_type:
            layer_shape = weights[i][0][0][3][0]
            strides = weights[i][0][0][4][0]
            W, b = weights[i][0][0][2][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            W = np.transpose(W, (1,0,2,3))
            b = b.reshape(-1)
            net_weights[layer_name] = (W,b)
            print(layer_name, W.shape, b.shape)
            net_graph[layer_name] = conv_layer(net_graph[last], W, b)
            last = layer_name

        if "relu" == layer_type:
            net_graph[layer_name] = relu_layer(net_graph[last])
            print(layer_name)
            last = layer_name

        if "pool" == layer_type:
            ksize   = weights[i][0][0][3][0]
            strides = weights[i][0][0][4][0]
            
            print(layer_name)

            net_graph[layer_name] = pool_layer(net_graph[last])
            last = layer_name
    net_graph['last'] = net_graph[last]
    return net_graph, net_weights

if __name__ == '__main__':
    test_array = np.random.randn(1,128,128,3).astype(np.float32)
    graph, weights = load_graph(data_path='vgg_weights/imagenet-vgg-verydeep-19.mat', input_image=test_array)
    print(graph['last'].get_shape())