import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

def after_max_pool_size(in_size, pool_size, pool_stride):
    assert len(in_size)==3
    return (in_size[0], int(1 + (in_size[1] - pool_size[0])/pool_stride), \
                        int(1 + (in_size[2] - pool_size[1])/pool_stride))

class ConvNet(object):
    """
    An 'n' layer convolutional network with the following architecture:

    (conv - relu - (i,j) max pool)) x (n-2) - (conv - relu - affine)x1 - softmax

    The network operates on minibatches of data (input shape (N,C,H,W)). Consisting:
        N images
        H image height
        W image width
        C input channels
    """
    def __init__(self, input_dim=(3,32,32), num_filters=[32,32], filter_size=[7,7],
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialization of a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: list of Number of filters to use in each convolutional layer
        - filter_size: list of Size of filters to use in each convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.D     = input_dim
        num_colors = self.D[0]
        height     = self.D[1]
        width      = self.D[2]
        self.H     = hidden_dim
        self.C     = num_classes

        self.weight_scale = weight_scale
        self.filter_size  = filter_size
        self.num_filters  = num_filters

        assert len(num_filters) == len(filter_size)

        #convolution size: [depth(filter/color), height(spatial), width(spatial)]
        #input_dim (3,32,32)-> (32,32,32)
        # with filters being (3,7,7) [with 32 of them]
        #relu -> maintains shape
        # 32,32,32 -> 32,32,32
        # 2x2 max pool (reduces spatial dimensions,  Height -> (Height - Spatial height) / stride + 1:
        # 32,32,32 -> 32,16,16
        # affine
        # 32,16,16 -> hidden_dim
        # relu:
        # hidden
        # affine
        # hidden -> n_classes -> softmax

        # 3,32,32 -> (num_filter[0],32,32)
        # with filters being 32x(3,7,7)
        # relu -> maintains shape
        # max pool (reduces shape) (32, l, l); l = (Height - Spatial height) / stride + 1

        # (num_filter[0],l,l) -> (num_filter[1],l,l) (conv layer 2)
        # relu -> maintains shape
        # max pool reduces (num_filter[1], l', l')

        self.params['W1'] = np.random.normal(0,self.weight_scale, (num_filters[0], num_colors, filter_size[0], filter_size[0]))
        self.params['b1'] = np.random.normal(0,self.weight_scale, num_filters[0])

        maxpool_size = after_max_pool_size((num_filters[0], height, width), (2,2), 2)

        self.params['W2'] = np.random.normal(0,self.weight_scale, (num_filters[1], num_filters[0], filter_size[1], filter_size[1]))
        self.params['b2'] = np.random.normal(0,self.weight_scale, num_filters[1])


        #assuming only one pooling layer
        #then a fully connected layer
        self.params['W3'] = np.random.normal(0, self.weight_scale, (num_filters[1] * maxpool_size[1] * maxpool_size[2],hidden_dim))
        self.params['b3'] = np.random.normal(0, self.weight_scale, hidden_dim)

        #fully connect layer to output
        self.params['W4'] = np.random.normal(0, self.weight_scale, (hidden_dim, num_classes))
        self.params['b4'] = np.random.normal(0, self.weight_scale, num_classes)

        for k, v in self.params.items():
          self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
    
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': int(1), 'pad': int((filter_size - 1) / 2)}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        scores, _cache_layer_1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        scores, _cache_layer_2 = conv_relu_forward(scores, W2, b2, conv_param)
        scores, _cache_layer_3 = affine_relu_forward(scores, W3, b3)
        scores, _cache_layer_4 = affine_forward(scores, W4, b4)

        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        loss, softmax_dx = softmax_loss(scores, y)
        dLayer3, grads['W4'], grads['b4'] = affine_backward(softmax_dx, _cache_layer_4)
        dLayer2, grads['W3'], grads['b3'] = affine_relu_backward(dLayer3, _cache_layer_3)
        dLayer1, grads['W2'], grads['b2'] = conv_relu_backward(dLayer2, _cache_layer_2)
        dLayer,  grads['W1'], grads['b1'] = conv_relu_pool_backward(dLayer1, _cache_layer_1)

        loss += 0.5 * self.reg * (np.sum(self.params['W1'] * self.params['W1']))
        loss += 0.5 * self.reg * (np.sum(self.params['W2'] * self.params['W2']))
        loss += 0.5 * self.reg * (np.sum(self.params['W3'] * self.params['W3']))
        loss += 0.5 * self.reg * (np.sum(self.params['W4'] * self.params['W4']))

        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']
        grads['W3'] += self.reg * self.params['W3']
        grads['W4'] += self.reg * self.params['W4']

        return loss, grads



class ConvNet_dropout(object):
    """
    An 'n' layer convolutional network with the following architecture:

    (conv - relu - (i,j) max pool)) x (n-2) - (conv - relu - affine)x1 - softmax

    The network operates on minibatches of data (input shape (N,C,H,W)). Consisting:
        N images
        H image height
        W image width
        C input channels
    """
    def __init__(self, input_dim=(3,32,32), num_filters=[32,32], filter_size=[7,7],
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, dropout=0,
                 dtype=np.float32, seed=None):
        """
        Initialization of a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: list of Number of filters to use in each convolutional layer
        - filter_size: list of Size of filters to use in each convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.D     = input_dim
        num_colors = self.D[0]
        height     = self.D[1]
        width      = self.D[2]
        self.H     = hidden_dim
        self.C     = num_classes
        self.use_dropout = dropout > 0

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        self.weight_scale = weight_scale
        self.filter_size  = filter_size
        self.num_filters  = num_filters

        assert len(num_filters) == len(filter_size)

        #convolution size: [depth(filter/color), height(spatial), width(spatial)]
        #input_dim (3,32,32)-> (32,32,32)
        # with filters being (3,7,7) [with 32 of them]
        #relu -> maintains shape
        # 32,32,32 -> 32,32,32
        # 2x2 max pool (reduces spatial dimensions,  Height -> (Height - Spatial height) / stride + 1:
        # 32,32,32 -> 32,16,16
        # affine
        # 32,16,16 -> hidden_dim
        # relu:
        # hidden
        # affine
        # hidden -> n_classes -> softmax

        # 3,32,32 -> (num_filter[0],32,32)
        # with filters being 32x(3,7,7)
        # relu -> maintains shape
        # max pool (reduces shape) (32, l, l); l = (Height - Spatial height) / stride + 1

        # (num_filter[0],l,l) -> (num_filter[1],l,l) (conv layer 2)
        # relu -> maintains shape
        # max pool reduces (num_filter[1], l', l')

        self.params['W1'] = np.random.normal(0,self.weight_scale, (num_filters[0], num_colors, filter_size[0], filter_size[0]))
        self.params['b1'] = np.random.normal(0,self.weight_scale, num_filters[0])

        maxpool_size = after_max_pool_size((num_filters[0], height, width), (2,2), 2)

        self.params['W2'] = np.random.normal(0,self.weight_scale, (num_filters[1], num_filters[0], filter_size[1], filter_size[1]))
        self.params['b2'] = np.random.normal(0,self.weight_scale, num_filters[1])


        #assuming only one pooling layer
        #then a fully connected layer
        self.params['W3'] = np.random.normal(0, self.weight_scale, (num_filters[1] * maxpool_size[1] * maxpool_size[2],hidden_dim))
        self.params['b3'] = np.random.normal(0, self.weight_scale, hidden_dim)

        #fully connect layer to output
        self.params['W4'] = np.random.normal(0, self.weight_scale, (hidden_dim, num_classes))
        self.params['b4'] = np.random.normal(0, self.weight_scale, num_classes)

        for k, v in self.params.items():
          self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
    
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': int(1), 'pad': int((filter_size - 1) / 2)}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        scores, _cache_layer_1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        scores, _cache_layer_2 = conv_relu_forward(scores, W2, b2, conv_param)
        if self.use_dropout:
            scores, _cache_dropout_layer_2 = dropout_forward(scores, self.dropout_param)
        scores, _cache_layer_3 = affine_relu_forward(scores, W3, b3)
        if self.use_dropout:
            scores, _cache_dropout_layer_3 = dropout_forward(scores, self.dropout_param)
        scores, _cache_layer_4 = affine_forward(scores, W4, b4)

        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        loss, softmax_dx = softmax_loss(scores, y)
        dLayer3, grads['W4'], grads['b4'] = affine_backward(softmax_dx, _cache_layer_4)
        if self.use_dropout:
            dLayer3 = dropout_backward(dLayer3, _cache_dropout_layer_3)
        dLayer2, grads['W3'], grads['b3'] = affine_relu_backward(dLayer3, _cache_layer_3)
        if self.use_dropout:
            dLayer2 = dropout_backward(dLayer2, _cache_dropout_layer_2)
        dLayer1, grads['W2'], grads['b2'] = conv_relu_backward(dLayer2, _cache_layer_2)
        dLayer,  grads['W1'], grads['b1'] = conv_relu_pool_backward(dLayer1, _cache_layer_1)

        loss += 0.5 * self.reg * (np.sum(self.params['W1'] * self.params['W1']))
        loss += 0.5 * self.reg * (np.sum(self.params['W2'] * self.params['W2']))
        loss += 0.5 * self.reg * (np.sum(self.params['W3'] * self.params['W3']))
        loss += 0.5 * self.reg * (np.sum(self.params['W4'] * self.params['W4']))

        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']
        grads['W3'] += self.reg * self.params['W3']
        grads['W4'] += self.reg * self.params['W4']

        return loss, grads

pass
