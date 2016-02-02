import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


#might be easier to build some helper parameters
#i.e. filter_bank(count,size,size)
# build_conv_net_layer(input_shape, filter_bank)
# note: options can be activation, drop, pool, batchnorm, resnet(?)
# build_affine_connection(filter_bank.shape, hidden_layer)
# build_affine_classification(hidden_layer, classification_size)
# classification_loss_function(type)
# 
# 

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
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
    self.H     = hidden_dim
    self.C     = num_classes

    self.weight_scale = weight_scale
    self.filter_size = filter_size
    self.num_filters = num_filters

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
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    #conv - relu - 2x2 max pool - affine - relu - affine - softmax
    #
    self.params['W1'] = np.random.normal(0,self.weight_scale, (num_filters, filter_size, filter_size))#covnet parameters
    self.params['b1'] = np.random.normal(0,self.weight_scale, num_filters)

    self.params['W2'] = np.random.normal(0,self.weight_scale, (32,32,hidden_dim))
    self.params['b2'] = np.random.normal(0,self.weight_scale, hidden_dim)

    self.params['W3'] = np.random.normal(0,self.weight_scale, (hidden_dim,num_classes))
    self.params['b3'] = np.random.normal(0,self.weight_scale, num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
