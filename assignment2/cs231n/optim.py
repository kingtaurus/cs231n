import numpy as np

"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
  """
  Performs vanilla stochastic gradient descent.

  config format:
  - learning_rate: Scalar learning rate.
  """
  if config is None:
    config = {}
  config.setdefault('learning_rate', 1e-2)
  w -= config['learning_rate'] * dw
  return w, config


def sgd_momentum(w, dw, config=None):
  """
  Performs stochastic gradient descent with momentum.

  config format:
  - learning_rate: Scalar learning rate.
  - momentum: Scalar between 0 and 1 giving the momentum value.
    Setting momentum = 0 reduces to sgd.
  - velocity: A numpy array of the same shape as w and dw used to store a moving
    average of the gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)
  v = config.get('velocity', np.zeros_like(w))
  
  next_w = None
  #############################################################################
  # TODO: Implement the momentum update formula. Store the updated value in   #
  # the next_w variable. You should also use and update the velocity v.       #
  #############################################################################
  v = config.get('momentum') * v - config.get('learning_rate') * dw
  next_w = w + v
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  config['velocity'] = v

  return next_w, config



def rmsprop(x, dx, config=None):
  """
  Uses the RMSProp update rule, which uses a moving average of squared gradient
  values to set adaptive per-parameter learning rates.

  config format:
  - learning_rate: Scalar learning rate.
  - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
    gradient cache.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - cache: Moving average of second moments of gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('decay_rate', 0.99)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('cache', np.zeros_like(x))

  learning_rate = config['learning_rate']
  decay_rate    = config['decay_rate']
  epsilon       = config['epsilon']
  cache         = config['cache']

  next_x = None
  #############################################################################
  # TODO: Implement the RMSprop update formula, storing the next value of x   #
  # in the next_x variable. Don't forget to update cache value stored in      #  
  # config['cache'].                                                          #
  #############################################################################
  config['cache']  = decay_rate * cache + (1 - decay_rate) * dx.reshape(x.shape) **2
  next_x = x - learning_rate * dx.reshape(x.shape) / np.sqrt(config['cache'] + epsilon)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return next_x, config

def adawindow(x, dx, config=None):
  #ADAGrad using a rolling window
  next_x = None
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('gradient_list', [])
  config.setdefault('window_size', 100)

  window_size = config['window_size']

  config['gradient_list'].append(np.linalg.norm(dx))
  #config['gradient_norm_squared'].append(np.linalg.norm(dx)**2)
  config['gradient_list'] = config['gradient_list'][-window_size:]
  #config['gradient_norm_squared'] = config['gradient_norm_squared'][-window_size:]

  grad_norm = np.sqrt(np.sum(np.array(config['gradient_list'])**2))
  # print(np.linalg.norm(dx),grad_norm )
  next_x = x - config['learning_rate'] * dx / (np.sqrt(grad_norm) + 1e-10)
  #note: since dx has already been added, |dx| should be strickly less than np.sqrt(grad_norm)

  return next_x, config

# def adadelta(x, dx, config=None):
#   #ADAGrad using accumulation buffer to estimate means and variances
#   next_x = None
#   if config is None: config = {}
#   config.setdefault('t',0)
#   config.setdefault('learning_rate',0.01)
#   config.setdefault('momentum', 0.9)

#   config.setdefault('mean_dx', np.zeros_like(dx))
#   config.setdefault('grad_squared', np.zeros_like(dx))

#   config.setdefault('mean_delta_x', np.zeros_like(x))
#   config.setdefault('delta_x_squared', np.zeros_like(x))

#   #http://caffe.berkeleyvision.org/tutorial/solver.html
#   t = config['t']
#   if t == 0:
#     print(dx.shape, np.mean(dx, axis=0).shape)
#   momentum = config['momentum']
#   delta_x = np.zeros_like(x)

#   config['mean_dx']      += momentum * config['mean_dx'] +      (1 - momentum) * np.mean(dx,axis=0)
#   config['grad_squared'] += momentum * config['grad_squared'] + (1 - momentum) * np.mean(dx**2,axis=0)

#   rms_grad  = np.sqrt(config['grad_squared'] - config['mean_dx'] + 1e-8)
#   rms_delta = np.sqrt(config['delta_x_squared'] - config['mean_delta_x'] + 1e-8)
#   delta_x = - dx * np.linalg.norm(rms_delta) / np.linalg.norm(rms_grad)
#   # if rms_grad > 1000:
#   #   print(rms_delta / rms_grad)

#   config['mean_delta_x']    += momentum * config['mean_delta_x'] +    (1 - momentum) * delta_x
#   config['delta_x_squared'] += momentum * config['delta_x_squared'] + (1 - momentum) * delta_x**2

#   next_x = x + config['learning_rate'] * delta_x

#   config['t'] += 1
#   return next_x, config

def adam(x, dx, config=None):
  """
  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.

  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
  - beta2: Decay rate for moving average of second moment of gradient.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - m: Moving average of gradient.
  - v: Moving average of squared gradient.
  - t: Iteration number.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(x))
  config.setdefault('v', np.zeros_like(x))
  config.setdefault('t', 0)

  learning_rate = config['learning_rate']
  beta1         = config['beta1']
  beta2         = config['beta2']
  epsilon       = config['epsilon']
  m             = config['m']
  v             = config['v']
  t             = config['t']
  x_shape = x.shape
  
  next_x = None
  #############################################################################
  # TODO: Implement the Adam update formula, storing the next value of x in   #
  # the next_x variable. Don't forget to update the m, v, and t variables     #
  # stored in config.                                                         #
  #############################################################################
  m_hat = None
  v_hat = None
  t += 1
  m = beta1 * m + (1 - beta1) * dx.reshape(x.shape)
  v = beta2 * v + (1 - beta2) * np.power(dx,2).reshape(x.shape)

  next_x = x - learning_rate * m / (1 - beta1 ** t) / (np.sqrt(v / ( 1 - beta2 ** t)) + epsilon)

  config['t'] = t
  config['m'] = m
  config['v'] = v
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return next_x, config

  
  
  

