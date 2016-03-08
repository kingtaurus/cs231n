import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # C, D      = W.shape
  # num_train = X.shape[1]

  # for i in np.arange(num_train):
  #   f_i = np.dot(W, X[:,i])
  #   log_c = np.max(f_i)
  #   f_i -= log_c

  #   #sum_i = np.sum(np.exp(f_i))
  #   sum_i = 0.
  #   for f_ij in f_i:
  #     sum_i += np.exp(f_ij)
  #   loss += -f_i[y[i]] + np.log(sum_i)
  #   for j in np.arange(C):
  #     p = np.exp(f_i[j])/sum_i
  #     dW[j, :] += (p-(j == y[i])) * X[:, i]

  # loss /= num_train
  # dW   /= num_train

  # loss += 0.5 * reg * np.sum(W*W)
  # dW   += reg * W
  num_classes = W.shape[0]
  num_train = X.shape[1]

  for i in range(num_train):
    # Compute vector of scores
    f_i = W.dot(X[:, i]) # in R^{num_classes}

    # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
    log_c = np.max(f_i)
    f_i -= log_c

    # Compute loss (and add to it, divided later)
    # L_i = - f(x_i)_{y_i} + log \sum_j e^{f(x_i)_j}
    sum_i = 0.0
    for f_i_j in f_i:
      sum_i += np.exp(f_i_j)
    loss += -f_i[y[i]] + np.log(sum_i)

    # Compute gradient
    # dw_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
    # Here we are computing the contribution to the inner sum for a given i.
    for j in range(num_classes):
      p = np.exp(f_i[j])/sum_i
      dW[j, :] += (p-(j == y[i])) * X[:, i]

  # Compute average
  loss /= num_train
  dW /= num_train

  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # Get shapes
  num_classes = W.shape[0]
  num_train = X.shape[1]

  # Compute scores
  f = np.dot(W, X)
  log_c = np.max(f,axis=0)
  # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
  f -= log_c

  # Loss: L_i = - f(x_i)_{y_i} + log \sum_j e^{f(x_i)_j}
  # Compute vector of stacked correct f-scores: [f(x_1)_{y_1}, ..., f(x_N)_{y_N}]
  # (where N = num_train)
  f_correct = f[y, np.arange(num_train)]
  loss = - np.mean(np.log(np.exp(f_correct)/np.sum(np.exp(f),axis=0)) )

  # Gradient: dw_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
  p = np.exp(f)/np.sum(np.exp(f), axis=0)
  ind = np.zeros(p.shape)
  ind[y, range(num_train)] = 1
  dW = np.dot((p-ind), X.T)
  dW /= num_train

  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
