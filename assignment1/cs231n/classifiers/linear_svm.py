import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  if W.shape[0] - X.shape[1] == 1:
    print("\nBias dimension probably has not been added to X!\n")

  if W.shape[0] != X.shape[1]:
    raise ValueError("Shape of the weight array (W) doesn't match shape of training data (X)!")

  if X.shape[0] != y.shape[0]:
    raise ValueError("X and y have incompatible shapes.\n" +
                             "X has %s samples, but y has %s." %
                             (X.shape[0], y.shape[0]))
  if X.ndim != 2:
    raise ValueError("Require X to have X.ndim = 2;")

  if W.ndim != 2:
    raise ValueError("Require W to have W.ndim = 2;")

  if y.ndim != 1:
    raise ValueError("True class labels (y) should be input as a vector.")

  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #because of the continue above, if j==correct class, THEN
        # INSIDE the if means we don't have to multiply by
        #        (margin > 0)
        dW[:,j]    += X[i,:]
        dW[:,y[i]] -= X[i,:]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  dW /= num_train
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  if W.shape[0] - X.shape[1] == 1:
    print("Bias dimension has not been added to X!")
  if W.shape[0] != X.shape[1]:
    raise ValueError("Shape of the weight array (W) doesn't match shape of training data (X)!")
  if X.shape[0] != y.shape[0]:
    raise ValueError("X and y have incompatible shapes.\n" +
                             "X has %s samples, but y has %s." %
                             (X.shape[0], y.shape[0]))
  if X.ndim != 2:
    raise ValueError("Require X to have X.ndim = 2;")
  if W.ndim != 2:
    raise ValueError("Require W to have W.ndim = 2;")
  if y.ndim != 1:
    raise ValueError("True class labels (y) should be input as a vector.")

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes  = W.shape[1]
  num_features = W.shape[0]
  num_train    = X.shape[0]

  scores         = np.dot(X,W)

  correct_scores = scores[np.arange(num_train), y]
  #this means: suppose that the correct classes are [9, 4, 2, ...] (vector)
  #            scores[0,9] is the score-weight for the first sample (training)
  #            scores[1,4] is the score-weight for the second sample (training)
  # (i.e. this samples the distribution at the i-th row, 'jth' column)
  # returns a new vector (correct_scores)
  margin = scores - correct_scores.reshape(num_train,1) + 1
  #remove the correct score from all elements
  #for margins estimate for each class
  margin[np.arange(num_train), y] = 0
  #remove the true class
  thresh = np.maximum(np.zeros((num_train,num_classes)), margin)

  #sum over all non-correct classes
  loss = np.sum(thresh)
  loss /= num_train

  loss += 0.5 * reg * np.sum(W*W)
  #np.sum(W*W) or np.linalg.norm(W)**2

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  thresh[thresh > 0] = 1
  above_threshold = np.sum(thresh, axis=1)
  #add all the columns together (i.e. form a new vector with # of rows elements
  #This corresponds to the number of classes that are incorrectly classified  

  thresh[np.arange(num_train),y] = -above_threshold[np.arange(num_train)]
  #NOTE: I found that this was faster than using :,y
  #      unsure if this is due to copying or some other checks being called
  #      during vectorized operations.

  dW = np.dot(X.T, thresh)

  dW /= num_train
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
