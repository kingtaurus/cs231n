import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    if X.ndim > 2:
        raise ValueError("Improperly formatted training input for X.")
    if y.ndim > 1:
        raise ValueError("Improperly formatted training input for y.")
    self.X_train = X
    self.y_train = y
    self.num_classes = len(np.unique(self.y_train))
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)
    if k < 1:
      raise ValueError('Invalid value %d for k (it must be > 0)' % k)
    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        dists[ i, j ] = np.linalg.norm(X[i] - self.X_train[j])
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      dists[i, :] = np.linalg.norm(self.X_train - X[i,:], axis=1)
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    XX = np.square(X).sum(axis=1)
    YY = np.square(self.X_train).sum(axis=1)
    XY = np.dot(X, self.X_train.T)
    dists = np.sqrt(np.matrix(XX).T + YY - 2 * XY)
    # print(np.matrix(XX).T.shape)
    # print(YY.T.shape)
    # print(YY.shape)
    # print(XY)
    #This does the following:
    # (1) np.matrix(XX).T converts this into n_test,1 (column vector)
    # (2) YY is treated as a 'row vector' (note: np.array with dimensions (n_train,),
    #     has the same behaviour when transformed.
    # (3) XY is (n_test,n_train)
    # SO numpy does the following (but vectorized)
    # dists[i,j] = np.sqrt(np.matrix(XX).T[i] + YY[j] - 2 * XY[i,j])
    #########################################################################
    # quick check (showing the behaviour)                                   #
    #########################################################################
    # print(np.sqrt(np.matrix(XX).T[1] + YY[2] - 2 * XY[1,2]))
    # print(dists[1,2])
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      k_closest_points = np.array(np.argsort(dists[i,:])).reshape(-1,)[:k]

      closest_y = list(self.y_train[k_closest_points])

      closest_classes = self.y_train[k_closest_points]
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      #this is testing code (using a size of 4)
      #then there are a bunch of cases that can be used to verify the way things
      #work
      #
      # THIS code should be refactored into function calls
      # (1) NEED unit tests
      # print(np.bincount(self.y_train[dists[i,:].argsort()[:4]]))
      #print(self.y_train[dists[i,:].argsort()])
      #print(self.y_train[dists[i,:].argsort()].flatten()[0:k])
      occurences = np.bincount(closest_y)

      #print(occurences)
      #bincount acts as a counter
      max = np.amax(occurences)
      #amax the maximum value in occurences (i.e. the voting rank, giving all occurences the same weight)
      #print(occurences)
      #print(max)
      # print("result = ", np.argwhere(occurences == max).flatten())
      #print("result = ", np.argwhere(occurences == max).flatten())
      max_classes = list(np.argwhere(occurences == max).flatten())
      #print(max_classes)
      #print(occurences)
      if k > 2 and len(max_classes) > 1:
        #if max is greater than one and there are multiple values of max
        #   in occurences ==> this implies something like this has occured:
        #   [2 0 2 0 ] ==> [0 2] ==> want minimal distance
        # print("multiple occurences")

        #this should be optimized by removing function calls that make duplicate data structures
        # .... dists[i,:].argsort()
        # .... np.argwhere(occurences == max).flatten()
        from collections import defaultdict
        class_to_total_dist   = defaultdict(int)
        # for x in k_closest_points:
        #     if self.y_train[x] in max_classes:
        #         closest_y.append(self.y_train[x])

        #Can't use Counter: Elements with equal counts are ordered arbitrarily:

        for x in k_closest_points:
            if self.y_train[x] in max_classes:
                class_to_total_dist[self.y_train[x]] += dists[i,x]**2
        # print(class_to_total_dist)
        closest_y = [min(class_to_total_dist, key=class_to_total_dist.get)]
        # print(closest_y)
        ## use l2 norm to add distances
        # print(class_to_total_dist)
        # print(min(class_to_total_dist, key=class_to_total_dist.get))
        #print(class_to_total_dist)
        # closest_y = [min(class_to_total_dist, key=class_to_total_dist.get)]
        #print(closest_y)
        #if float(max) > float(k)/2 + 0.0001:
        y_pred[i] =  closest_y[0]
        #########################################################################
        #                           END OF YOUR CODE                            #
        #########################################################################

    return y_pred

  def predict_proba_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a probability of a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,num_classes) containing predicted
             labels for the test data, where y[i] is the predicted naive
             probability label for the test point X[i].
        """
        num_test    = dists.shape[0]
        num_classes = len(np.unique(self.y_train))
        y_pred = np.zeros((num_test,num_classes))
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            k_closest_points = np.array(np.argsort(dists[i,:])).reshape(-1,)[:k]
            closest_y = list(self.y_train[k_closest_points])
            closest_classes = self.y_train[k_closest_points]
            occurences = np.zeros(num_classes)
            occurences = np.bincount(closest_y, minlength=num_classes)
            probability = occurences / k
            y_pred[i] = probability
        return y_pred
