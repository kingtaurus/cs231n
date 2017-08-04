import numpy as np
from past.builtins import xrange


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
        self.X_train = X
        self.y_train = y

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
        for i in xrange(num_test):
            for j in xrange(num_train):
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
        for i in xrange(num_test):
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
        for i in xrange(num_test):
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
            closest_classes = self.y_train[k_closest_points]
            closest_y       = closest_classes[0]
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            occurences = np.bincount(closest_classes)
            #bincount acts as a counter
            #If you just want the smallest numerical label:
            # y_pred[i] = np.sort(np.argmax(occurences), axis=None)
            max = np.amax(occurences)
            #argmax the maximum value in occurences (i.e. the voting rank, giving all occurences the same weight)
            max_classes = np.argwhere(occurences == max).flatten()
            #set it default to the first element
            if k > 2 and len(max_classes) > 1:
                #if max is greater than one and there are multiple values of max
                #   in occurences ==> this implies something like this has occured:
                #   [2 0 1 0 2] ==> [0 2] ==> want minimal distance (of the max occurences);

                from collections import defaultdict
                class_to_total_dist   = defaultdict(int)
                #Can't use Counter: Elements with equal counts are ordered arbitrarily;

                for x in k_closest_points:
                    if self.y_train[x] in max_classes:
                        class_to_total_dist[self.y_train[x]] += dists[i,x]**2
                closest_min_y = [min(class_to_total_dist, key=class_to_total_dist.get)]
                closest_y = closest_min_y[0]
                #take the minimum element and put it into closest_y
                #NOTE: that closest_y is set earlier to the first element of closest_classes;
            y_pred[i] =  int(closest_y)
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
        y_pred      = np.zeros((num_test,num_classes))
        occurences  = np.zeros(num_classes)

        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            k_closest_points = np.array(np.argsort(dists[i,:])).reshape(-1,)[:k]
            closest_classes = self.y_train[k_closest_points]
            occurences = np.bincount(closest_classes, minlength=num_classes)
            y_pred[i] = occurences / k
        return y_pred
