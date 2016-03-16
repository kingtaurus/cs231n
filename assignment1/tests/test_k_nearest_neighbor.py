'''
HOW TO RUN THIS CODE (if tests are within the assignment 1 root):
python -m py.test test_k_nearest_neighbor.py -vv -s -q
python -m py.test test_k_nearest_neighbor.py -vv -s -q --cov

py.test.exe --cov=cs231n/ test_k_nearest_neighbor.py --cov-report html

(if the tests are within the subfolder tests)
PYTHONPATH=${PWD} py.test.exe tests/ -v --cov-report html
python -m pytest tests -v --cov-report html

Open index.html contained within htmlcov
'''

import pytest
import numpy as np
import random

from collections import defaultdict, OrderedDict, Counter

from cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor
from cs231n.data_utils import load_CIFAR10, load_CIFAR_batch

batch_id = random.choice(list(range(1,6)))

def test_assert():
    assert 1

#TODO: these fixtures should probably be moved to a single file
#since they will be used to test a bunch of classifiers

#TODO: reshaping the training/testing data sets should be handled
#      within a different fixture (reasoning: want to test the
#      the behaviour when the 'incorrect default shape is passed in')




#fixtures used to construct data set
@pytest.fixture(scope='module')
def Xtrain():
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, *_ = load_CIFAR10(cifar10_dir)
    return X_train

#fixtures used to construct data set
@pytest.fixture(scope='module')
def ytrain():
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    return y_train

@pytest.fixture(scope='module')
def Xtrain_small():
    cifar10_batch_dir = 'cs231n/datasets/cifar-10-batches-py/data_batch_%d' % (batch_id)
    X_train, y_train = load_CIFAR_batch(cifar10_batch_dir)
    return X_train

@pytest.fixture(scope='module')
def ytrain_small():
    cifar10_batch_dir = 'cs231n/datasets/cifar-10-batches-py/data_batch_%d' % (batch_id)
    X_train, y_train = load_CIFAR_batch(cifar10_batch_dir)
    return y_train

@pytest.fixture(scope='module')
def Xtest():
    cifar10_test_dir = 'cs231n/datasets/cifar-10-batches-py/test_batch'
    X_test, y_test = load_CIFAR_batch(cifar10_test_dir)
    return X_test

@pytest.fixture(scope='module')
def ytest():
    cifar10_test_dir = 'cs231n/datasets/cifar-10-batches-py/test_batch'
    X_test, y_test = load_CIFAR_batch(cifar10_test_dir)
    return y_test

@pytest.fixture(scope='function')
def sample_train(Xtrain, ytrain, count=3000):
    def make_sample(count=count):
        if count > ytrain.shape[0]:
            count = random.uniform(50,ytrain.shape[0])
        idx = np.random.choice(np.arange(len(ytrain)), count, replace=False)
        return Xtrain[idx], ytrain[idx]
    return make_sample

@pytest.fixture(scope='function')
def sample_test(Xtest, ytest, count=1000):
    def make_sample(count=count):
        if count > ytest.shape[0]:
            count = random.uniform(50,ytest.shape[0])
        idx = np.random.choice(np.arange(len(ytest)), count, replace=False)
        return Xtest[idx], ytest[idx]
    return make_sample

def test_Xtrain_shape(Xtrain):
    assert Xtrain[0].ndim == 3
    assert Xtrain.shape == (50000,32,32,3)

def test_ytrain_shape(ytrain):
    assert ytrain.ndim == 1
    assert ytrain.shape == (50000,)

def test_Xtest_shape(Xtest):
    assert Xtest[0].ndim == 3
    assert Xtest.shape == (10000,32,32,3)

def test_ytest_shape(ytest):
    assert ytest.ndim == 1
    assert ytest.shape == (10000,)

def test_Xtrain_small_shape(Xtrain_small):
    assert Xtrain_small[0].ndim == 3
    assert Xtrain_small.shape == (10000,32,32,3)

def test_sampling(sample_train):
    Xtrain, ytrain = sample_train(count=3500)
    assert Xtrain.shape == (3500,32,32,3)
    assert ytrain.shape == (3500,)

def test_KNN_train(sample_train, sample_test):
    #this test is designed to verify that input shapes are correct
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    with pytest.raises(ValueError):
        knn = KNearestNeighbor()
        knn.train(Xtrain, ytrain)

def test_KNN_train_1(sample_train, sample_test):
    #this test is designed to verify that input shapes are correct
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))

    with pytest.raises(ValueError):
        knn = KNearestNeighbor()
        knn.train(Xtrain, Xtrain)

def test_KNN_train_reshape_input(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

    knn = KNearestNeighbor()
    knn.train(Xtrain,ytrain)

def test_KNN_dists_twoloop_shape(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

    knn = KNearestNeighbor()
    knn.train(Xtrain,ytrain)
    assert knn.compute_distances_two_loops(Xtest).shape == (Xtest.shape[0], Xtrain.shape[0])

def test_KNN_dists_oneloop_shape(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

    knn = KNearestNeighbor()
    knn.train(Xtrain,ytrain)
    assert knn.compute_distances_one_loop(Xtest).shape == (Xtest.shape[0], Xtrain.shape[0])



# @pytest.mark.parametrize("train_count, test_count", [(10 * x + 1, 5 * x + 1) for x in range(1,10)])
### if you have an equal number of values
#@pytest.mark.parametrize("train_count", [10 * x + 1 for x in range(1,10)])
#@pytest.mark.parametrize("test_count", [10 * x + 1 for x in range(1,11)])
# def test_KNN_dists_noloop_shape(sample_train, sample_test, train_count, test_count):
#
# This is nice because it tests over a bunch of values
#
@pytest.mark.parametrize("in_count", [150 * x + 1 for x in range(1,10)])
def test_KNN_dists_noloop_shape(sample_train, sample_test, in_count):
    Xtrain, ytrain = sample_train(count=in_count)
    Xtest, ytest   = sample_test(count=in_count-30)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

    knn = KNearestNeighbor()
    knn.train(Xtrain,ytrain)
    assert knn.compute_distances_no_loops(Xtest).shape == (Xtest.shape[0], Xtrain.shape[0])

def test_KNN_dists_two_to_one(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

    knn = KNearestNeighbor()
    knn.train(Xtrain,ytrain)
    dist_two = knn.compute_distances_two_loops(Xtest)
    dist_one = knn.compute_distances_one_loop(Xtest)
    assert np.linalg.norm(dist_two - dist_one, ord='fro') < 0.001

def test_KNN_dists_two_to_none(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

    knn = KNearestNeighbor()
    knn.train(Xtrain,ytrain)
    dist_two = knn.compute_distances_two_loops(Xtest)
    dist_no  = knn.compute_distances_no_loops(Xtest)
    assert np.linalg.norm(dist_two - dist_no, ord='fro') < 0.001

def test_KNN_dists_one_to_none(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

    knn = KNearestNeighbor()
    knn.train(Xtrain,ytrain)
    dist_one = knn.compute_distances_one_loop(Xtest)
    dist_no  = knn.compute_distances_no_loops(Xtest)
    assert np.linalg.norm(dist_one - dist_no, ord='fro') < 0.001

def test_KNN_predict_labels_shape(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

    knn = KNearestNeighbor()
    knn.train(Xtrain,ytrain)

    dist_no  = knn.compute_distances_no_loops(Xtest)
    assert knn.predict_labels(dist_no, k=1).shape == ytest.shape
    assert knn.predict_labels(dist_no, k=2).shape == ytest.shape
    assert knn.predict_labels(dist_no, k=3).shape == ytest.shape
    assert knn.predict_labels(dist_no, k=4).shape == ytest.shape

@pytest.mark.parametrize("k", list(range(-1,-10,-1)))
def test_KNN_predict_k_outofrange_parameter(sample_train, sample_test, k):
    Xtrain, ytrain = sample_train(count=40)
    Xtest,  ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

    knn = KNearestNeighbor()
    knn.train(Xtrain,ytrain)

    with pytest.raises(ValueError):
        knn.predict(Xtest,k)

@pytest.mark.parametrize("k", list(range(1,3)))
def test_KNN_predict_k_parameter(sample_train, sample_test, k):
    Xtrain, ytrain = sample_train(count=40)
    Xtest,  ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

    knn = KNearestNeighbor()
    knn.train(Xtrain,ytrain)

    assert knn.predict(Xtest,k).shape == ytest.shape

@pytest.mark.parametrize("num_loops", list(range(-1,-10,-1)) + list(range(3,10,1)))
def test_KNN_predict_num_loop_parameter(sample_train, sample_test, num_loops):
    Xtrain, ytrain = sample_train(count=40)
    Xtest,  ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

    knn = KNearestNeighbor()
    knn.train(Xtrain,ytrain)

    with pytest.raises(ValueError):
        knn.predict(Xtest,0,num_loops).shape

@pytest.mark.parametrize("num_loops", list(range(0,3)))
@pytest.mark.parametrize("k", [1])
def test_KNN_predict_loop_parameter(sample_train, sample_test, k, num_loops):
    Xtrain, ytrain = sample_train(count=40)
    Xtest,  ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

    knn = KNearestNeighbor()
    knn.train(Xtrain,ytrain)

    assert knn.predict(Xtest,k,num_loops).shape == ytest.shape

def test_KNN_predict_incorrect_shape(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=500)
    Xtest, ytest   = sample_test(count=125)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

    knn = KNearestNeighbor()
    knn.train(Xtrain,ytrain)

    with pytest.raises(ValueError):
        knn.predict(ytrain)#using ytrain, shich has incorrect dimensions;

#options are 'module', 'fixture', 'session'
### Extra Test Code: ###
@pytest.fixture(scope='module')
def inlists():
    import random
    sample_list = []
    possible_multiple = []
    for i in range(0,100):
        sample_list.append(random.sample(range(0,10),5))
        possible_multiple.append(sample_list[i])

    for i in range(0,100):
        possible_multiple.append([random.choice(range(0,5)) for j in range(0,4)])

    return sample_list, possible_multiple

# def list_to_tests():
#     import random
#     for i in range(0,20):
#         print(random.sample(range(0, 10), 4))
#         #generates a set of 4 unique random samples from [0,10)
#         print([random.choice(range(0,5)) for j in range(0,4)])
#         #non-unique samples




## Use list_to_tests to generate test cases
## (1) should be able to use list operations to check numpy behaviour
##     (i.e. Counter, OrderedDict, etc. to check argmax, argsort, argwhere)
## (2)
def argmax(lst):
    return list.index(max(lst))

def argmax_alt(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def counter_t(iterable):
    return Counter(iterable)

def argwhere(lst, value):
    return [i for i,x in enumerate(lst) if x == value]

def argwhere_alt(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

def argsort_1(seq):
    # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3383106#3383106
    #non-lambda version by Tony Veijalainen
    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]

def argsort_2(seq):
    # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3383106#3383106
    #lambda version by Tony Veijalainen
    return [x for x,y in sorted(enumerate(seq), key = lambda x: x[1])]

def argsort_3(seq):
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    #by unutbu
    return sorted(range(len(seq)), key=seq.__getitem__)


###
# # How to handle multiple parameterizations of tests
###

# testdata_1 = [ x * 10 for x in range(1,10)]
# testdata_2 = [ x * 10 for x in range(1,15)]

# @pytest.mark.parametrize("a", testdata_1)
# def test_a(a):
#     assert a > 5

# @pytest.mark.parametrize("a", testdata_1)
# @pytest.mark.parametrize("b", testdata_2)
# def test_ab(a,b):
#     assert a > 5
#     assert b < 150


#requires -s to be passed
@pytest.fixture(params=["mysql","pg"], scope='function')
def db(request):
    def fin():
        print(" teardown")
    request.addfinalizer(fin)
    return request.param

#requires
def test_db(db):
    assert len(db) > 1














