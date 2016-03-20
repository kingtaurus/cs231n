'''
HOW TO RUN THIS CODE (if tests are within the assignment 1 root):
python -m py.test tests/test_features.py -vv -s -q
python -m py.test tests/test_features.py -vv -s -q --cov

py.test.exe --cov=cs231n/ tests/test_features.py --cov-report html

(if the tests are within the subfolder tests)
PYTHONPATH=${PWD} py.test.exe tests/ -v --cov-report html
python -m pytest tests -v --cov-report html

Open index.html contained within htmlcov
'''


import pytest
import numpy as np
import random

from collections import defaultdict, OrderedDict, Counter

from cs231n.classifiers.linear_classifier import LinearSVM
from cs231n.classifiers.linear_svm        import svm_loss_naive, svm_loss_vectorized
from cs231n.classifiers                   import Softmax
from cs231n.classifiers.softmax           import softmax_loss_naive, softmax_loss_vectorized

from cs231n.classifiers.neural_net        import TwoLayerNet
from cs231n.gradient_check                import grad_check_sparse, eval_numerical_gradient
from cs231n.data_utils                    import load_CIFAR10, load_CIFAR_batch

batch_id = random.choice(list(range(1,6)))

def rel_error(x,y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def test_assert():
    assert 1

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

@pytest.fixture(scope='function')
def sample_train_with_bias(Xtrain, ytrain, count=3000):
    #Reshape the copy
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0],-1))

    #Add bias to the copy
    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])
    def make_sample(count=count):
        if count > ytrain.shape[0]:
            count = random.uniform(50, ytrain.shape[0])
        idx = np.random.choice(np.arange(len(ytrain)), count, replace=False)
        return Xtrain[idx], ytrain[idx]
    return make_sample

@pytest.fixture(scope='function')
def sample_test_with_bias(Xtest, ytest, count=3000):
    #Reshape the copy
    Xtest = np.reshape(Xtest, (Xtest.shape[0],-1))

    #Add bias to the copy
    Xtest = np.hstack([Xtest, np.ones((Xtest.shape[0], 1))])
    def make_sample(count=count):
        if count > ytest.shape[0]:
            count = random.uniform(50, ytest.shape[0])
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

