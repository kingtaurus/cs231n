'''
HOW TO RUN THIS CODE (if tests are within the assignment 1 root):
python -m py.test tests/test_softmax.py -vv -s -q
python -m py.test tests/test_softmax.py -vv -s -q --cov

py.test.exe --cov=cs231n/ test_softmax.py --cov-report html

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
from cs231n.gradient_check                import grad_check_sparse, eval_numerical_gradient
from cs231n.data_utils                    import load_CIFAR10, load_CIFAR_batch

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

@pytest.fixture(scope='function')
def sample_train_with_bias(Xtrain, ytrain, count=3000):
	Xtrain_copy = np.copy(Xtrain)

	#Reshape the copy
	Xtrain_copy = np.reshape(Xtrain_copy, (Xtrain_copy.shape[0],-1))

	#Add bias to the copy
	Xtrain_copy = np.hstack([Xtrain_copy, np.ones((Xtrain_copy.shape[0], 1))])
	def make_sample(count=count):
		if count > ytrain.shape[0]:
			count = random.uniform(50, ytrain.shape[0])
		idx = np.random.choice(np.arange(len(ytrain)), count, replace=False)
		return Xtrain_copy[idx], ytrain[idx]
	return make_sample

@pytest.fixture(scope='function')
def sample_test_with_bias(Xtest, ytest, count=3000):
	Xtest_copy = np.copy(Xtest)

	#Reshape the copy
	Xtest_copy = np.reshape(Xtest_copy, (Xtest_copy.shape[0],-1))

	#Add bias to the copy
	Xtest_copy = np.hstack([Xtest_copy, np.ones((Xtest_copy.shape[0], 1))])
	def make_sample(count=count):
		if count > ytest.shape[0]:
			count = random.uniform(50, ytest.shape[0])
		idx = np.random.choice(np.arange(len(ytest)), count, replace=False)
		return Xtest_copy[idx], ytest[idx]
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

def test_sampling_with_bias(sample_train, sample_train_with_bias, count=3500):
	Xtrain_no_bias, y_train_no_bias = sample_train(count=count)
	Xtrain, ytrain = sample_train_with_bias(count=count)
	assert Xtrain_no_bias.shape == (count,32,32,3)
	assert y_train_no_bias.shape == (count,)

	assert Xtrain.shape[0] == Xtrain_no_bias.shape[0]

	assert np.prod(Xtrain.shape) > np.prod(Xtrain_no_bias.shape)

	#assert that the dimensions
	assert (np.prod(Xtrain.shape) / Xtrain.shape[0]) == (1 + np.prod(Xtrain_no_bias.shape)/Xtrain_no_bias.shape[0])
