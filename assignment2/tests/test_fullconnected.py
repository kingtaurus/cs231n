#!/usr/bin/env python3

'''
HOW TO RUN THIS CODE (if tests are within the assignment 1 root):
python -m py.test tests/test_fullconnected.py -vv -s -q
python -m py.test tests/test_fullconnected.py -vv -s -q --cov

py.test.exe --cov=cs231n/ tests/test_fullconnected.py --cov-report html

(if the tests are within the subfolder tests)
PYTHONPATH=${PWD} py.test.exe tests/ -v --cov-report html
python -m pytest tests -v --cov-report html

Open index.html contained within htmlcov
'''

import pytest
import numpy as np
import random

from collections                          import defaultdict, OrderedDict, Counter
from cs231n.gradient_check                import grad_check_sparse, eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.data_utils                    import load_CIFAR10, load_CIFAR_batch

from cs231n.layers import affine_forward, affine_backward, relu_forward, relu_backward
from cs231n.layers import softmax_loss, svm_loss
from cs231n.layer_utils import affine_relu_forward, affine_relu_backward
from cs231n.classifiers.fc_net import TwoLayerNet, FullyConnectedNet

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


def test_affine_forward():
	num_inputs = 2
	input_shape = (4, 5, 6)
	output_dim = 3

	input_size = num_inputs * np.prod(input_shape)
	weight_size = output_dim * np.prod(input_shape)

	x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
	w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
	b = np.linspace(-0.3, 0.1, num=output_dim)

	out, _ = affine_forward(x, w, b)
	correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
	                        [ 3.25553199,  3.5141327,   3.77273342]])

	# Compare your output with ours. The error should be around 1e-9.
	assert out.shape == correct_out.shape
	assert rel_error(out, correct_out) < 5e-7

def test_affine_backward():
	x = np.random.randn(10, 2, 3)
	w = np.random.randn(6, 5)
	b = np.random.randn(5)
	dout = np.random.randn(10, 5)

	dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
	dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
	db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

	_, cache = affine_forward(x, w, b)
	dx, dw, db = affine_backward(dout, cache)

	assert dx.shape == dx.shape
	assert dw.shape == dw.shape
	assert db.shape == db.shape

	assert rel_error(dx_num,dx) < 5e-7
	assert rel_error(dw_num,dw) < 5e-7
	assert rel_error(db_num,db) < 5e-7


def test_relu_forward():
	# Test the relu_forward function
	x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

	out, _ = relu_forward(x)
	correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
	                        [ 0.,          0.,          0.04545455,  0.13636364,],
	                        [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])

	# Compare your output with ours. The error should be around 1e-8
	assert out.shape == correct_out.shape
	assert rel_error(out, correct_out) < 5e-7

@pytest.mark.parametrize("dim", list(range(1,8)))
def test_relu_forward_multiple_dim(dim):
    testing_shape = []
    for y in range(0,dim):
        testing_shape.append(np.random.randint(3,8))
    shape = tuple(testing_shape)
    #y = np.random.randn(*testing_shape)
    x = np.random.standard_normal(shape)
    assert x.shape == relu_forward(x)[0].shape
    x[x<0] = 0
    assert rel_error(x, relu_forward(x)[0]) < 5e-7

def test_sandwich_layers(samples = random.randrange(1,10)):
	for x in range(0,samples):
		x = np.random.randn(2, 3, 4)
		w = np.random.randn(12, 10)
		b = np.random.randn(10)

		dout = np.random.randn(2, 10)
		#need an input for gradient being backpropagated into this layer

		out, cache = affine_relu_forward(x, w, b)
		dx, dw, db = affine_relu_backward(dout, cache)

		dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
		dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
		db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)

		assert rel_error(dx_num, dx) < 5e-7
		assert rel_error(dw_num, dw) < 5e-7
		assert rel_error(db_num, db) < 5e-7

		assert dx.shape == dx_num.shape
		assert dw.shape == dw_num.shape
		assert db.shape == db_num.shape
		assert out.shape == dout.shape

@pytest.mark.parametrize("num_classes", [10])
def test_softmax(num_classes, samples=random.randrange(1,10)):
	num_classes, num_inputs = num_classes, 50
	x = 0.001 * np.random.randn(num_inputs, num_classes)
	y = np.random.randint(num_classes, size=num_inputs)

	dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
	loss, dx = softmax_loss(x, y)
	assert dx_num.shape == dx.shape
	assert loss < - np.log( 0.8 / num_classes) and loss > - np.log( 1.2 / num_classes)
	assert rel_error(dx_num, dx) < 5e-7

@pytest.mark.parametrize("num_classes", [10])
def test_svm(num_classes, samples=random.randrange(1,10)):
	num_classes, num_inputs = num_classes, 50
	x = 0.001 * np.random.randn(num_inputs, num_classes)
	y = np.random.randint(num_classes, size=num_inputs)

	dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
	loss, dx = svm_loss(x, y)
	assert dx_num.shape == dx.shape
	assert loss > num_classes * ( 1 - 1.2 / num_classes)
	assert rel_error(dx_num, dx) < 5e-7


def test_twolayerNN():
	assert 1
