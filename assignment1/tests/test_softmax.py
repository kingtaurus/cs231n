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

from cs231n.classifiers                   import Softmax
from cs231n.classifiers.softmax           import softmax_loss_naive, softmax_loss_vectorized
from cs231n.gradient_check                import grad_check_sparse, eval_numerical_gradient
from cs231n.data_utils                    import load_CIFAR10, load_CIFAR_batch

batch_id = random.choice(list(range(1,6)))

def rel_error(x,y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

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

def test_sampling_with_bias(sample_train, sample_train_with_bias, count=3500):
    Xtrain_no_bias, y_train_no_bias = sample_train(count=count)
    Xtrain, ytrain = sample_train_with_bias(count=count)
    assert Xtrain_no_bias.shape == (count,32,32,3)
    assert y_train_no_bias.shape == (count,)

    assert Xtrain.shape[0] == Xtrain_no_bias.shape[0]

    assert np.prod(Xtrain.shape) > np.prod(Xtrain_no_bias.shape)

    #assert that the dimensions
    assert (np.prod(Xtrain.shape) / Xtrain.shape[0]) == (1 + np.prod(Xtrain_no_bias.shape)/Xtrain_no_bias.shape[0])


def test_softmax_loss_naive_no_bias_X(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=20)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))

    #i.e. using the correct W size
    W = np.random.randn(Xtrain.shape[1] + 1,10) * 0.0001
    with pytest.raises(ValueError):
        loss, grad = softmax_loss_naive(W, Xtrain, ytrain, 1e2)
    #this will fail because W is larger by 1

def test_softmax_loss_vectorized_no_bias_X(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=20)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))

    W = np.random.randn(Xtrain.shape[1] + 1,10) * 0.0001
    with pytest.raises(ValueError):
        loss, grad = softmax_loss_vectorized(W, Xtrain, ytrain, 1e2)
    #this will fail because W is larger by 1


def test_softmax_loss_naive_no_bias_W(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=20)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    #using the incorrect W size
    W = np.random.randn(Xtrain.shape[1],10) * 0.0001

    #add the bias dimension
    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])

    with pytest.raises(ValueError):
        loss, grad = softmax_loss_naive(W, Xtrain, ytrain, 1e2)
    #this will fail because W is larger by 1

def test_softmax_loss_vectorized_no_bias_W(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=20)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    #using the incorrect W size
    W = np.random.randn(Xtrain.shape[1],10) * 0.0001

    #add the bias dimension (to X)
    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])

    with pytest.raises(ValueError):
        loss, grad = softmax_loss_vectorized(W, Xtrain, ytrain, 1e2)
    #this will fail because W is larger by 1


@pytest.mark.parametrize("train_count", [10 * x + 1 for x in range(1,10)])
def test_softmax_loss_naive_vectorized_comparison(sample_train_with_bias, train_count):
    Xtrain, ytrain = sample_train_with_bias(count=train_count)

    W = np.random.randn(Xtrain.shape[1],10) * 0.0001
    loss, grad = softmax_loss_vectorized(W, Xtrain, ytrain, 1e2)
    loss_naive, grad_naive = softmax_loss_naive(W, Xtrain, ytrain, 1e2)
    assert np.abs(loss - loss_naive) < 0.0001
    assert np.linalg.norm(grad - grad_naive) < 0.0001

@pytest.mark.parametrize("train_count", [10 * x + 1 for x in range(1,10)])
@pytest.mark.parametrize("reg", [10. * x + 1 for x in range(10,100000, 10000)])
def test_softmax_loss_naive_vectorized_comparison_reg(sample_train, train_count, reg):
    Xtrain, ytrain = sample_train(count=train_count)
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])

    W = np.random.randn(Xtrain.shape[1],10) * 0.0001
    loss, grad = softmax_loss_vectorized(W, Xtrain, ytrain, reg)
    loss_naive, grad_naive = softmax_loss_naive(W, Xtrain, ytrain, reg)
    assert np.abs(loss - loss_naive) < 0.0001
    assert np.linalg.norm(grad - grad_naive) < 0.0001

@pytest.mark.parametrize("train_count", [10 * x + 1 for x in range(1,10)])
@pytest.mark.parametrize("reg", [10. * x + 1 for x in range(10,100000, 10000)])
def test_softmax_loss_vectorized_comparison_mean(sample_train, train_count, reg):
    Xtrain, ytrain = sample_train(count=train_count)
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))

    mean_image = np.mean(Xtrain, axis=0)
    Xtrain_mean_removed = Xtrain - mean_image

    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])
    Xtrain_mean_removed = np.hstack([Xtrain_mean_removed, np.ones((Xtrain_mean_removed.shape[0], 1))])

    W = np.random.randn(Xtrain.shape[1],10) * 0.0001
    loss, grad = softmax_loss_vectorized(W, Xtrain, ytrain, reg)
    loss_mean_removed, grad_mean_removed = softmax_loss_vectorized(W, Xtrain_mean_removed, ytrain, reg)
    #assert np.abs(loss - loss_mean_removed) > 0.01
    assert np.linalg.norm(grad - grad_mean_removed) > 1.0

@pytest.mark.parametrize("train_count", [50])
def test_softmax_loss_vectorized_numerical_gradient(sample_train, train_count, reg=0.0):
    Xtrain, ytrain = sample_train(count=train_count)
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    mean_image = np.mean(Xtrain, axis=0)
    Xtrain -= mean_image
    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])

    W = np.random.randn(Xtrain.shape[1],10) * 0.0001
    loss, grad = softmax_loss_vectorized(W, Xtrain, ytrain, 0.)

    f = lambda w: softmax_loss_vectorized(w, Xtrain, ytrain, reg)[0]
    g = lambda w: softmax_loss_vectorized(w, Xtrain, ytrain, reg)[1]

    grad_analytic = g(W)
    param_grad_num = eval_numerical_gradient(f, W, verbose=False, h=1e-7)
    assert rel_error(param_grad_num, grad_analytic) < 1e-4


@pytest.mark.parametrize("train_count", [50,100,200,500,1000,2000])
def test_softmax_loss_vectorized_numerical_check(sample_train, train_count, reg=0.0, check_count=20):
    Xtrain, ytrain = sample_train(count=train_count)
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    mean_image = np.mean(Xtrain, axis=0)
    Xtrain -= mean_image
    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])

    W = np.random.randn(Xtrain.shape[1],10) * 0.0001
    loss, grad = softmax_loss_vectorized(W, Xtrain, ytrain, 0.)

    f = lambda w: softmax_loss_vectorized(w, Xtrain, ytrain, 0.0)[0]
    g = lambda w: softmax_loss_vectorized(w, Xtrain, ytrain, 0.0)[1]
    #(f(W+vec(h)) - f(W-vec(h)))/2/|vec(h)| = approximately dot(f'(W),vec(h)) * vec(h)
    #grad(loss) = grad vectorized

    num_checks = check_count
    grad_analytic = g(W)
    for i in range(num_checks):
        ix = tuple([random.randrange(m) for m in W.shape])
        shift = np.zeros(W.shape)
        shift[ix] = 1e-7
        grad_numerical = (f(W + shift) - f(W - shift)) / (2 * 1e-7)
        assert( abs(grad_numerical - grad_analytic[ix]) / (abs(grad_numerical) + abs(grad_analytic[ix])) < 0.0001)

def test_softmax_train(sample_train, sample_test):
    #this test is designed to verify that input shapes are correct
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    with pytest.raises(ValueError):
        #Xtrain has the wrong shape (that is why there is a value error)
        softmax = Softmax()
        softmax.train(Xtrain, ytrain)

def test_softmax_train_1(sample_train, sample_test):
    #this test is designed to verify that input shapes are correct
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])

    with pytest.raises(ValueError):
        softmax = Softmax()
        softmax.train(Xtrain, Xtrain)

def test_softmax_train_2(sample_train, sample_test):
    #this test is designed to verify that input shapes are correct
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])

    with pytest.raises(ValueError):
        #this will catcha Valueerror associated with bad unpacking of a tuple
        softmax = Softmax()
        softmax.train(ytrain, ytrain)

def test_softmax_train_reshape_input(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=300)
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])

    softmax = Softmax()
    loss = softmax.train(Xtrain, ytrain, reg=0, learning_rate=1e-6)
    assert loss[0] > loss[-1]

def test_softmax_train_reshape_input(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=4000)
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])

    softmax = Softmax()
    loss = softmax.train(Xtrain, ytrain, reg=0, learning_rate=1e-6, num_iters=2000)
    assert loss[0] > loss[-1]
    assert loss[-1] / loss[0] < 0.15

def test_softmax_random_weights(sample_train, weight_size=0.0001, regularization=1.0):
    Xtrain, ytrain = sample_train(count=7000)
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])

    W = np.random.randn(3073,10) * weight_size
    loss, grad = softmax_loss_naive(W, Xtrain, ytrain, regularization)
    assert loss > 1.8 and loss < 2.8
    #expect a loss of  loss ~ -np.log(0.1)
    # i.e. we get it right 1/10 times
