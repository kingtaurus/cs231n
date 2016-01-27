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

from cs231n.classifiers.linear_classifier import LinearSVM
from cs231n.classifiers.linear_svm        import svm_loss_naive, svm_loss_vectorized
from cs231n.gradient_check                import grad_check_sparse, eval_numerical_gradient
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

def test_SVM_loss_naive_no_bias(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=20)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))

    W = np.random.randn(3073,10) * 0.0001
    with pytest.raises(ValueError):
        loss, grad = svm_loss_naive(W, Xtrain, ytrain, 1e2)
    #this will fail because W is larger by 1

def test_SVM_loss_vectorized_no_bias(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=20)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))

    W = np.random.randn(3073,10) * 0.0001
    with pytest.raises(ValueError):
        loss, grad = svm_loss_vectorized(W, Xtrain, ytrain, 1e2)
    #this will fail because W is larger by 1

@pytest.mark.parametrize("train_count", [10 * x + 1 for x in range(1,10)])
def test_SVM_loss_naive_vectorized_comparison(sample_train, train_count):
    Xtrain, ytrain = sample_train(count=train_count)
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])

    W = np.random.randn(Xtrain.shape[1],10) * 0.0001
    loss, grad = svm_loss_vectorized(W, Xtrain, ytrain, 1e2)
    loss_naive, grad_naive = svm_loss_naive(W, Xtrain, ytrain, 1e2)
    assert np.abs(loss - loss_naive) < 0.0001
    assert np.linalg.norm(grad - grad_naive) < 0.0001

@pytest.mark.parametrize("train_count", [10 * x + 1 for x in range(1,10)])
@pytest.mark.parametrize("reg", [10. * x + 1 for x in range(10,100000, 10000)])
def test_SVM_loss_naive_vectorized_comparison_reg(sample_train, train_count, reg):
    Xtrain, ytrain = sample_train(count=train_count)
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])

    W = np.random.randn(Xtrain.shape[1],10) * 0.0001
    loss, grad = svm_loss_vectorized(W, Xtrain, ytrain, reg)
    loss_naive, grad_naive = svm_loss_naive(W, Xtrain, ytrain, reg)
    assert np.abs(loss - loss_naive) < 0.0001
    assert np.linalg.norm(grad - grad_naive) < 0.0001


@pytest.mark.parametrize("train_count", [10 * x + 1 for x in range(1,10)])
@pytest.mark.parametrize("reg", [10. * x + 1 for x in range(10,100000, 10000)])
def test_SVM_loss_vectorized_comparison_mean(sample_train, train_count, reg):
    Xtrain, ytrain = sample_train(count=train_count)
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))

    mean_image = np.mean(Xtrain, axis=0)
    Xtrain_mean_removed = Xtrain - mean_image

    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])
    Xtrain_mean_removed = np.hstack([Xtrain_mean_removed, np.ones((Xtrain_mean_removed.shape[0], 1))])

    W = np.random.randn(Xtrain.shape[1],10) * 0.0001
    loss, grad = svm_loss_vectorized(W, Xtrain, ytrain, reg)
    loss_mean_removed, grad_mean_removed = svm_loss_vectorized(W, Xtrain_mean_removed, ytrain, reg)
    #assert np.abs(loss - loss_mean_removed) > 0.01
    assert np.linalg.norm(grad - grad_mean_removed) > 1.0


@pytest.mark.parametrize("check_count", list(range(10,15)))
def test_SVM_grad_vectorized_comparison_sparse(sample_train, check_count):
    Xtrain, ytrain = sample_train(count=500)
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))

    mean_image = np.mean(Xtrain, axis=0)
    Xtrain -= mean_image

    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])

    W = np.random.randn(Xtrain.shape[1],10) * 0.0001
    loss, grad = svm_loss_vectorized(W, Xtrain, ytrain, 0.)

    f = lambda w: svm_loss_vectorized(w, Xtrain, ytrain, 0.0)[0]
    g = lambda w: svm_loss_vectorized(w, Xtrain, ytrain, 0.0)[1]
    #(f(W+vec(h)) - f(W-vec(h)))/2/|vec(h)| = approximately dot(f'(W),vec(h)) * vec(h)
    #grad(loss) = grad vectorized

    num_checks = check_count
    grad_analytic = g(W)
    for i in range(num_checks):
        ix = tuple([random.randrange(m) for m in W.shape])
        shift = np.zeros(W.shape)
        shift[ix] = 1e-5
        grad_numerical = (f(W + shift) - f(W - shift)) / (2 * 1e-5)
        assert( abs(grad_numerical - grad_analytic[ix]) / (abs(grad_numerical) + abs(grad_analytic[ix])) < 0.01)
#this needs to be reworked -> want to check that changing the regularization parameter
#causes the loss and gradient to be different
# @pytest.mark.parametrize("reg", [10. * x + 1 for x in range(10,100000, 10000)])
# @pytest.mark.parametrize("reg2", [10. * x + 1 for x in range(10,100000, 10000)])
# def test_SVM_loss_naive_vectorized_comparison_reg2(sample_train, reg, reg2):
#     Xtrain, ytrain = sample_train(count=400)
#     Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
#     Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])

#     W = np.random.randn(Xtrain.shape[1],10) * 0.0001
#     loss, grad = svm_loss_vectorized(W, Xtrain, ytrain, reg)
#     loss_naive, grad_naive = svm_loss_naive(W, Xtrain, ytrain, reg2)

#     if np.abs(reg - reg2) < 0.001:
#         assert np.abs(loss - loss_naive) < 0.0001
#         assert np.linalg.norm(grad - grad_naive) < 0.0001
#     else:
#         assert np.abs(loss - loss_naive) > 0.0001
#         assert np.linalg.norm(grad - grad_naive) > 0.0001


def test_SVM_train(sample_train, sample_test):
    #this test is designed to verify that input shapes are correct
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    with pytest.raises(ValueError):
        lin_svm = LinearSVM()
        lin_svm.train(Xtrain, ytrain)

def test_SVM_train_1(sample_train, sample_test):
    #this test is designed to verify that input shapes are correct
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))

    with pytest.raises(ValueError):
        lin_svm = LinearSVM()
        lin_svm.train(Xtrain, Xtrain)

def test_SVM_train_2(sample_train, sample_test):
    #this test is designed to verify that input shapes are correct
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))

    with pytest.raises(ValueError):
        #this will catcha Valueerror associated with bad unpacking of a tuple
        lin_svm = LinearSVM()
        lin_svm.train(ytrain, ytrain)

def test_SVM_train_reshape_input(sample_train, sample_test):
    Xtrain, ytrain = sample_train(count=40)
    Xtest, ytest   = sample_test(count=10)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

    lin_svm = LinearSVM()
    lin_svm.train(Xtrain,ytrain)


# @pytest.mark.parametrize("train_count", [10 * x + 1 for x in range(1,10)])
# @pytest.mark.parametrize("test_count", [10 * x + 1 for x in range(1,11)])
# def test_SVM_predict_shape(sample_train, sample_test, train_count, test_count):
#     Xtrain, ytrain = sample_train(train_count)
#     Xtest, ytest   = sample_test(test_count)

#     Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
#     Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

#     lin_svm = LinearSVM()
#     lin_svm.train(Xtrain,ytrain)
#     assert 1

# @pytest.mark.parametrize("train_count", [10 * x + 1 for x in range(1,10)])
# @pytest.mark.parametrize("test_count", [10 * x + 1 for x in range(1,11)])
# def test_SVM_weight_shape(sample_train, sample_test, train_count, test_count):
#     assert 1

# @pytest.mark.parametrize("train_count, test_count", [(10 * x + 1, 5 * x + 1) for x in range(1,10)])
### if you have an equal number of values
#@pytest.mark.parametrize("train_count", [10 * x + 1 for x in range(1,10)])
#@pytest.mark.parametrize("test_count", [10 * x + 1 for x in range(1,11)])
# def test_KNN_dists_noloop_shape(sample_train, sample_test, train_count, test_count):
#
# This is nice because it tests over a bunch of values
#
# @pytest.mark.parametrize("in_count", [150 * x + 1 for x in range(1,10)])
# def test_KNN_dists_noloop_shape(sample_train, sample_test, in_count):
#     Xtrain, ytrain = sample_train(count=in_count)
#     Xtest, ytest   = sample_test(count=in_count-30)

#     Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
#     Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

#     knn = KNearestNeighbor()
#     knn.train(Xtrain,ytrain)
#     assert knn.compute_distances_no_loops(Xtest).shape == (Xtest.shape[0], Xtrain.shape[0])

# def test_KNN_dists_one_to_none(sample_train, sample_test):
#     Xtrain, ytrain = sample_train(count=40)
#     Xtest, ytest   = sample_test(count=10)

#     Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
#     Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

#     knn = KNearestNeighbor()
#     knn.train(Xtrain,ytrain)
#     dist_one = knn.compute_distances_one_loop(Xtest)
#     dist_no  = knn.compute_distances_no_loops(Xtest)
#     assert np.linalg.norm(dist_one - dist_no, ord='fro') < 0.001

# def test_KNN_predict_labels_shape(sample_train, sample_test):
#     Xtrain, ytrain = sample_train(count=40)
#     Xtest, ytest   = sample_test(count=10)

#     Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
#     Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

#     knn = KNearestNeighbor()
#     knn.train(Xtrain,ytrain)

#     dist_no  = knn.compute_distances_no_loops(Xtest)
#     assert knn.predict_labels(dist_no, k=1).shape == ytest.shape
#     assert knn.predict_labels(dist_no, k=2).shape == ytest.shape
#     assert knn.predict_labels(dist_no, k=3).shape == ytest.shape
#     assert knn.predict_labels(dist_no, k=4).shape == ytest.shape

# @pytest.mark.parametrize("k", list(range(-1,-10,-1)))
# def test_KNN_predict_k_outofrange_parameter(sample_train, sample_test, k):
#     Xtrain, ytrain = sample_train(count=40)
#     Xtest,  ytest   = sample_test(count=10)

#     Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
#     Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

#     knn = KNearestNeighbor()
#     knn.train(Xtrain,ytrain)

#     with pytest.raises(ValueError):
#         knn.predict(Xtest,k)

# @pytest.mark.parametrize("k", list(range(1,3)))
# def test_KNN_predict_k_parameter(sample_train, sample_test, k):
#     Xtrain, ytrain = sample_train(count=40)
#     Xtest,  ytest   = sample_test(count=10)

#     Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
#     Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

#     knn = KNearestNeighbor()
#     knn.train(Xtrain,ytrain)

#     assert knn.predict(Xtest,k).shape == ytest.shape

# @pytest.mark.parametrize("num_loops", list(range(-1,-10,-1)) + list(range(3,10,1)))
# def test_KNN_predict_num_loop_parameter(sample_train, sample_test, num_loops):
#     Xtrain, ytrain = sample_train(count=40)
#     Xtest,  ytest   = sample_test(count=10)

#     Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
#     Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

#     knn = KNearestNeighbor()
#     knn.train(Xtrain,ytrain)

#     with pytest.raises(ValueError):
#         knn.predict(Xtest,0,num_loops).shape

# @pytest.mark.parametrize("num_loops", list(range(0,3)))
# @pytest.mark.parametrize("k", [1])
# def test_KNN_predict_loop_parameter(sample_train, sample_test, k, num_loops):
#     Xtrain, ytrain = sample_train(count=40)
#     Xtest,  ytest   = sample_test(count=10)

#     Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
#     Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

#     knn = KNearestNeighbor()
#     knn.train(Xtrain,ytrain)

#     assert knn.predict(Xtest,k,num_loops).shape == ytest.shape

# def test_KNN_predict_incorrect_shape(sample_train, sample_test):
#     Xtrain, ytrain = sample_train(count=500)
#     Xtest, ytest   = sample_test(count=125)

#     Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], -1))
#     Xtest = np.reshape(Xtest, (Xtest.shape[0], -1))

#     knn = KNearestNeighbor()
#     knn.train(Xtrain,ytrain)

#     with pytest.raises(ValueError):
#         knn.predict(ytrain)#using ytrain, shich has incorrect dimensions;

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
@pytest.fixture(params=["mysql","pg"],scope='function')
def db(request):
    def fin():
        print(" teardown")
    request.addfinalizer(fin)
    return request.param

#requires
def test_db(db):
    assert len(db) > 1














