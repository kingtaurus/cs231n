
__all__ = (
    'DIGITS',
    'LETTERS',
    'CHARS',
    'sigmoid',
    'softmax',
)

import numpy

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS

def softmax(a):
    exps = numpy.exp(a.astype(numpy.float64))
    return exps / numpy.sum(exps, axis=-1)[:, numpy.newaxis]

def sigmoid(a):
    return 1. / (1. + numpy.exp(-a))
