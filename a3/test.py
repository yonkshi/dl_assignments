import scipy.sparse as sp
import numpy as np
from numpy.random import multivariate_normal
from scipy.io import loadmat
'''
d: dimension of char space (28 types of chars)
nlen: dimension of char (max number of char per word)


n1: number of filters at layer one
nf: generic number of filters
dd: filter height
k: width of filter, not related to 
'''


def make_mx_matrix(x_input, d, k, nf):
    nlen = int(x_input.shape[0] / d)
    X_input = x_input.reshape(d, nlen)

    vec_x_cols = d * k
    x_rows = (nlen - k + 1)
    rows = x_rows * nf
    cols = vec_x_cols * nf

    x_vecs = np.ndarray((x_rows, vec_x_cols))
    for i in range(x_rows):
        idx = i * d
        x_vecs[i, :] = x_input[idx:idx + vec_x_cols]

    mx = np.zeros((rows, cols))

    for i in range(rows):
        idx = (i % nf) * vec_x_cols
        r = int(i / nf)
        mx[i, idx:idx+vec_x_cols] = x_vecs[r,:]

    return mx.astype('int')


x = np.arange(1,21,1).reshape((5,4)).flatten()

d = 4
k = 2
nf = 3

xx = make_mx_matrix(x, d, k, nf)
print('hello_world')