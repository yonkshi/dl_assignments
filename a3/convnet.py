import numpy as np
from numpy.random import randn
from scipy.io import loadmat

NAMES = 'ascii_names.txt'
D = 28 # Number of chars
N_LEN = 19 # Maximum char for longest word

X = None
'''
d: dimension of char space (28 types of chars)
nlen: dimension of char (max number of char per word)


n1: number of filters at layer one
nf: generic number of filters
dd: filter height
k: width of filter, not related to 
'''

class HyperParam:
    def __init__(self, k=[5,5],n=[5,5], K = 18, fsize=1337):
        self.k = k
        self.n = n # default filter size
        self.fsize = fsize
        self.K = K # output dimensions
        self.sigs = [0.01, 0.01, 0.01] #TODO He initialization?

class ConvNet():
    def __init__(self, filter_size=2, hyperparam=HyperParam(), ):

        self.hp = hyperparam
        self.f = [None] * filter_size
        # initialize filters
        n1 = self.hp.n[0]
        d = D
        for i, ki in enumerate(self.hp.k):
            ni = self.hp.n[i]
            sig = self.hp.sigs[i]
            self.f[i] = randn(ni, ki, d) * sig
            d = ni

        self.w = randn(self.hp.K, self.hp.fsize) * self.hp.sigs[-1]

    def make_mf_matrix(self, F, nlen):
        dd, k, nf = F.shape
        flen = k * dd
        rows = (nlen - k + 1) * nf
        columns = nlen * dd
        F = F.reshape((flen, nf), order='F')

        mf = np.zeros((rows, columns))

        for i in range(rows):
            idx = int(i / nf) * dd
            mf[i, idx:idx + flen] = F[:, i % nf]
        return mf

    def make_mx_matrix(self, x_input, d, k, nf):
        x_input_ = x_input.flatten()
        nlen = int(x_input_.shape[0] / d)

        vec_x_cols = d * k
        x_rows = (nlen - k + 1)
        rows = x_rows * nf
        cols = vec_x_cols * nf

        x_vecs = np.ndarray((x_rows, vec_x_cols))
        for i in range(x_rows):
            idx = i * d
            x_vecs[i, :] = x_input_[idx:idx + vec_x_cols]

        mx = np.zeros((rows, cols))

        for i in range(rows):
            idx = (i % nf) * vec_x_cols
            r = int(i / nf)
            mx[i, idx:idx + vec_x_cols] = x_vecs[r, :]

        return mx.astype('int')

    def hey(self):
        print('yo')
