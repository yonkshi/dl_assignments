import numpy as np
from scipy.sparse import csr_matrix
from numpy import tensordot, maximum, sum
from numpy.random import randn
from time import time
from typing import List
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
    def __init__(self, k=[5,3],nf=[20,20], K = 18, fsize=1337, batch_size = 100, learning_rate=0.001, momentum_coeff = 0.9):
        self.k = k # filter width
        self.nf = nf # number of filters
        self.fsize = (N_LEN - k[-1] + 1) * nf[-1]
        self.K = K # output dimensions
        self.sigs = [0.01, 0.01, 0.01] #TODO He initialization?
        self.batch_size = batch_size
        self.eps = 0.01
        self.etta = learning_rate
        self.rho = momentum_coeff
        self.precomputed_v1_dimension = D * k[0] * nf[0]



        # Speed comparison bottleneck

class ConvNet():
    def __init__(self, filter_size=2, hyperparam=HyperParam(), ):

        self.hp = hyperparam
        self.f = [None] * filter_size
        self.nlen = [N_LEN]
        self.filter_size = filter_size
        # initialize filters
        n1 = self.hp.nf[0]
        d = D
        for i, ki in enumerate(self.hp.k):
            ni = self.hp.nf[i]
            sig = self.hp.sigs[i]
            self.f[i] = randn(d, ki, ni) * sig
            self.nlen.append(self.nlen[i] - ki + 1)

            d = ni
        fsize = self.hp.nf[-1] * self.nlen[-1]
        self.w = randn(self.hp.K, fsize) * self.hp.sigs[-1]

        self.mf=[]
        self.dF = [0] * filter_size
        self.dF2 = 0
        self.dF1 = 0
        self.dW = 0

        # mommentum
        self.dW_momentum = 0
        self.dF_momentum = [0] * filter_size

    def compute_batch(self, X, Y, W=None):
        batchsize = self.hp.batch_size
        data_size = X.shape[1]
        print(self.compute_loss(X,Y))
        t0 = time()
        print('begin preprocessing ...')
        self.pre_process_mx(X, batchsize)
        print('preprocess time', time() - t0)
        t0 = time()
        for start in np.arange(0, data_size, batchsize):
            end = start + batchsize
            if end > data_size: end = data_size

            x_batch = X[:, start:end]
            y_batch = Y[:, start:end]
            i = int(start / batchsize)
            self.pre_mx1_batch = self.precomputed_mx1[i]

            p = self.forward(x_batch)
            dw, df = self.backward(y_batch, p)

            self.dW_momentum = self.dW_momentum * self.hp.rho + dw * self.hp.etta
            self.dF_momentum = [ f * self.hp.rho + df[i] for i,f in enumerate(self.dF_momentum)]

            self.w -= self.dW_momentum
            self.f = [ f - self.dF_momentum[i] for i, f in enumerate(self.f)]


        print('epoch time:', time()-t0)
        print(self.compute_loss(X, Y))

    def forward(self, x_input, params=None):
        if params is not None:
            weights, f1, f2 = params
            filters = [f1, f2]
            modify_self = False
        else:
            weights, filters = self.w, self.f
            modify_self = True #

        X = [x_input]
        n_len = N_LEN
        # Filters
        for f in filters:
            x = X[-1]
            mf = self.make_mf_matrix(f, n_len)
            s1 = mf.dot(x)
            s2 = np.maximum(s1, 0)
            n_len = int(mf.shape[0]/f.shape[2]) # (n_len-k+1) * nf / nf
            X.append(s2)
            if modify_self: self.mf.append(mf)

        # softmax
        s = weights.dot(X[-1])
        nom = np.exp(s)
        denom = np.sum(nom, axis=0)
        p = nom / denom
        if modify_self: self.x = X
        return p

    def backward(self, y, p):
        G = -(y-p)
        x = self.x[-1]
        n = G.shape[1]

        # Dense layer
        dW = G.dot(x.T) / n

        G_ = self.w.T.dot(G)
        ind = x > 0
        G_ = G_ * ind

        # conv layers
        i = len(self.dF)

        for dF in reversed(self.dF):
            i -= 1

            x = self.x[i]
            f = self.f[i]
            d, k, nf = f.shape

            # use precomputed value
            if i == 0:
                #mx = self.pre_mx1_batch
                v_vec = self._branch0(G_, n)
                v_vec2 = self._branch1(x, d, k, nf, G_, n)
            else:

                #mx = self.make_mx_matrix(x, d, k, nf)
                v_vec = self._branch1(x,d,k,nf, G_, n)
            #v_vec = np.einsum('ik,ijk->j', G_, mx) / n

            v = v_vec.reshape(f.shape)
            # bug fix
            if i == 0:
                v = v_vec.reshape(f.shape, order='F')

            self.dF[i] = v
            mf = self.mf[i]
            G_ = mf.T.dot(G_)
            G_ = G_ * (x > 0)

        return dW, self.dF

    def _branch0(self, G_, n):
        cols = self.pre_mx1_batch
        v_vec = np.zeros(self.hp.precomputed_v1_dimension)
        for col_ix, gix in cols:
            to_be_summed = G_.take(gix)
            v_vec[col_ix] = np.sum(to_be_summed)
        return v_vec / n

    def _branch1(self, x, d, k, nf, G_, n):
        #mx = self.make_mx_matrix(x, d, k, nf)
        #v_vec = np.einsum('ik,ijk->j', G_, mx) / n

        # Optmized version
        mx = self.make_mx_matrix(x, d, k, nf, optimized=True)
        new_g = G_.reshape(nf,mx.shape[0],-1, order='F') # nf should actually be cols, but einsum eliminates the need
        v_vec = np.einsum('ijk,lik->jl', mx, new_g).flatten() / n

        return v_vec

    def pre_process_mx(self,X, batch_size):
        d, k, nf = self.f[0].shape
        precomputed_mx1, debug = self.make_mx_matrix_sparse(X, d, k, nf)
        #mx_rows, mx_cols, _ = precomputed_mx1.shape

        # TODO REMOVE ME: DEBUG ONLY
        old_mx = self.make_mx_matrix(X, d, k, nf)
        # change axis so that it's batch x dd x output, and sorted in that order
        swapped = np.swapaxes(old_mx, 2,1)
        swapped = np.swapaxes(swapped, 0,1)
        old_where = np.argwhere(swapped > 0)

        diff = debug - old_where

        print('hello world')


        # swapped to: mx_cols(output dim) x mx_rows x data_size
        swapped = np.swapaxes(precomputed_mx1, 1, 0)
        data_size = X.shape[1]

        self.precomputed_mx1 = []
        print('begin loop')
        t1 = time()
        # precompute to save time on mod
        for b_start in np.arange(0, data_size, batch_size):
            print(b_start/batch_size, time()-t1)
            b_end = b_start+batch_size if b_start+batch_size < data_size else data_size
            batch = swapped[:,:, b_start:b_end]

            # second loop actually speeds things up a bit
            cols_in_batch = []
            for j in range(mx_cols):
                ix_2d = np.argwhere(batch[j,:,:] > 0)
                if ix_2d.size == 0:  #Empty column
                        continue

                gix = ix_2d[:, 0] * batch_size + ix_2d[:, 1] # g_row * batch_size + batch number
                cols_in_batch.append((j, gix))

            self.precomputed_mx1.append(cols_in_batch)

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

    def make_mx_matrix(self, x_input, d, k, nf, optimized=False):
        x_input_ = x_input
        nlen = int(x_input_.shape[0] / d)

        non_zeros_per_mx_row = self.hp.k[0]

        vec_x_cols = d * k
        x_rows = (nlen - k + 1)
        rows = x_rows * nf
        cols = vec_x_cols * nf

        batch_size = x_input.shape[1]

        x_vecs = np.ndarray((x_rows, vec_x_cols, batch_size))

        # stride x to list
        for i in range(x_rows):
            idx = i * d
            x_vecs[i, :] = x_input_[idx:idx + vec_x_cols]

        if optimized:
            return x_vecs

        mx = np.zeros((rows, cols, batch_size))

        # add in into large matrix
        for i in range(rows):
            idx = (i % nf) * vec_x_cols
            r = int(i / nf)
            deleme = x_vecs[r, :]
            mx[i, idx:idx + vec_x_cols] = deleme

        return mx

    def preprocess_mx_superefficient(self, x_input, batch_size):

        d, k, nf = self.f[0].shape

        nonzeros = np.argwhere(x_input.T > 0 )

        # split up into single word, assuming each
        unique, split_ix = np.unique(nonzeros[:,0],return_index=True)
        words = np.split(nonzeros[:,1], split_ix[1:]) # fixed a bug

        for batch_ix, word in enumerate(words):
            # create a list of x sub vectors for each word
            nlen = int(x_input.shape[0] / d)
            vec_x_cols = d * k
            x_rows = (nlen - k + 1)

            # Stride x to list
            # Words have consecutive character with nlen each, terminate at the last character,
            # also assume strides are 1, but easy change (modify i) to accomondate different strides
            debug_row_in_mx = 0
            mx = []
            for i in range(x_rows):
                if i >= word.size:
                    break # end of word
                end = i + k if i + k < word.size else word.size
                x_vec = word[i:end] - d * i # assume 1 stride (aka 1 character move)

                # loop to an additional
                for i in range(nf):
                    offset = i * vec_x_cols
                    mx_row = x_vec + offset
                    mx.append(mx_row)

            # TODO optmize for cols out dim
        return mx,

    def compute_num_grads_center(self, X, Y, h=1e-5):
        """
        A somewhat slow method to numerically approximate the gradients using the central difference.
        :param X: Data batch. d x n
        :param Y: Labels batch. K x n
        :param h: Step length, default to 1e-5. Should obviously be kept small.
        :return: Approximate gradients
        """
        # df/dx ≈ (f(x + h) - f(x - h))/2h according to the central difference formula

        params = [np.copy(self.w), np.copy(self.f[0]), np.copy(self.f[1])]
        num_grads = []

        for i, param in enumerate(params):

            grad = np.zeros(param.shape)
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                old_value = param[ix]
                param[ix] = old_value + h
                plus_cost = self.compute_loss(X, Y, params)
                param[ix] = old_value - h
                minus_cost = self.compute_loss(X, Y, params)
                param[ix] = old_value  # Restore original value

                grad[ix] = (plus_cost - minus_cost) / (2 * h)
                it.iternext()  # go to next index

            if i > 0 and param.shape[0] == param.shape[2]:
                grad = grad.transpose(2, 1, 0)  # to make sure the 3D arrays come the same way as in backward function
            num_grads.append(grad)

        return num_grads

    def compute_loss(self, X, Y, params=None):
        p = self.forward(X, params=params)
        batch_count = Y.shape[1]
        # faster sum
        prod = sum(Y * p, axis=0)
        # prod_alt = np.einsum('ij, ij->i', Y, p.T)
        loss = - np.log(prod)
        summed = np.sum(loss)
        loss_normalized = summed / batch_count
        return loss_normalized

    def check_grad(self, a, n, eps):
        diff = np.abs(a - n) / np.maximum(eps, np.amax(np.abs(a) + np.abs(n)))
        if np.amax(diff) < 1e-6:
            return True
        else:
            return False

