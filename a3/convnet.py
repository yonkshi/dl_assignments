import numpy as np
from numpy import sum
from numpy.random import normal
import matplotlib.pyplot as plt
from time import time


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
    def __init__(self, k=[3,4],nf=[5,6], K = 18, fsize=1337, batch_size = 100, learning_rate=0.001, momentum_coeff = 0.9):
        self.k = k # filter width
        self.nf = nf # number of filters
        self.fsize = (N_LEN - k[-1] + 1) * nf[-1]
        self.K = K # output dimensions
        self.batch_size = batch_size
        self.eps = 0.01
        self.etta = learning_rate
        self.rho = momentum_coeff
        self.precomputed_v1_dimension = D * k[0] * nf[0]

        # compute nlen
        self.nlen = [N_LEN]
        for i in range(len(nf)): self.nlen += [self.nlen[-1] - k[i] + 1]

        # He initialization
        # Input layer, hidden conv layer, and dense layer
        sig = np.array([k[0], self.nf[0] * k[1], self.nlen[-1] * nf[-1]])
        self.sigma = np.sqrt(2/sig)

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
            #self.f[i] = normal(0, self.hp.sigma[i], (d, ki, ni))
            self.f[i] = np.random.randn(d, ki, ni) * 0.1
            d = ni

        fsize = self.hp.nf[-1] * self.hp.nlen[-1]
        #self.w = normal(0, self.hp.sigma[-1],(self.hp.K, fsize))
        self.w = np.random.randn(self.hp.K, fsize) * 0.1
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
        t0 = time()
        print('begin preprocessing ...')
        self.preprocess_mx_superefficient(X, batchsize)
        print('preprocess time', time() - t0)
        t0 = time()
        for start in np.arange(0, data_size, batchsize):
            self.mf = []
            end = start + batchsize
            if end > data_size: end = data_size

            x_batch = X[:, start:end]
            y_batch = Y[:, start:end]
            i = int(start / batchsize)
            self.mx1_batch = self.precomputed_mx1_full[i]

            p = self.forward(x_batch)
            dw, df = self.backward(y_batch, p)

            self.dW_momentum = self.dW_momentum * self.hp.rho + dw * self.hp.etta
            self.dF_momentum = [ f * self.hp.rho + df[i] for i,f in enumerate(self.dF_momentum)]

            self.w -= self.dW_momentum
            self.f = [ f - self.dF_momentum[i] for i, f in enumerate(self.f)]

            #print('loss',self.compute_loss(X, Y))

            print('loss',self.loss(x_batch, y_batch, p=p))
            print('accuracy', '%0.2f%%' % self.accuracy(x_batch, y_batch, p=p))

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


            if i == 0:
                # use precomputed value
                v_vec = np.zeros(self.hp.precomputed_v1_dimension)
                for col_ix, gix in self.mx1_batch.items():
                    to_be_summed = G_.take(gix)
                    v_vec[col_ix] = np.sum(to_be_summed)
                v_vec /= n
            else:
                mx = self.make_mx_matrix(x, d, k, nf, optimized=True)
                new_g = G_.reshape(mx.shape[0], nf, -1)  # nf should actually be cols, but einsum eliminates the need
                v_vec = np.einsum('ijk,iyk->jy', mx, new_g) / n

            v = v_vec.reshape(f.shape, order='F')

            self.dF[i] = v
            mf = self.mf[i]
            G_ = mf.T.dot(G_)
            G_ = G_ * (x > 0)

        return dW, self.dF

    def preprocess_mx_superefficient(self, x_input, batch_size):

        d, k, nf = self.f[0].shape

        nonzeros = np.argwhere(x_input.T > 0 )

        # split up into single word, assuming each
        unique, split_ix = np.unique(nonzeros[:,0],return_index=True)
        words = np.split(nonzeros[:,1], split_ix[1:]) # fixed a bug

        # used to pre-preprocess batch index, tiny speed up
        batch_indices = unique % batch_size
        batches = []
        cols_idx = {}
        for word_ix, word in enumerate(words):

            batch_ix = batch_indices[word_ix]
            if batch_ix == 0: # each batch
                cols_idx = {}
                batches.append(cols_idx)


            # create a list of x sub vectors for each word
            nlen = int(x_input.shape[0] / d)
            vec_x_cols = d * k
            x_rows = (nlen - k + 1)

            # Stride x to list
            # Words have consecutive character with nlen each, terminate at the last character,
            # also assume strides are 1, but easy change (modify i) to accomondate different strides
            debug_row_in_mx = 0
            mx = []
            mx_row_i = 0
            for i in range(x_rows):
                if i >= word.size:
                    break # end of word
                end = i + k if i + k < word.size else word.size
                x_vec = word[i:end] - d * i # assume 1 stride (aka 1 character move)

                # loop to an additional
                for j in range(nf):
                    offset = j * vec_x_cols
                    mx_row = x_vec + offset
                    mx.append(mx_row)

                    for out_col in mx_row:
                        if out_col not in cols_idx:
                            cols_idx[out_col] = []
                        gix = mx_row_i * batch_size + batch_ix  # g_row * batch_size + batch number
                        cols_idx[out_col].append(gix)
                    mx_row_i += 1

        self.precomputed_mx1_full = batches

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
        nlen = int(x_input.shape[0] / d)

        vec_x_cols = d * k
        x_rows = (nlen - k + 1)
        rows = x_rows * nf
        cols = vec_x_cols * nf

        batch_size = x_input.shape[1]

        x_vecs = np.ndarray((x_rows, vec_x_cols, batch_size))

        # stride x to list
        for i in range(x_rows):
            idx = i * d
            x_vecs[i, :] = x_input[idx:idx + vec_x_cols]

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
                plus_cost = self.loss(X, Y, params)
                param[ix] = old_value - h
                minus_cost = self.loss(X, Y, params)
                param[ix] = old_value  # Restore original value

                grad[ix] = (plus_cost - minus_cost) / (2 * h)
                it.iternext()  # go to next index

            num_grads.append(grad)

        return num_grads

    def loss(self, X, Y, p=None, params=None):
        if p is None:
            p = self.forward(X, params=params)
        batch_count = Y.shape[1]
        # faster sum
        prod = sum(Y * p, axis=0)
        # prod_alt = np.einsum('ij, ij->i', Y, p.T)
        loss = - np.log(prod)
        summed = np.sum(loss)
        loss_normalized = summed / batch_count
        return loss_normalized

    def accuracy(self, X, Y, p=None, params=None):
        if p is None:
            p = self.forward(X, params=params)
        winners = np.argmax(p, axis=0)  # 2D vector
        truth = np.argmax(Y, axis=0)
        diff = truth - winners

        s1 = np.count_nonzero(diff)
        s2 = Y.shape[1]
        s3 = s1/s2
        accuracy = 1 -  s3
        return accuracy * 100

    def check_grad(self, a, n, eps):
        print('begin numerical grad')
        ndw, ndf1, ndf2 = self.compute_num_grads_center(x_batch, y_batch)
        df1, df2 = df
        diffw = dw-ndw
        diff1 = ndf1 - df1
        diff2 = ndf2 - df2
        diff1_max = np.max(np.abs(diff1))
        snd2 = np.abs(ndf1).sum()
        sd2 = np.abs(df1).sum()
        print('F1 cumulative diff', snd2 - sd2)
        print(diff1.sum())
        z = df2
        zz =  ndf2
        print('done numerical grad')

    def _plot_loss(self, loss, validation_loss=None):
        plt.figure()
        plt.plot(loss)
        plt.title('training loss')
        plt.xlabel('update cycle')
        plt.show()

    def _plot_accuracy(self, accuracy, validation_accuracy=None):
        plt.figure()
        plt.plot(accuracy)
        plt.title('prediction accuracy')
        plt.xlabel('update cycle')
        plt.show()


