import numpy as np
from numpy.random import randn
from numpy import abs
import matplotlib.pyplot as plt


class RNN:

    def __init__(self, m=5, K=80, seq_len=25, eta=1e-1, sig=1e-2):
        # Hyper Param
        self.K = K # output dim
        self.seq_len = seq_len
        self.eta = eta
        self.m = m

        # biases
        self.b = np.zeros((m))
        self.c = np.zeros((K))

        # weights
        self.U = randn(m, K) * sig
        self.V = randn(K, m) * sig
        self.W = randn(m, m) * sig

        # initial
        self.H = np.ndarray([seq_len, m]) # hidden
        self.P = np.ndarray([seq_len, K]) # Store sequence output
        self.A = np.ndarray([seq_len, m]) # store all A output of sequence

        self.h0 = np.zeros((m))
        self.H[-1,:] - self.h0



    def train(self, Data):
        max_len = len(Data) - self.seq_len - 1 # -1 is reserved for label
        smoothed_loss = 0

        pass

    def predict(self, X):
        P = self.forward(X)
        dim = np.arange(0, self.K)

        sampled_idx = []
        for i, p in enumerate(P):
            char = np.random.choice(dim, p=p)
            sampled_idx.append(char)

        # To One Hot
        out = np.zeros((self.seq_len, self.K))
        rows = np.arange(0, self.seq_len)
        out[rows, sampled_idx] = 1
        return out

    def forward(self, X, param=None):
        '''

        :param x: seq_len x K matrix
        :return:
        '''
        for i, x in enumerate(X):
            a = self.W.dot(self.H[i-1]) + self.U.dot(x) + self.b
            self.A[i,:] = a

            h = np.tanh(a)
            self.H[i, :] = h

            o = self.V.dot(h) + self.c

            nom = np.exp(o)
            denom = np.sum(nom)

            p = nom / denom
            self.P[i,:] = p
        return self.P

    def backward(self, P, Y, X):

        dV = 0
        dW = 0
        dU = 0
        dc = 0
        db = 0
        da = np.zeros((self.m))
        for t, (p, y, x) in reversed(list(enumerate(zip(P, Y, X)))): # prop back in time
            dO = -(y-p)

            dc += dO # second bias
            dV += np.outer(dO, self.H[t])

            dh = dO.dot(self.V) + da.dot(self.W)  # \delta a_(t+1)
            db += dh # first bias

            da = (1-(np.tanh(self.A[t]) ** 2)) * dh
            dW += np.outer(da, self.H[t-1])

            dU += np.outer(da, x)
        # normalization stuff
        # dV /= self.seq_len
        # dW /= self.seq_len
        # dU /= self.seq_len
        # dc /= self.seq_len
        # db /= self.seq_len
        return dU, dV, dW, db, dc,

    def loss(self, X, Y, param=None):
        P = self.forward(X, param)
        prod = np.sum(P * Y, axis=1)
        out = -np.sum(np.log(prod)) # /self.seq_len
        return out

    def check_grad(self, X, Y):
        P = self.forward(X)
        dU, dV, dW, dB, dC = self.backward(P, Y, X)
        print('num_grad computing')
        numgrads = self.compute_num_grads_center(X,Y)
        nU, nV, nW, nB, nC = tuple(numgrads)


        ddB = dB - nB
        ddU = dU - nU
        diffU = dU - nU
        diffV = dV - nV
        diffW = dW - nW
        diffB = dB - nB
        diffC = dC - nC



        print('hello world')

    def compute_num_grads_center(self, X, Y, h=1e-5):
        """
        A somewhat slow method to numerically approximate the gradients using the central difference.
        :param X: Data batch. d x n
        :param Y: Labels batch. K x n
        :param h: Step length, default to 1e-5. Should obviously be kept small.
        :return: Approximate gradients
        """
        # df/dx ≈ (f(x + h) - f(x - h))/2h according to the central difference formula

        params = [self.U, self.V, self.W,  self.b, self.c]
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