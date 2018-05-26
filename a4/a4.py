from scipy.io import loadmat, whosmat
import numpy as np
import matplotlib.pyplot as plt
import string


char_list = [] # For decoding
char_dict = [] # For encoding
seq_len = 24
num_epochs = 10
def main():
    global char_list, char_dict
    lines, char_dict, char_list = get_text()

    start = 10

    data = encode(lines)

    rnn = RNN(K=len(char_list), seq_len=seq_len)
    rnn.train(data)

import numpy as np
from numpy.random import randn
from numpy import abs
import matplotlib.pyplot as plt
from time import time


class RNN:

    def __init__(self, m=5, K=len(char_list), seq_len=seq_len, eta=1e-1, sig=1e-2):
        # Hyper Param
        self.K = K # output dim
        self.seq_len = seq_len
        self.eta = eta
        self.m = m

        # params
        self.b = np.zeros((m))
        self.c = np.zeros((K))

        self.U = randn(m, K) * sig
        self.V = randn(K, m) * sig
        self.W = randn(m, m) * sig

        # initial
        self.H = np.ndarray([seq_len, m]) # hidden
        self.P = np.ndarray([seq_len, K]) # Store sequence output
        self.A = np.ndarray([seq_len, m]) # store all A output of sequence

        self.h0 = np.zeros((m))
        self.H[-1,:] = self.h0

        # values for adagrad
        self.mU = 0
        self.mV = 0
        self.mW = 0
        self.mb = 0
        self.mc = 0

        print('Hyper Params:\nm:{} k:{}, seq_len:{}, eta:{}, init_sigma:{} '.format(m, K, seq_len, eta, sig))

    def train(self, Data):
        seq_len = self.seq_len
        max_len = len(Data) - seq_len - 1 # -1 is reserved for label
        print('updates per epoch', max_len)
        smoothed_loss = 0
        smoothed_looses = [] # for plotting
        np.random.seed(130)
        for ep in range(num_epochs):
            print('\n\n\n\n=== Epoch %d===' % ep)
            t0 = time()
            # reset h prev
            self.H[-1, :] = np.zeros(self.m)
            for i in range(100_000):
                e = i * seq_len
                if e > len(Data) - seq_len - 1:
                    break

                x = Data[e:e+seq_len,...]
                y = Data[e+1:e + seq_len +1 , ...]
                p = self.forward(x)
                grads = self.backward(p, y, x)

                # Adagrad
                dU, dV, dW, db, dc = grads
                adaU, self.mU = self.ada(dU, self.mU)
                self.U -= adaU
                adaV, self.mV = self.ada(dV, self.mV)
                self.V -= adaV
                adaW, self.mW = self.ada(dW, self.mW)
                self.W -= adaW
                adab, self.mb = self.ada(db, self.mb)
                self.b -= adab
                adac, self.mc = self.ada(dc, self.mc)
                self.c -= adac

                loss = self.loss(x, y, P=p)
                smoothed_loss = smoothed_loss * 0.999 + 0.001 * loss

                if i % 100 == 0:
                    smoothed_looses.append(smoothed_loss)
                if i % 1000 == 0:
                    step = i + ep * int(max_len/seq_len)
                    print(step, ':', smoothed_loss)
                    self.check_grad(x, y)
                if i % 5000 == 0:
                    print('ada coefficients:')
                    #mU = np.mean(1/np.sqrt(self.mU))
                    print('{:.2e}'.format(self._ada_log(self.mU)))
                    print('{:.2e}'.format(self._ada_log(self.mV)))
                    print('{:.2e}'.format(self._ada_log(self.mW)))
                    print('{:.2e}'.format(self._ada_log(self.mb)))
                    print('{:.2e}'.format(self._ada_log(self.mc)))
                    idx = np.random.randint(0, max_len-250)
                    for j in range(10):
                        start = idx + seq_len * j
                        end = start + seq_len
                        rand_data = Data[start:end,...]
                        predicted = self.predict(rand_data)
                        print(decode(predicted), end='')
                    print('')
            print('epoch time:', time() - t0)

        self.plot_loss(smoothed_looses)

    def ada(self, grads, m_prev):
        # clip grads
        grads = np.minimum(np.maximum(grads, -5),5)
        etta = self.eta
        m = m_prev + np.square(grads)
        delta = grads * etta / np.sqrt(m + (1e-8))
        return delta, m

    def predict(self, X, P=None):
        if P is None:
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
            #a = self.W.dot(self.H[i-1]) + self.U.dot(x) + self.b
            a = eindot(self.W, self.H[i-1])  + eindot(self.U, x) + self.b
            self.A[i,:] = a

            h = np.tanh(a)
            self.H[i, :] = h

            #o = self.V.dot(h) + self.c
            o = eindot(self.V, h) + self.c

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

        #dc = np.sum(-(P-Y), axis=0)

        for t, (p, y, x) in reversed(list(enumerate(zip(P, Y, X)))): # prop back in time
            dO = -(y-p)

            dc += dO # second bias
            #dV += np.outer(dO, self.H[t])
            dV += einouter(dO, self.H[t])

            #dh = dO.dot(self.V) + da.dot(self.W)  # dot
            dh = eindot(dO, self.V) + eindot(da, self.W)

            da = (1-(np.tanh(self.A[t]) ** 2)) * dh

            db += da # first bias
            #dW += np.outer(da, self.H[t - 1])
            dW += einouter(da, self.H[t-1]) # outer

            #dU += np.outer(da, x)
            dU += einouter(da, x) # outer
        # normalization stuff
        # dV /= self.seq_len
        # dW /= self.seq_len
        # dU /= self.seq_len
        # dc /= self.seq_len
        # db /= self.seq_len
        return dU, dV, dW, db, dc,

    def loss(self, X, Y, param=None, P=None):
        if P is None:
            P = self.forward(X, param)
        prod = np.sum(P * Y, axis=1)
        out = -np.sum(np.log(prod)) # /self.seq_len
        return out

    def check_grad(self, X, Y):
        P = self.forward(X)
        grads = self.backward(P, Y, X)
        print('num_grad computing')
        numgrads = self.compute_num_grads_center(X,Y)
        num_grads = tuple(numgrads)

        grads_names = ['dU', 'dV', 'dW', 'db', 'dc']
        print('gradient differences')
        for grad, n_grad, grad_name in zip(grads, num_grads, grads_names):
            diff = np.sum(abs(grad - n_grad))
            print('{}: {:.2e}'.format(grad_name, diff))
        print('\n')
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

    def _ada_log(self, ada):
        ada = np.sqrt(ada + (1e-8))
        ada = 1/ ada
        ada = np.abs(ada)
        return np.mean(ada)

    def plot_loss(self, loss_set):
        plt.plot(loss_set)
        name = 'Hairy Potter Adagrad loss filtered words'
        plt.title(name)
        plt.xlabel('epochs (x100)')
        plt.savefig(name)
        plt.show()
        pass

def get_text(max=10000000):
    """
    Get the 'ascii_names.txt' file
    :return:
    """

    with open('data/goblet_book.txt','r', encoding='UTF-8') as f:
        content = f.read()

    # filter
    #content = content.lower()
    punc = string.punctuation
    punc = punc.replace('.','') # don't filter , and .
    punc = punc.replace(',','')
    punc = punc.replace('"', '')
    translator = str.maketrans('ü', 'u', punc + '•' + '\t')
    content = content.translate(translator)

    vacab_list = list(set(content))
    vocab_dict = {letter: index for index, letter in enumerate(vacab_list)}

    return content, vocab_dict, vacab_list

def encode(text):
    char_len = len(text)
    int_repres = [char_dict[c] for c in list(text)] # Map to int
    onehot = np.zeros([char_len, len(char_dict.keys())])
    onehot[np.arange(0,char_len), int_repres] = 1
    return onehot

def decode(onehot):
    str_idx = np.argmax(onehot, axis=-1)
    string_arr = [char_list[c] for c in list(str_idx)] # Map to char
    decoded = ''.join(string_arr)
    return decoded

def eindot(a, b):
    if len(a.shape) < 2: # vec dot mat
        op = 'j,jk->k'
    elif len(b.shape) < 2: # mat dot vec
        op = 'jk,k->j'
    else:
        op = 'ij,jk->ik'
    dot = np.einsum(op, a, b)
    return dot

def einouter(a,b):
    dot = np.einsum('i,j->ij',a,b)
    return dot


if __name__ == '__main__':
    main()
