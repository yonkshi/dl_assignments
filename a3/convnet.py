import numpy as np
from numpy import sum
from numpy.random import normal
import matplotlib.pyplot as plt
from time import time
from confusion import *


NAMES = 'ascii_names.txt'
D = 28 # Number of chars
N_LEN = 19 # Maximum char for longest word

X = None
UPDATE_STEPS = 20000
BATCH_SAMPLE_SIZE = 150 # batch size
BALANCED_DATA = True
CHECK_GRAD = True

'''
d: dimension of char space (28 types of chars)
nlen: dimension of char (max number of char per word)


n1: number of filters at layer one
nf: generic number of filters
dd: filter height
k: width of filter, not related to 
'''

class HyperParam:
    def __init__(self, k=[5, 3],nf=[20, 20], K = 18, learning_rate=1e-6, momentum_coeff=0.9, momentum_decay=1):
        self.k = k # filter width
        self.nf = nf # number of filters
        self.fsize = (N_LEN - k[-1] + 1) * nf[-1]
        self.K = K # output dimensions

        self.etta = learning_rate
        self.rho = momentum_coeff
        self.precomputed_v1_dimension = D * k[0] * nf[0]
        self.hidden_layers = len(nf)-1 #exclude output
        self.rho_decay = momentum_decay

        # compute nlen
        self.nlen = [N_LEN]
        for i in range(len(nf)):
            self.nlen += [self.nlen[-1] - k[i] + 1]

        # He initialization
        # Input layer, hidden conv layer, and dense layer
        sigma = [k[0]] # input sparse layer
        sigma += [ k * n for k, n in zip(k[1:],nf[:-1])]
        sigma += [self.nlen[-1] * nf[-1]]
        sig = np.array(sigma)
        self.sigma = np.sqrt(2/sig)

        # Speed comparison bottleneck

    def __str__(self):
        k = ' '.join(map(str,self.k))
        nf = ' '.join(map(str,self.nf))
        return 'layers: {}, k: {}, nf: {}, etta: {}, rho: {}, rho decay: {}, balanced: {}'.format(self.hidden_layers+1, k, nf, self.etta, self.rho, self.rho_decay, BALANCED_DATA)

class ConvNet():
    def __init__(self, hyperparam=HyperParam(), ):

        self.hp = hyperparam
        self.hidden_f = [None] * self.hp.hidden_layers
        self.nlen = [N_LEN]
        # initialize filters
        d = D
        self.input_f = normal(0, self.hp.sigma[0], (d, self.hp.k[0], self.hp.nf[0]))
        #self.input_f = np.random.randn(d, self.hp.k[0], self.hp.nf[0]) * 0.1
        d = self.hp.nf[0]
        for i, ki in enumerate(self.hp.k):
            if i == 0: # skip input layer
                continue
            ni = self.hp.nf[i]
            self.hidden_f[i-1] = normal(0, self.hp.sigma[i], (d, ki, ni))
            #self.hidden_f[i-1] = np.random.randn(d, ki, ni) * 0.1
            d = ni

        fsize = self.hp.nf[-1] * self.hp.nlen[-1]
        self.w = normal(0, self.hp.sigma[-1],(self.hp.K, fsize))
        #self.w = np.random.randn(self.hp.K, fsize) * 0.1
        self.mf=[]
        self.dF2 = 0
        self.dF1 = 0
        self.dW = 0

        # mommentum
        self.dW_momentum = 0
        self.dF_momentum = [0] * self.hp.hidden_layers
        self.input_dF_momentum = 0
        print(self.hp)

    def compute_batch(self, X, Y, validation_idx, labels,  W=None):
        # preparing data
        x_val_full = X[:, validation_idx]
        y_val_full = Y[:, validation_idx]
        x_train = np.delete(X, validation_idx, axis=1) # split validation_set for
        y_train = np.delete(Y, validation_idx, axis=1)

        # cutting up classes:
        y_sum = y_train.sum(axis=1).astype('int')
        balanced_batch_size = (np.min(y_sum)/17).astype('int')
        start = 0
        classes = []
        for l in y_sum:
            end = start + l
            classes.append((start, end))
            start = end

        training_size = x_train.shape[1]
        t0 = time()
        print('begin preprocessing ...')
        #self.preprocess_mx_superefficient(X, batchsize)
        self.preprocess_mx_fast(X)
        print('preprocess time', time() - t0)
        t0 = time()
        loss = []
        accuracy = []
        validation_loss = []
        validation_accuracy = []
        z = 20000
        for z in range(UPDATE_STEPS):
            self.mf = []
            if BALANCED_DATA:
                samples = np.array([]).astype('int')
                for a, b in classes:
                    idxs = np.random.randint(a, b, balanced_batch_size)
                    samples = np.concatenate((samples, idxs))
                samples.astype('int')
                x_batch = x_train[:, samples]
                y_batch = y_train[:, samples]

            else:
                samples = np.random.randint(0, training_size, BATCH_SAMPLE_SIZE)
                x_batch = x_train[:, samples]
                y_batch = y_train[:, samples]




            # Training sampling:
            # train_samples = np.random.randint(0, training_size, VALIDATION_SAMPLE_SIZE)
            # x_val = X[:, train_samples]
            # y_val = Y[:, train_samples]

            #i = int(start / batchsize)
            #self.mx1_batch = self.preprocess_mx_superefficient(x_batch, x_batch.shape[1]) #self.precomputed_mx1_full[i]
            self.mx1_batch = self.sparse_mx1_to_col(samples)
            p = self.forward(x_batch)
            dw, hidden_df, input_df = self.backward(y_batch, p)

            if CHECK_GRAD:
                ndw, nidf, nhdf = self.compute_num_grads_center(x_batch, y_batch)

                diff_w = np.sum(np.abs(dw) - np.abs(ndw))
                diff_input_df = np.sum(np.abs(input_df) - np.abs(nidf))
                diff_hidden_df = np.sum(np.abs(hidden_df) - np.abs(nhdf))
                print('hello world')

            self.dW_momentum = self.dW_momentum * self.hp.rho + dw * self.hp.etta
            self.dF_momentum = [ f * self.hp.rho + hidden_df[i] for i,f in enumerate(self.dF_momentum)]
            self.input_dF_momentum = self.input_dF_momentum * self.hp.rho + input_df * self.hp.etta

            self.hp.rho *= self.hp.rho_decay #rho decay

            self.w -= self.dW_momentum
            self.hidden_f = [f - self.dF_momentum[i] for i, f in enumerate(self.hidden_f)]
            self.input_f -= self.input_dF_momentum

            # benchmark stuff
            if z % 100 == 0 and z > 0:
                print('=== %d / %d Complete ===' % (z, UPDATE_STEPS) )
                print('time:', '%0.2f' % (time()-t0))
                print('loss:', self.loss(x_batch, y_batch, p=p))
                print('accu:', self.accuracy(x_batch, y_batch, p=p))
                print('validation loss:', self.loss(x_val_full, y_val_full))
                print('validation accu:', self.accuracy(x_val_full, y_val_full))

                # Benchmark and reporting stuff
                loss += [self.loss(x_batch, y_batch, p=p)]
                accuracy += [self.accuracy(x_batch, y_batch, p=p)]

                validation_loss += [self.loss(x_val_full, y_val_full)]
                validation_accuracy += [self.accuracy(x_val_full, y_val_full)]

                t0 = time()

            if (z+1) % 5000 == 0 and z > 0:
                validation_p = self.forward(x_val_full)
                cm = get_confusion_matrix(validation_p, y_val_full)
                target_names = np.loadtxt('category_labels.txt', dtype=str)[:,1]
                plot_confusion_matrix(cm, target_names, title='Validation, step=%d' % z)
        # plotting

        validation_p = self.forward(x_val_full)
        cm = get_confusion_matrix(validation_p, y_val_full)
        target_names = np.loadtxt('category_labels.txt', dtype=str)[:,1]
        plot_confusion_matrix(cm, target_names, title='Balanced confusion matrix, step=%d' % z)

        #np.save('val_unbalanced', validation_accuracy)
        #np.save('train_unbalanced', accuracy)
        self.save()
        self._plot_loss(loss, validation_loss)
        self._plot_accuracy(accuracy, validation_accuracy)

    def forward(self, x_input, params=None):
        if params is not None:
            weights, input_f, f2 = params
            filters = [f2]
            modify_self = False
        else:
            weights, input_f, filters = self.w, self.input_f, self.hidden_f
            modify_self = True #

        X = [x_input]
        n_len = N_LEN

        # input layer
        x = X[-1]
        mf = self.make_mf_matrix(input_f, n_len)
        s1 = mf.dot(x)
        s2 = np.maximum(s1, 0)
        n_len = int(mf.shape[0] / input_f.shape[2])  # (n_len-k+1) * nf / nf
        X.append(s2)
        if modify_self: self.mf.append(mf)

        # Hidden Filters
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
        df = [None] * self.hp.hidden_layers
        for i in np.arange(self.hp.hidden_layers)[::-1]: # [::-1] reverses list

            x = self.x[i+1] # previous input, and offset the input layer
            f = self.hidden_f[i]
            d, k, nf = f.shape

            mx = self.make_mx_matrix(x, d, k, nf, optimized=True)
            new_g = G_.reshape(mx.shape[0], nf, -1)  # nf should actually be cols, but einsum eliminates the need
            v_vec = np.einsum('ijk,iyk->jy', mx, new_g) / n

            v = v_vec.reshape(f.shape, order='F')

            #self.dF[i] = v
            df[i] = v
            mf = self.mf[i+1] # offset from input layer
            G_ = mf.T.dot(G_)
            G_ = G_ * (x > 0)

        # input conv layer
        # use precomputed value
        v_vec = np.zeros(self.hp.precomputed_v1_dimension)
        for col_ix, gix in self.mx1_batch.items():
            to_be_summed = G_.take(gix)
            v_vec[col_ix] = np.sum(to_be_summed)
        v_vec /= n
        input_df = v_vec.reshape(self.input_f.shape, order='F')

        return dW, df, input_df

    def preprocess_mx_superefficient(self, x_input, batch_size):

        d, k, nf = self.input_f.shape

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

        #self.precomputed_mx1_full = batches

        return cols_idx

    def preprocess_mx_fast(self, x_input):
        d, k, nf = self.input_f.shape

        nonzeros = np.argwhere(x_input.T > 0)
        # split up into words, one per
        unique, split_ix = np.unique(nonzeros[:,0],return_index=True)
        words = np.split(nonzeros[:,1], split_ix[1:]) # fixed a bug

        # used to pre-preprocess batch index, tiny speed up

        preprocessed = []
        mx_row_lengths = []
        mx_col_lengths = []
        for word_ix, word in enumerate(words):

            # create a list of x sub vectors for each word
            nlen = int(x_input.shape[0] / d)
            vec_x_cols = d * k
            x_rows = (nlen - k + 1)

            # Stride x to list
            # Words have consecutive character with nlen each, terminate at the last character,
            # also assume strides are 1, but easy change (modify i) to accomondate different strides
            mx = []
            for i in range(x_rows):
                if i >= word.size:
                    break  # end of word
                end = i + k if i + k < word.size else word.size
                x_vec = word[i:end] - d * i  # assume 1 stride (aka 1 character move)

                # n filters duplicate
                for j in range(nf):
                    offset = j * vec_x_cols
                    mx_row = x_vec + offset
                    mx.append(mx_row)

            preprocessed.append(mx)
        self.precomputed_mx1_full = np.array(preprocessed, dtype='object')

    def sparse_mx1_to_col(self, indices):
        '''
        Uses preprocessed mx1 and convert into mx1 dot output indices, so that
        :param indices:
        :return:
        '''

        batch_size = len(indices)

        # contain output column indices
        cols_idx = {}

        # loop through batch
        for batch_i, full_i in enumerate(indices):
            mx_mat = self.precomputed_mx1_full[full_i]
            # loop through row of each mx mat
            for r, row in enumerate(mx_mat):
                cols = mx_mat[r]
                # loop through columns of mx mat
                for col in cols:
                    if col not in cols_idx:
                        cols_idx[col] = []
                    gix = r * batch_size + batch_i  # g_row * batch_size + batch number
                    cols_idx[col].append(gix)
        return cols_idx

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

        params = [np.copy(self.w), np.copy(self.input_f), np.copy(self.hidden_f[0])]
        num_grads = []

        for i, param in enumerate(params):

            grad = np.zeros(param.shape)
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                old_value = param[ix]
                param[ix] = old_value + h
                plus_cost = self.loss(X, Y, None, params)
                param[ix] = old_value - h
                minus_cost = self.loss(X, Y, None, params)
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
        plt.plot(loss, label='training')
        plt.plot(validation_loss, label='validation')
        plt.title('training loss')
        plt.xlabel('update cycle')
        plt.legend()
        plt.show()

    def _plot_accuracy(self, accuracy, validation_accuracy=None):
        plt.figure()
        plt.plot(accuracy, label='training')
        plt.plot(validation_accuracy, label='validation')
        plt.title('prediction accuracy')
        plt.xlabel('update cycle')
        plt.legend()
        plt.show()

    def save(self):
        filename = 'h%d_'% (len(self.hp.k),)
        filename += ''.join(['%sd%d_' % (k, self.hp.nf[i])for i,k in enumerate(self.hp.k)])

        with open('saved_states/%s_params.txt'%filename, 'w+') as file:
            params = str(self.hp)
            file.write(params)

        np.save('saved_states/' + filename + 'w', self.w)
        np.save('saved_states/' + filename + 'input_f', self.input_f)
        for i, hidden_f in enumerate(self.hidden_f):
            np.save('saved_states/' + filename + 'hidden_f' + str(i), hidden_f)
        print('saved')

    def load(self):
        '''
        Hardcoded load function, to save some time
        :return:
        '''
        self.input_f = np.load('saved_states/h2_5d20_3d20_input_f.npy')
        self.w = np.load('saved_states/h2_5d20_3d20_w.npy')
        self.hidden_f = [np.load('saved_states/h2_5d20_3d20_hidden_f0.npy')]




