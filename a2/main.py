import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import os

K = 10
d = 3072
global_mean_X = None
NUM_HIDDEN_NODES = 50
def main():
    find_lambda_and_etta()
    # #find_and_plot_learning_rate()
    return
    np.random.seed(400)
    train_X, train_Y, train_y = load_batch('data_batch_1', training=True)
    validate_X, validate_Y, validate_y = load_batch('data_batch_2')
    test_X, test_Y, test_y = load_batch('data_batch_3')

    tX = np.vstack([train_X, validate_X])
    tY = np.vstack([train_Y, validate_Y])

    train_X = tX[:-1000]
    train_Y = tY[:-1000]
    validate_X = tX[-1000:]
    validate_Y = tY[-1000:]



    batch_params = [
        # Batch_size, etta, rho, epochs, lambda, decay_rate
        {'batchsize': 100, 'etta': 0.033743940665, 'rho': 0.9, 'epochs': 30, 'lambda': 6.892099060126683e-07, 'decay_rate': 0.95, },
        ]
    for batch_param in batch_params:
        lambda_ = batch_param['lambda']
        XEnt = CrossEntropyPerceptron(lambda_, NUM_HIDDEN_NODES)
        tra_err, val_err, test_err = mini_batch_gd(train_X, train_Y, batch_param, XEnt, validate_X, validate_Y, test_X, test_Y, early_stopping = True)
        test_error = XEnt.compute_accuracy(test_X, test_y)
        #XEnt.visualize_weight('Weights %s' % param_to_str(batch_param))
        print('Test accuracy %0.2f%%' % (test_error * 100))

        plt.plot(tra_err, label='training cost')
        plt.plot(val_err, label='validation cost')
        plt.plot(test_err, label='Test cost')

    plt.title('Most optimal configuration')
    plt.legend()
    plt.savefig('momentum vs none.png',
                pad_inches=0,
                bbox_inches='tight')
    plt.show()

def find_and_plot_learning_rate():


    np.random.seed(400)
    train_X, train_Y, train_y = load_batch('data_batch_1', training=True)
    validate_X, validate_Y, validate_y = load_batch('data_batch_2')
    test_X, test_Y, test_y = load_batch('data_batch_3')

    batch_param = {'batchsize': 107, 'etta':0.2, 'rho':0.9, 'epochs':5, 'lambda':1e-6, 'decay_rate':0.9, }

    plt.figure()
    plt.title('Random learning rate search')
    samples = {}
    for ett in (np.random.sample(10) * 0.06 + 0.005):
        print('trying ett', ett)
        lambda_ = batch_param['lambda']
        batch_param['etta'] = ett
        XEnt = CrossEntropyPerceptron(lambda_, NUM_HIDDEN_NODES)
        trainerr, valerr, testerr = mini_batch_gd(train_X, train_Y, batch_param, XEnt)

        test_error = XEnt.compute_cost(test_X, test_Y)
        samples['%0.4f'%ett] = test_error

        #plt.plot(testerr, label='%0.3f'%ett)
    plt.legend()
    plt.show()
    print('samples:')
    for w in sorted(samples, key=samples.get):
        print('learning rate: %s, cost %.5f' % (w, samples[w]))

    #XEnt.visualize_weight('Weights %s' % param_to_str(batch_param))
    #print('Test accuracy %0.2f%%' % (test_error * 100))


def find_lambda_and_etta():
    np.random.seed(400)
    train_X, train_Y, train_y = load_batch('data_batch_1', training=True)
    validate_X, validate_Y, validate_y = load_batch('data_batch_2')
    test_X, test_Y, test_y = load_batch('data_batch_3')

    batch_param = {'batchsize': 107, 'etta': 0.2, 'rho': 0.9, 'epochs': 3, 'lambda': 1e-6, 'decay_rate': 0.9, }

    plt.figure()
    plt.title('Random learning rate search')
    samples = {}


    # exponential coarse search
    emax = 0
    emin = -7
    lambdas = emin + (emax - emin) * np.random.sample(10)
    lambdas = np.power(10, (lambdas))
    ettas = np.power(10,(emin + (emax - emin) * np.random.sample(10)))

    # Fine search
    # lmin = 0.0
    # lmax = 0.000001
    # etmin = 0.006
    # etmax = 0.045
    # lambdas = lmin + (lmax - lmin) * np.random.sample(10)
    # ettas = etmin + (etmax - etmin) * np.random.sample(25)

    for i, lambda_ in enumerate(lambdas):
        print((i  * 10), '%')
        for ett in ettas:
            batch_param['lambda'] = lambda_
            batch_param['etta'] = ett
            XEnt = CrossEntropyPerceptron(lambda_, NUM_HIDDEN_NODES)
            trainerr, valerr, testerr = mini_batch_gd(train_X, train_Y, batch_param, XEnt, validate_X, validate_Y, early_stopping=True)

            test_error = XEnt.compute_cost(test_X, test_Y)
            accuracy = XEnt.compute_accuracy(test_X, test_y)
            if np.isnan(test_error):
                test_error = 1000
            config = (lambda_, ett, accuracy)
            samples[accuracy] = (lambda_, ett, test_error)

        # plt.plot(testerr, label='%0.3f'%ett)
    plt.legend()
    plt.show()
    print('Sorted best config:')
    for accuracy, (lambda_, etta, err,) in sorted(samples.items(), reverse= True):
        print('lambda: %.12f, etta: %.12f, cost %.5f, accuracy: %2.3f%%' % (lambda_, etta, err, accuracy*100))
    print('done')

def mini_batch_gd(X, Y, gd_param,XEnt, validate_X=None, validate_Y=None, test_X = None, test_Y = None, early_stopping = False):

    batch_size, eta, rho, n_epochs, lambda_, decay_rate = gd_param.values()
    n_batches = int(X.shape[0]/batch_size)
    train_err = []
    validate_error = []
    test_error = []
    valerr = 1e100
    early_stopping_value = 2 # monotonic increase after n epoch will stop
    early_stop_counter = 0
    for e in range(n_epochs):
        err = XEnt.compute_cost(X, Y)
        train_err.append(err)
        if validate_X is not None:
            valerr_next = XEnt.compute_cost(validate_X, validate_Y)
            #early stopping
            if valerr_next > valerr and early_stopping:
                early_stop_counter += 1
                if early_stop_counter > early_stopping_value:
                    break
            else:
                early_stop_counter = 0
            valerr = valerr_next
            validate_error.append(valerr)

        if test_X is not None:
            t_err = XEnt.compute_cost(test_X, test_Y)
            test_error.append(t_err)
            print('epoch %d:' % e, t_err)
        for i in range(n_batches):
            start = int(i * batch_size)
            end = int(i * batch_size + batch_size)
            X_batch = X[start:end,...]
            Y_batch = Y[start:end,...]
            # train
            XEnt.train(X_batch,Y_batch,eta, rho)

        eta *= decay_rate

    #plot_error(train_err, n_epochs, batch_size, eta,lambda_, validate_error)

    return train_err, validate_error, test_error

def test_functions(X, Y, y):
    X = X[0:107, :]
    Y = Y[0:107, :]
    Xent = CrossEntropyPerceptron(0.1, NUM_HIDDEN_NODES)
    Xent.update_data(X, Y)
    p = Xent.train(X, Y, etta=0.01)
    print(p)

def compare_grad(x, y):
    XEnt = CrossEntropyPerceptron(0.1)
    XEnt.update_data(x, y)
    XEnt.evalulate_classifier()
    grad_W, grad_b = XEnt.compute_gradient()
    grad_W_num, grad_b_num = XEnt.numerical_grad(1e-6)
    diff = grad_diff(grad_W, grad_W_num, (1e-6))
    abs_diff = grad_W - grad_W_num
    print('Absolute diff:', np.mean(abs_diff), 'normalized diff', np.mean(diff))

def plot_error(train_error, epoch, batch_size, eta, lambda_, validation_err=None):
    plt.plot(train_error, label='train error')
    plt.plot(validation_err, label='validate error')
    param_str = param_to_str((batch_size, eta, epoch, lambda_))
    plt.title('Error rate at epoch, %s' % param_str)
    plt.legend()
    plt.savefig('error %s.png' %param_str,
                pad_inches=0,
                bbox_inches='tight')
    plt.show()

def load_batch(filename, training=False):
    path = 'Datasets/cifar-10-batches-py'
    file = os.path.join(path, filename)

    batch = np.load(file, encoding='bytes')
    data = np.array(batch[b'data'])
    labels = np.array(batch[b'labels'])
    labels = labels[:,np.newaxis]

    le = OneHotEncoder(sparse=True)
    one_k = le.fit_transform(labels)

    #csr mat -> np array
    one_k = one_k.toarray()

    # X (normalize), Y and y
    normalized_data = data/255.

    if training:
        global global_mean_X
        global_mean_X = normalized_data.mean(1, keepdims=True) # so it's (n, 1) rather than (n,)
    zero_mean_data = normalized_data - global_mean_X


    return zero_mean_data, one_k, labels

def visualize(X):

    x = X[0]
    x = x.reshape(3,32,32).transpose(1,2,0)*256
    x = x.astype('uint8')
    plt.imshow(x)
    plt.show()

def grad_diff(g1, g2, eps):
    nom = np.absolute(g1-g2)
    g1_abs = np.absolute(g1)
    g2_abs= np.absolute(g2)

    eps_mat = np.ones(g1.shape) * eps
    denom = np.maximum(eps_mat, g1_abs + g2_abs)

    return nom/denom

def param_to_str(param):
    return 'BatchSize_%d  etta_%0.2f epoch_%d lambda_%0.2f  ' % param

class CrossEntropyPerceptron(object):
    def __init__(self, lambda_, m):
        self.lambda_ = lambda_
        self.m = m # number of hidden nodes

        # momentum
        self.momentum_w = 0
        self.momentum_b = 0
        self.momentum_w2 = 0
        self.momentum_b2 = 0
        self.init_Wb()

    def update_data(self, X, Y):
        self.X = X
        self.Y = Y

    def update_wb(self, W, b):
        self.W = W
        self.b = b

    def init_Wb(self):
        # num hidden nodes
        m = self.m

        self.W = np.random.normal(0, 0.001, (K,m))
        self.b = np.random.normal(0, 0.001, (K,1))

        self.W2 = np.random.normal(0, 0.001, (m,d))
        self.b2 = np.random.normal(0, 0.001, (m,1))

    def visualize_weight(self, title):
        plt.figure(figsize=(7, 3))
        ax = plt.subplot(3, 5, 3)
        ax.set_axis_off()

        plt.title(title)
        for i in range(K):
            im = self.W[i, ...].reshape(3, 32, 32);
            ax = plt.subplot(3, 5, i + 6) # reserve first row for title
            ax.set_axis_off()
            min_ = np.min(im)
            max_ = np.max(im)
            s_im = (im - min_) / (max_ - min_)
            s_im = np.transpose(s_im, [1,2,0,])
            plt.imshow(s_im)
        plt.savefig('%s.png' % title,
                    pad_inches=0,
                    bbox_inches='tight')
        plt.show()

    def compute_cost(self, X, Y):
        W, W2, _lambda = self.W, self.W2, self.lambda_

        D = X.shape[0]
        self.update_data(X, Y)
        p = self.evalulate_classifier()

        # faster sum
        prod = np.sum(Y * p.T, axis=1)
        # prod_alt = np.einsum('ij, ij->i', Y, p.T)
        loss = - np.log(prod)
        summed = np.sum(loss)

        # regularization term
        reg = _lambda * (np.sum(np.square(W) + np.sum(np.square(W2))))

        ret = summed / D + reg
        return ret

    def compute_accuracy(self, X, y):
        W, b, lambda_ = self.W, self.b, self.lambda_
        D = X.shape[0]
        self.X = X
        p = self.evalulate_classifier()
        winners = np.argmax(p, axis=0)[:, np.newaxis]  # 2D vector

        diff = y - winners
        percentage = 1 - np.count_nonzero(diff) / D
        return percentage

    def train(self, X, Y, etta, rho, compare_with_numerical = False):
        self.update_data(X, Y)
        self.evalulate_classifier()
        grad_w, grad_b, g = self.compute_gradient()
        grad_w2, grad_b2 = self.compute_hidden_gradient(g)

        self.momentum_w = self.momentum_w * rho + etta * grad_w
        self.momentum_b = self.momentum_b * rho + etta * grad_b
        self.momentum_b2 = self.momentum_b2 * rho + etta * grad_b2
        self.momentum_w2 = self.momentum_w2 * rho + etta * grad_w2

        if compare_with_numerical:
            ngrad_w, ngrad_b = self.numerical_grad()
            ngrad_w2, ngrad_b2 = self.numerical_grad(calculate_hidden=True)
            diff_w = grad_w - ngrad_w
            diff_b = ngrad_b - grad_b
            diff_w2 = grad_w2 - ngrad_w2
            diff_b2 = ngrad_b2 - grad_b2

            mean_w = np.mean(diff_w)
            mean_b = np.mean(diff_b)
            mean_w2 = np.mean(diff_w2)
            mean_b2 = np.mean(diff_b2)
            print('k')


        self.W -= self.momentum_w
        self.b -= self.momentum_b

        self.W2 -= self.momentum_w2
        self.b2 -= self.momentum_b2

    def compute_hidden_gradient(self, g):

        X, W2, W, lambda_, s2 = self.X, self.W2, self.W, self.lambda_, self.s2
        D = X.shape[0]

        g = g.dot(W).T
        s2_m = np.where(s2>0, 1, 0)
        g = g * s2_m
        delta_b2 = np.sum(g, axis=1)
        delta_w2 = g.dot(X)

        delta_b2 /= D
        delta_w2 /= D

        delta_w2 += 2 * lambda_ * W2
        return delta_w2, delta_b2[:, None]

    def compute_gradient(self, ):
        h, Y, W, b, lambda_, P = self.h, self.Y, self.W, self.b, self.lambda_, self.P
        D = Y.shape[0]

        g = P.T - Y
        delta_b = np.sum(g, axis=0)
        delta_W = h.dot(g).T

        delta_b /= D
        delta_W /= D

        delta_W += 2 * lambda_ * W

        return delta_W, delta_b[:, None], g  # converting to 2 dim vector

    def numerical_grad(self, h=(1e-5), calculate_hidden=False):
        X, Y, lambda_ = self.X, self.Y, self.lambda_

        # perserve value
        if calculate_hidden:
            W = np.copy(self.W2)
            b = np.copy(self.b2)
        else:
            W = np.copy(self.W)
            b = np.copy(self.b)

        grad_W = np.zeros(W.shape);
        grad_b = np.zeros(b.shape);



        c = self.compute_cost(X, Y);
        print('numerical start')
        for i in range(len(b)):
            b_try = np.copy(b);
            b_try[i] += h;

            # update so compute cost can use these values
            if calculate_hidden:
                self.b2= b_try
            else:
                self.b = b_try

            c2 = self.compute_cost(X, Y);
            grad_b[i] = (c2 - c) / h
        print('numerical half way..')

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.copy(W);
                W_try[i, j] = W_try[i, j] + h;

                # update so compute cost can use these values
                if calculate_hidden:
                    self.W2 = W_try
                else:
                    self.W = W_try

                c2 = self.compute_cost(X, Y)
                grad_W[i, j] = (c2 - c) / h
        print('numerical completed!')
        return grad_W, grad_b

    def evalulate_classifier(self):
        X, Y, W, W2, b, b2,  lambda_= self.X, self.Y, self.W, self.W2, self.b, self.b2, self.lambda_

        #input layer
        s2 = W2.dot(X.T) + b2

        zero_mat = np.zeros(s2.shape)
        h = np.maximum(zero_mat, s2) # ReLU
        s = W.dot(h) + b

        # softmax
        nom = np.exp(s)
        denom = np.sum(nom, axis=0)
        p = nom / denom
        self.P = p
        self.h = h
        self.s2 = s2
        return p


main()

