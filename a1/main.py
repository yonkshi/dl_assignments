import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import os

K = 10
d = 3072
def main():
    np.random.seed(400)
    train_X, train_Y, train_y = load_batch('data_batch_1')
    validate_X, validate_Y, validate_y = load_batch('data_batch_2')
    test_X, test_Y, test_y = load_batch('data_batch_3')

    batch_params = [
        # Batch size, etta, epochs, lambda,
        (100, 0.1, 40, 0),
        (100, 0.01, 40, 0),
        (100, 0.01, 40, .1),
        (100, 0.01, 40, 1),
        ]
    for batch_param in batch_params:
        lambda_ = batch_param[3]
        XEnt = CrossEntropyPerceptron(lambda_)
        mini_batch_gd(train_X, train_Y, batch_param, XEnt, validate_X, validate_Y)
        test_error = XEnt.compute_accuracy(test_X, test_y)
        XEnt.visualize_weight('Weights %s' % param_to_str(batch_param))
        print('Test accuracy %0.2f%%' % (test_error * 100))
def compare_grad(x, y):
    XEnt = CrossEntropyPerceptron(0.1)
    XEnt.update_data(x,y)
    XEnt.evalulate_classifier()
    grad_W, grad_b = XEnt.compute_gradient()
    grad_W_num, grad_b_num = XEnt.numerical_grad(1e-6)
    diff = grad_diff(grad_W, grad_W_num, (1e-6))
    abs_diff = grad_W - grad_W_num
    print('Absolute diff:', np.mean(abs_diff), 'normalized diff', np.mean(diff))


class CrossEntropyPerceptron(object):
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        self.init_Wb()

    def update_data(self, X, Y):
        self.X = X
        self.Y = Y

    def update_wb(self, W, b):
        self.W = W
        self.b = b

    def init_Wb(self):
        self.W = np.random.normal(0, 0.01, (K,d))
        self.b = np.random.normal(0, 0.01, (K,1))

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
        W, b, _lambda = self.W, self.b, self.lambda_

        D = X.shape[0]
        self.update_data(X, Y)
        p = self.evalulate_classifier()

        # faster sum
        prod = np.sum(Y * p.T, axis=1)
        # prod_alt = np.einsum('ij, ij->i', Y, p.T)
        loss = - np.log(prod)
        summed = np.sum(loss)

        # regularization term
        reg = _lambda * np.sum(np.square(W))

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

    def train(self, X, Y, etta, ):
        self.update_data(X, Y)
        self.evalulate_classifier()
        grad_w, grad_b = self.compute_gradient()
        self.W -= etta * grad_w
        self.b -= etta * grad_b
        return grad_w, grad_b

    def compute_gradient(self, ):
        X, Y, W, b, lambda_, P = self.X, self.Y, self.W, self.b, self.lambda_, self.P
        D = X.shape[0]

        g = P.T - Y
        delta_b = np.sum(g, axis=0)
        delta_W = g.T.dot(X)

        delta_b /= D
        delta_W /= D

        delta_W += 2 * lambda_ * W

        return delta_W, delta_b[:, None]  # converting to 2 dim vector

    def numerical_grad(self, h):
        X, Y, W, b, lambda_, P = self.X, self.Y, self.W, self.b, self.lambda_, self.P
        no = len(W);
        d = len(X);

        grad_W = np.zeros(W.shape);
        grad_b = np.zeros(no);

        c = self.compute_cost(X, Y);
        print('numerical start')
        for i in range(len(b)):
            b_try = np.copy(b);
            b_try[i] += h;
            self.update_wb(W,b_try)
            c2 = self.compute_cost(X, Y);
            grad_b[i] = (c2 - c) / h
        print('numerical half way..')
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.copy(W);
                W_try[i, j] = W_try[i, j] + h;
                self.update_wb(W_try, b)
                c2 = self.compute_cost(X, Y)
                grad_W[i, j] = (c2 - c) / h
        print('numerical completed!')
        return grad_W, grad_b

    def evalulate_classifier(self):
        X, Y, W, b, lambda_= self.X, self.Y, self.W, self.b, self.lambda_
        s = W.dot(X.T) + b

        # softmax
        nom = np.exp(s)
        denom = np.sum(nom, axis=0)
        p = nom / denom
        self.P = p
        return p

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

def load_batch(filename):
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
    return normalized_data, one_k, labels

def visualize(X):

    x = X[0]
    x = x.reshape(3,32,32).transpose(1,2,0)*256
    x = x.astype('uint8')
    plt.imshow(x)
    plt.show()

def mini_batch_gd(X, Y, gd_param,XEnt, validate_X=None, validate_Y=None):

    batch_size, eta, n_epochs, lambda_ = gd_param
    n_batches = int(X.shape[0]/batch_size)
    train_err = []
    validate_error = []
    for e in range(n_epochs):
        err = XEnt.compute_cost(X, Y)
        train_err.append(err)

        if validate_X is not None:
            valerr = XEnt.compute_cost(validate_X, validate_Y)
            validate_error.append(valerr)
        print('epoch %d:' % e, err)
        for i in range(n_batches):
            start = int(i * batch_size)
            end = int(i * batch_size + batch_size)
            X_batch = X[start:end,...]
            Y_batch = Y[start:end,...]

            #compare_grad(X_batch, Y_batch)

            #train
            XEnt.train(X_batch,Y_batch,eta)


    plot_error(train_err, n_epochs, batch_size, eta,lambda_, validate_error)
def grad_diff(g1, g2, eps):
    nom = np.absolute(g1-g2)
    g1_abs = np.absolute(g1)
    g2_abs= np.absolute(g2)

    eps_mat = np.ones(g1.shape) * eps
    denom = np.maximum(eps_mat, g1_abs + g2_abs)

    return nom/denom

def param_to_str(param):
    return 'BatchSize_%d  etta_%0.2f epoch_%d lambda_%0.2f  ' % param

main()

