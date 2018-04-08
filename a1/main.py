from a1.cifar import *
import numpy as np

np.random.seed(400)
train_X, train_Y, train_y = load_batch('data_batch_1')
validate_X, validate_Y, validate_y = load_batch('data_batch_2')
test_X, test_Y, test_y = load_batch('data_batch_3')
W,b = init_Wb()

def mini_batch_gd(X, Y, gd_param, W, b, validate_X=None, validate_Y=None):

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

            #train
            XEnt.train(X_batch,Y_batch,eta)

    plot_error(train_err, n_epochs, batch_size, eta,lambda_, validate_error)
    return W, b

def param_to_str(param):
    return 'BatchSize_%d  etta_%0.2f epoch_%d lambda_%0.2f  ' % param

batch_param = (50, 0.01, 100, 0.01)
lambda_ = batch_param[3]
XEnt = CrossEntropyPerceptron(lambda_)
W, b = mini_batch_gd(train_X, train_Y, batch_param, W, b, validate_X, validate_Y)
#test_error = compute_accuracy(test_X, test_y, W, b)
test_error = XEnt.compute_accuracy(test_X,test_y)
print('Test accuracy %0.2f%%' % (test_error*100))

#visualize_weight(W, 'Weights visualized for configuration %s' % param_to_str(batch_param))


#c = compute_cost(train_x, train_y, W, b, 0)
#acc = compute_accuracy(X, y, W, b)

#
#
# train_x = train_X[0:100]
# train_y = train_Y[0:100]
# P = evalulate_classifier(train_x, W, b)
# #
# # visualize(train_x)
# grad_w_loop, grad_b_loop = compute_gradient_loop(train_x, train_y, P, W, 0)
# grad_w_num, grad_b_num = numerical_grad(train_x, train_y, W, b, 0.0, 1e-6)
# grad_w, grad_b = compute_gradient(train_x, train_y, P, W, 0.0)
# #
# #
# diff = grad_w_num - grad_w
# dff2 = grad_b_num[...,None] - grad_b
# #numerical_cost = compute_cost(X, Y, W, b, 0.1)
# #
# diffB = grad_w_num - grad_w_loop
# diffB2 = grad_b_num - grad_b_loop
#
# print('grad_W diff: ',graddiff(grad_w, grad_w_num))
# print('grad_b diff: ',graddiff(grad_b, grad_b_num))
#

# #
# diffC = grad_w - grad_w_loop
# diffC2 = grad_b - grad_b_loop
print('hello world')