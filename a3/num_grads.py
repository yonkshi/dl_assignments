import numpy as np

def compute_num_grads_center(X, Y, h=1e-5):
    """
    A somewhat slow method to numerically approximate the gradients using the central difference.
    :param X: Data batch. d x n
    :param Y: Labels batch. K x n
    :param h: Step length, default to 1e-5. Should obviously be kept small.
    :return: Approximate gradients
    """
    # df/dx ≈ (f(x + h) - f(x - h))/2h according to the central difference formula

    params = [self.W, self.F1, self.F2]
    num_grads = []

    for i, param in enumerate(params):

        grad = np.zeros(param.shape)
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            old_value = param[ix]
            param[ix] = old_value + h
            plus_cost = self.compute_loss(X, Y)
            param[ix] = old_value - h
            minus_cost = self.compute_loss(X, Y)
            param[ix] = old_value  # Restore original value

            grad[ix] = (plus_cost - minus_cost) / (2 * h)
            it.iternext()  # go to next index

        if i > 0 and param.shape[0] == param.shape[2]:
            grad = grad.transpose(2, 1, 0)  # to make sure the 3D arrays come the same way as in backward function
        num_grads.append(grad)

    return num_grads