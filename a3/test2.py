import numpy as np
from convnet import *
from scipy.io import loadmat


matfile = loadmat('DebugInfo.mat')
F = matfile['F']
X_input = matfile['X_input']
x_input = matfile['x_input']
vecF = matfile['vecF']
vecS = matfile['vecS']
S = matfile['S']

d, n_len = X_input.shape
_, k, nf = F.shape

c = ConvNet()
mf = c.make_mf_matrix(F, n_len)
mx = c.make_mx_matrix(x_input, d, k, nf)

s = mf.dot(x_input)

diff = vecS - s
diff_mean = diff.mean()

s2 = mx.dot(vecF)
diff2 = vecS - s2
diff2_mean = diff2.mean()
print('hello')

ff = np.arange(1, 40+1).reshape(5,4,2, order='F')
mf2 = c.make_mf_matrix(ff, 8)
print('hello')
