import numpy as np

A = np.tile(np.array([0,0.1]),[2,2])
B = np.arange(0,12).reshape(3,4)

td= np.tensordot(A,B,(1,1))


a = A[:,None,:]
b = B[None,:,:]
s = np.sum(a*b,axis=-1)
print(td.shape)

