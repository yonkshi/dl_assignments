import numpy as np
from time import time
np.random.seed(0)

batch_size = 100
row = 10
mx_cols = 13
rate = 0.1
repeat = 100
# generate mx
size = row*mx_cols
mx = np.tile(np.random.choice([0,1], size, p=[1-rate, rate]).reshape(row,mx_cols,1), (1,1,batch_size))
t0 = time()
flattened = mx.reshape(row*mx_cols, batch_size, order='F')
loc = np.argwhere(flattened)
print('pre_processing_time', time() - t0)

t0 = time()
for i in range(repeat):
    g = np.arange(1000,1000+row)[:,None]
    g2 = np.tile(g, (1,mx_cols, batch_size)).T
    g3 = g2 * mx
    g4 = g3[g3>0]
print('original time', time() - t0)

t1 = time()
for i in range(repeat):
    g = np.arange(1000, 1000 + row)[:, None]
    take = np.take(g, loc, mode='wrap')
print('improved time', time() - t1)
#mask = mx > 0
#masked = np.ma.masked_array(G, mask=mask, fill_value=0)
print('hello')
