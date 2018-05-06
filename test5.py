import numpy as np
from time import time
np.random.seed(0)

batch_size = 100
row = 300
mx_cols = 2800
rate = 0.3
repeat = 10
# generate mx
size = row*mx_cols

idx = np.random.randint(0, 300 * 100, 2500)
idx2 = np.random.randint(0, 300, 2500)
idx3 = np.random.randint(0, 300, 2500)
t0 = time()

g_size = 300

simulated_g = np.arange(0, g_size * batch_size).reshape(g_size, batch_size, order='F')

simulated_nf = 10
res = simulated_g.reshape(simulated_nf, 30, -1, order='F') # nf should be cols but because of ordering it needs to be cols
res2 = np.moveaxis(res,0,1)

#
# ix = np.array([[0,1],[0,5],[1,4],[2,0],[5,7],[9,8]]) # <=== wrong, redo this
# ix_new = ix[:,0] * g_size + ix[:,1]
# check = simulated_g.flatten('F').take(ix_new)
# delete_me = simulated_g.take([5,10, 102])
# ix_2 = ix[:,0] + ix[:,1]*batch_size
# check2 = simulated_g.take(ix_new)

# for i in range(20000):
#     simulated_g = simulated_g + 1
#     simulated_g.flatten('F')
# TODO Figure out how to convert G x batch index into flattened indices << DONE

# TODO Handle when a column has fully 0

# TODO handle occurences where idx = 2 << Not needed anymore

print('index time', time() - t0)

mx = np.tile(np.random.choice([0,1], size, p=[1-rate, rate]).reshape(row,mx_cols,1), (1,1,batch_size))

# nonz = np.count_nonzero(mx, axis=0)
# t0 = time()
# flattened = mx.reshape(row*mx_cols, batch_size, order='F')
# loc = np.argwhere(flattened)


t0 = time()
print('begin')
for i in range(repeat):
    for j in range(mx_cols):
        for k in range(batch_size):
            zz = np.argwhere(mx[:,j,k] > 0)
    print(zz.shape)
    # g = np.arange(1000,1000+row)[:,None]
    # g2 = np.tile(g, (1,mx_cols, batch_size)).T
    # g3 = g2 * mx
    # g4 = g3[g3>0]
print('argwhere everything', time() - t0)

t1 = time()
for i in range(repeat):
    g = np.arange(1000, 1000 + row)[:, None]
    take = np.take(g, loc, mode='wrap')
print('improved time', time() - t1)
#mask = mx > 0
#masked = np.ma.masked_array(G, mask=mask, fill_value=0)
print('hello')
