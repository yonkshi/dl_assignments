from scipy.io import loadmat, whosmat
import numpy as np
import matplotlib.pyplot as plt

train_balanced = np.load('train_balanced.npy')
train_unbalanced = np.load('train_unbalanced.npy')
val_balanced = np.load('val_balanced.npy')
val_unbalanced = np.load('val_unbalanced.npy')


plt.plot(train_balanced, label='Balanced training set', color='g', linestyle=':')
plt.plot(val_balanced, label='Balanced validation set', color='g')

plt.plot(train_unbalanced, label='Unbalanced training set', color='r', linestyle=':')
plt.plot(val_unbalanced, label='Unbalanced validation set', color='r')
plt.legend()
plt.xlabel('update steps (x100)')
plt.ylabel('accuracy')
plt.title('Balanced vs Unbalanced accuracy')
plt.show()

