from scipy.io import loadmat, whosmat
import numpy as np
from convnet import *
import matplotlib.pyplot as plt



NAMES = 'ascii_names.txt'
D = 28 #Input dim
N_LEN = 19
matfile = loadmat('DebugInfo.mat')
K = 18 # Output dim
X = None



def main():
    np.random.seed(1501)
    names, labels, val_idx = get_names()
    N = len(names)
    global X
    # X = np.zeros((N, N_LEN * D))
    # for i, name in enumerate(names):
    #     X[i,...] = encode(name)

    _, unique_idx = np.unique(labels, return_index=True)

    x_batch = np.vstack(map(encode,names)).T
    y_batch = label_encode(labels)
    c = ConvNet()
    c.load()
    #predict('barrera', c)
    c.compute_batch(x_batch, y_batch, val_idx, labels)

def predict(name, convnet):
    with open('category_labels.txt') as f:
        content = f.readlines()

    languages = []
    for line in content:
        languages.append( line.split(' ')[1].rstrip() )
    encoded_name = encode(name)
    p = convnet.forward(encoded_name, )
    for i, lang in enumerate(languages):
        print(lang, '%0.2f%%' % (p[i] * 100))

def get_names(max=10000000):
    """
    Get the 'ascii_names.txt' file
    :return:
    """
    with open('ascii_names.txt') as f:
        content = f.readlines()
    names = []
    nums = []
    for i, line in enumerate(content):
        splitted = line.rsplit(None,1)
        if len(splitted ) > 2:
            print('wtf')
        name, num = splitted
        names.append(name)
        nums.append(int(num)-1) # 0 index
        if i >= max-1:
            break


    with open('valset_idx.txt') as f:
        content = f.readlines()
    for i, line in enumerate(content):
        numstr = line.split(' ')
    numstr.pop() # remove trailing empty string
    nparr = np.array(numstr)
    val_idx = nparr.astype(np.int)

    return names, nums, val_idx

def label_encode(l):
    rows = len(l)
    global K
    encoded = np.zeros((rows,K))
    encoded[np.arange(rows),l] = 1
    return encoded.T

def encode(name):
    # sanitize:
    sname = sanitize(name)
    index = np.arange(0,len(sname)) # Used for updating ascii
    asciimap = [ord(c) - 97 for c in sname]
    encoded = np.zeros((N_LEN, D))
    encoded[index,asciimap] = 1
    encoded = encoded.flatten()
    return encoded

def decode(encoded):
    encoded = encoded.reshape((N_LEN,D))
    array = np.argmax(encoded, axis=-1) + 97
    w_len = np.argmin(np.sum(encoded, axis=-1)) # word length
    array = array[:w_len]
    name = ''.join(map(chr,array)) #converts ascii to string
    return desanitize(name)

def sanitize(name):
    return name.lower().replace(',','')\
        .replace(' ','|')\
        .replace('\'','{') # replace ' to ascii 123

def desanitize(name):
    return name.replace('|',' ').replace('{','\'')

def hehe(strrr):
    return len(sanitize(strrr))

def analyze_encode(names):
    longest = sorted(map(hehe, names))[-1]
    return 'he'



if __name__ == '__main__':
    main()