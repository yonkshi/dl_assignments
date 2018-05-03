from scipy.io import loadmat, whosmat
import numpy as np
from convnet import *

NAMES = 'ascii_names.txt'
D = 28 #Input dim
N_LEN = 19
matfile = loadmat('DebugInfo.mat')

X = None



def main():
    names, num = get_names()
    N = len(names)
    global X
    # X = np.zeros((N, N_LEN * D))
    # for i, name in enumerate(names):
    #     X[i,...] = encode(name)
    #
    X = np.vstack(map(encode,names))
    e = encode('hohoho')
    d = decode(e)
    print(d)

def get_names():
    """
    Get the 'ascii_names.txt' file
    :return:
    """
    with open('ascii_names.txt') as f:
        content = f.readlines()
    names = []
    nums = []
    for line in content:
        splitted = line.rsplit(None,1)
        if len(splitted ) > 2:
            print('wtf')
        name, num = splitted
        names.append(name)
        nums.append(num)
    return names, num

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
    array = np.argmax(encoded, axis=1) + 97
    w_len = np.argmin(np.sum(encoded, axis=1)) # word length
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