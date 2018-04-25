from scipy.io import loadmat, whosmat
NAMES = 'ascii_names.txt'


whosma = whosmat('DebugInfo.mat')
matfile = loadmat('DebugInfo.mat')


def get_names():
    """
    Get the 'ascii_names' file
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

print(matfile)