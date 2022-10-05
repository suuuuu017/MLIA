import numpy as np
from scipy import ndimage

if __name__ == '__main__':
    a = np.arange(12.).reshape((4, 3))

    print(a[0:1, :])