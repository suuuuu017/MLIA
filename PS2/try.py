import numpy as np
from scipy import ndimage

if __name__ == '__main__':
    a = np.arange(12.).reshape((4, 3))
    b = ndimage.map_coordinates(a, [[[0.5, 2, 1], [0.5, 2, 1], [0.5, 2, 1]],
                                    [[0.5, 1, 1], [0.5, 1, 1], [0.5, 1, 1]]], order=3)
    print(b)