import numpy as np

if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 2], [1, 2]])
    print(type(a))
    print(type(b))
    print(a + 1)
    print(np.square(a))
    print(np.square(b))
    print(np.square(a) + np.square(b))
    print(np.sqrt(np.square(a) + np.square(b)))
    print(np.square(a) / np.square(b))
    print(a**2)