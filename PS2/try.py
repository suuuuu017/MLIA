import numpy as np

if __name__ == '__main__':
    a = np.array([[1,2,3], [4,5,6]])
    print(a)
    print(a.transpose())
    print(a.transpose().mean(1).transpose())