import numpy as np
from numpy.linalg import eig
from scipy import io
import numpy as np
from numpy.linalg import eig
from matplotlib import pyplot as plt

if __name__ == '__main__':
    data = io.loadmat('MINIST_Q2/mnist.mat')
    # print(data['trainX'][0][461])
    image = data['trainX'].transpose()
    # print(data['trainX'].mean(0)[461])
    # print(data['trainX'] - data['trainX'].mean(0))
    adjImg = image - data['trainX'].mean(1)
    # print(adjImg[0][461])
    cov = np.dot(adjImg, adjImg.transpose())
    print(cov.shape)
    w, v = eig(cov)
    print('E-value:', w.real)
    maxIndex = np.where(w.real == max(w.real))
    print(max(w.real))
    print(maxIndex)
    # print('E-vector', v.real.shape)
    v = v.real.transpose()
    print(type(v[0]))
    plt.imshow(v[0].reshape((28,28)), cmap='gray')
    plt.show()
