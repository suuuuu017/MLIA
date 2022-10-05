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
    # plt.imshow(image[:][0].reshape((28, 28)), cmap='gray')
    # plt.show()
    adjImg = image - data['trainX'].mean(1)
    # print(adjImg[0][461])
    # TODO: check np.cov
    # cov = np.cov(adjImg)
    cov = np.dot(adjImg, adjImg.transpose()) / 470.0
    # print(cov.shape)
    # TODO: eig or svd
    w, v = eig(cov)
    print('E-value:', w.real)
    maxIndex = np.where(w.real == max(w.real))
    print(max(w.real))
    print(maxIndex)
    # print('E-vector', v.real.shape)
    v = v.real.transpose()
    print(type(v[0]))

    fig, axs = plt.subplots(1, 10)
    for i in range(10):
        axs[i].imshow(v[i].reshape((28,28)), cmap='gray')
    # plt.imshow(v[0].reshape((28,28)), cmap='gray')
    plt.show()
