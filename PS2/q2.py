import numpy as np
from numpy.linalg import eig
from scipy import io
import numpy as np
from numpy.linalg import svd, eig
from matplotlib import pyplot as plt

if __name__ == '__main__':
    data = io.loadmat('MINIST_Q2/mnist.mat')
    image = data['trainX']
    print("data shape is", image.shape)
    print("mean dimension is", data['trainX'].mean(0).shape)
    adjImg = image - data['trainX'].mean(0)
    adjImg = adjImg.transpose()
    cov = np.dot(adjImg, adjImg.transpose()) / 470.0
    print("cov shape is",cov.shape)
    _, w, v = svd(cov)
    print('E-value:', w)
    w = w**2
    v = v.real
    eigenVal = []
    for i in range(w.size):
        eigenVal.append(w[i])
    plt.plot(eigenVal)
    plt.show()

    fig, axs = plt.subplots(1, 10)
    for i in range(10):
        axs[i].imshow(v[i].reshape((28,28)), cmap='gray')
    plt.show()
