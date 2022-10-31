from scipy import io
import numpy as np
from matplotlib import pyplot as plt
import math

def energy(label, beta, image, sig):
    # firstPart = label - 1
    # secondPart = math.exp(-1 * np.dot(image, beta)) / (1 + math.exp(-1 * np.dot(image, beta)))
    # thirdPart = beta / sigma

    deri = np.zeros((784, 1))

    for p in range(784):
        sum = 0
        for i in range(85):
            sum = sum + ((label[i] - 1) + math.exp(-1 * np.dot(image[i], beta))
                  / (1 + math.exp(-1 * np.dot(image[i], beta)))) * image[i][p]
        sum = sum - beta[p] / sig
        deri[p] = sum

    deri = deri * -1

    return deri


if __name__ == '__main__':
    data = io.loadmat('MINIST_Q2/mnist.mat')
    # the keys and values in data dict
    for key, _ in data.items():
        print(key)
    image = data['trainX']
    label = data['trainY']
    print(image.shape)

    # flatten the label
    label = np.squeeze(label)
    print("data label is", np.squeeze(label)[0])

    # filter out 0,1 data and label
    filterData = []
    for idx, x in enumerate(label):
        if x == 0 or x == 1:
            print(image[idx])
            input()
            filterData.append(image[idx])
    filterData = np.array(filterData)
    print("filtered data size is", filterData.shape)

    # gradient descent
    beta = np.random.rand(784, 1)
    # print(beta)

    stepsize = 0.01

    # TODO: what should sigma be
    sig = 0.001

    for iter in range(500):
        beta = beta + stepsize * energy(label, beta, filterData, sig)
        print(beta)
