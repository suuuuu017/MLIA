from scipy import io
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg
import math
from numpy.linalg import svd

def energy(label, beta, image, sig, pcaN):
    # firstPart = label - 1
    # secondPart = math.exp(-1 * np.dot(image, beta)) / (1 + math.exp(-1 * np.dot(image, beta)))
    # thirdPart = beta / sigma

    deri = np.zeros((pcaN, 1))

    for p in range(pcaN):
        sum = 0
        for i in range(85):
            sum = sum + ((1 - label[i]) - math.exp(-1 * np.dot(image[i], beta))
                  / (1 + math.exp(-1 * np.dot(image[i], beta)))) * image[i][p]
        sum = sum + beta[p] / sig
        deri[p] = sum

    # deri = deri * -1

    return deri


if __name__ == '__main__':
    data = io.loadmat('MINIST_Q2/mnist.mat')
    # the keys and values in data dict
    for key, _ in data.items():
        print(key)
    input()
    image = data['trainX']
    label = data['trainY']
    print(image.shape)

    # pca element number
    pcaN = 10

    # flatten the label
    label = np.squeeze(label)
    print("data label is", np.squeeze(label)[0])

    # filter out 0,1 data and label
    filterData = []
    filterLabel = []
    for idx, x in enumerate(label):
        if x == 0 or x == 1:
            print(image[idx])
            # input()
            newImg = np.append(image[idx], 1)
            # print(newImg)
            # input()
            filterData.append(newImg)
            filterLabel.append(x)
    filterData = np.array(filterData)
    filterLabel = np.array(filterLabel)
    print("filtered data size is", filterData.shape)
    input()

    adjData = filterData - filterData.mean(0)
    adjData = adjData.transpose()
    cov = np.dot(adjData, adjData.transpose()) / (adjData.shape[0] - 1)
    _, w, v = svd(cov)
    v = v.real
    print("eigen vector size is", v.shape)
    input()

    pcaVec = v[0:pcaN, :]
    print("pcaVec size is", pcaVec.shape)
    input()

    pcaImg = np.dot(filterData, pcaVec.T)
    print("pca Img size is", pcaImg.shape)
    input()

    # gradient descent
    beta = np.random.rand(pcaN, 1)
    # print(beta)

    stepsize = 0.01

    # TODO: what should sigma be
    sig = 1

    for iter in range(50):
        beta = beta - stepsize * energy(filterLabel, beta, pcaImg, sig, pcaN)
        print(np.linalg.norm(beta))

    # predict
    testImage = data['testX']
    testLabel = data['testY']

    # flatten the label
    testLabel = np.squeeze(testLabel)
    print("testLabel label is", np.squeeze(testLabel)[0])

    # filter out 0,1 data and label
    filterTestData = []
    filterTestLabel = []
    for idx, x in enumerate(testLabel):
        if x == 0 or x == 1:
            # print(testImage[idx])
            # input()
            newImg = np.append(testImage[idx], 1)
            # print(newImg)
            # input()
            filterTestData.append(newImg)
            filterTestLabel.append(x)
    filterTestData = np.array(filterTestData)
    filterTestLabel = np.array(filterTestLabel)

    pcaTestData = np.dot(filterTestData, pcaVec.T)

    acc = 0
    for i in range(115):
        l = 1 / (1 + math.exp(-1 * np.dot(pcaTestData[i], beta)))
        if l > 0.5:
            l = 1
        else:
            l = 0
        if l == filterTestLabel[i]:
            acc = acc + 1

    print("acc is ", acc / 116)
