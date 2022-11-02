from scipy import io
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg
import math

def energy(label, beta, image, sig, dataSize):
    # firstPart = label - 1
    # secondPart = math.exp(-1 * np.dot(image, beta)) / (1 + math.exp(-1 * np.dot(image, beta)))
    # thirdPart = beta / sigma

    deri = np.zeros((785, 1))

    for p in range(785):
        sum = 0
        for i in range(dataSize):
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

    # flatten the label
    label = np.squeeze(label)
    print("data label is", np.squeeze(label)[0])

    # filter out 0,1 data and label
    filterData = []
    filterLabel = []
    for idx, x in enumerate(label):
        if x == 6 or x == 8:
            print(image[idx])
            # input()
            newImg = np.append(image[idx], 1)
            # print(newImg)
            # input()
            filterData.append(newImg)
            if x == 6:
                mLabel = 0
            else:
                mLabel = 1
            filterLabel.append(mLabel)
    filterData = np.array(filterData)
    filterLabel = np.array(filterLabel)
    print("filtered data size is", filterData.shape)

    # gradient descent
    beta = np.random.rand(785, 1)
    # print(beta)

    stepsize = 0.1

    # TODO: what should sigma be
    sig = 1

    while True:
        d = energy(filterLabel, beta, filterData, sig, filterData.shape[0])
        if np.linalg.norm(d) < 0.1:
            break
        beta = beta - stepsize * energy(filterLabel, beta, filterData, sig, filterData.shape[0])
        print(np.linalg.norm(d))

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
        if x == 6 or x == 8:
            print(testImage[idx])
            # input()
            newImg = np.append(testImage[idx], 1)
            # print(newImg)
            # input()
            filterTestData.append(newImg)
            if x == 6:
                mLabel = 0
            else:
                mLabel = 1
            filterTestLabel.append(mLabel)
    filterTestData = np.array(filterTestData)
    filterTestLabel = np.array(filterTestLabel)

    acc = 0
    for i in range(filterTestData.shape[0]):
        l = 1 / (1 + math.exp(-1 * np.dot(filterTestData[i], beta)))
        if l > 0.5:
            l = 1
        else:
            l = 0
        if l == filterTestLabel[i]:
            acc = acc + 1

    print("acc is ", acc / 116)
