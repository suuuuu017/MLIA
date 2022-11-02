from scipy import io
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg
import math
import cv2
import random

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
            newImg = np.append(image[idx], 1)
            filterData.append(newImg)

            if x == 6:
                mLabel = 0
            else:
                mLabel = 1
            filterLabel.append(mLabel)
    # filterLabel = np.array(filterLabel)
    # filterData = np.array(filterData)

    # len = filterLabel.shape[0]
    len = len(filterLabel)
    s = math.ceil(600/len)

    augLabel = []
    augData = []
    for idx, x in enumerate(label):
        if x == 6 or x == 8:
            # print(image[idx])

            # data augmentation
            plainimage = image[idx].reshape((28, 28))
            # print(plainimage.shape)
            # plt.imshow(plainimage, cmap='gray')
            # plt.show()

            for i in range(s):
                center = (14, 14)

                ang = random.randint(0, 60)
                # ang = 0

                rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=ang, scale=1)
                rotated_image = cv2.warpAffine(src=plainimage, M=rotate_matrix, dsize=(28, 28))
                # plt.imshow(rotated_image, cmap='gray')
                # plt.show()

                tx, ty = random.randint(-5, 5), random.randint(-5, 5)
                # TODO: someting wrong with translation
                # tx, ty = 0, 0
                translation_matrix = np.array([
                    [1, 0, tx],
                    [0, 1, ty]
                ], dtype=np.float32)
                translated_image = cv2.warpAffine(src=rotated_image, M=translation_matrix, dsize=(28, 28))
                # plt.imshow(translated_image, cmap='gray')
                # plt.show()
                augimage = translated_image.reshape((1, 784))
                newImg = np.append(augimage, 1)

                augData.append(newImg)


                if x == 6:
                    mLabel = 0
                else:
                    mLabel = 1

                augLabel.append(mLabel)
                print(mLabel)

    augData = np.array(augData)
    augLabel = np.array(augLabel)

    list1 = []
    list2 = []
    c = list(zip(augData, augLabel))

    sampleN = 600
    for a, b in random.sample(c, sampleN - len):
        filterData.append(a)
        filterLabel.append(b)

    # list1 = np.array(list1)
    # list2 = np.array(list2)

    filterData = np.array(filterData)
    filterLabel = np.array(filterLabel)



    print("filtered data size is", filterData.shape)
    print("augLabel is", filterLabel.shape)
    input()

    # gradient descent
    beta = np.random.rand(785, 1)
    # print(beta)

    stepsize = 0.01

    # TODO: what should sigma be
    # sig = 1
    sig = 4 # for 600 data

    for iter in range(150):
        oldNorm = np.linalg.norm(beta)
        beta = beta - stepsize * energy(filterLabel, beta, filterData, sig, filterData.shape[0])
        print(np.linalg.norm(beta))
        # if oldNorm < np.linalg.norm(beta):
        #     break

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
            # print(testImage[idx])
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
