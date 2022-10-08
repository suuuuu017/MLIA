import matplotlib.pyplot as plt
import math
import numpy as np

def calculateTan(lastT, lastY, stepSize):
    tan = 2 - math.exp(-4 * lastT) - 2 * lastY
    y = lastY + tan * stepSize
    return y

def isInt(a, b):
    return abs(a-b) < 0.0001

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    interval = [0.1, 0.05, 0.01, 0.005, 0.001]
    i = 0
    labels = ['Step Size = 0.1', 'Step Size = 0.05', 'Step Size = 0.01', 'Step Size = 0.005', 'Step Size = 0.001']
    colors = ['b', 'g', 'y', 'r', 'm']
    for stepSize in interval:
        lastT = 0.0
        lastY = 1.0
        T = []
        Y = []
        Tint = []
        Yint = []
        for t in np.arange(0.0, 6.0, stepSize):
            print("last T, last Y ", lastT, lastY)
            T.append(lastT)
            Y.append(lastY)
            if isInt(lastT, 1) or isInt(lastT, 2) or isInt(lastT, 3) or isInt(lastT, 4) or isInt(lastT, 5):
                Tint.append(lastT)
                Yint.append(lastY)
            newY = calculateTan(lastT, lastY, stepSize)
            lastY = newY
            lastT = lastT + stepSize

        # t = [1.0, 2.0, 3.0, 4.0, 5.0]
        # atcy = []
        # for point in t:
        #     atcy.append(1.0 + 0.5 / math.exp(4.0 * point) - 0.5 / math.exp(2.0 * point))
        # plt.scatter(t, atcy, marker = 'P', color = 'yellow', label= 'Actual Value')
        plt.plot(T, Y, color = colors[i], label=labels[i])
        i = i + 1
        # plt.scatter(Tint, Yint, marker = '.', color = 'red', label='Euler Approximation Value')
        # plt.legend(loc='lower right')

        # print(atcy)
        # print(Yint)
    plt.legend(loc='lower right')
    plt.show()