import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np

def calculateTan(lastT, lastY, stepSize):
    tan = 2 - math.exp(-4 * lastT) - 2 * lastY
    y = lastY + tan * stepSize
    return y

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for stepSize in [0.1, 0.05, 0.01, 0.005, 0.001]:
        lastT = 0.0
        lastY = 1.0
        T = []
        Y = []
        for t in np.arange(0.0, 6.0, stepSize):
            print("last T, last Y ", lastT, lastY)
            T.append(lastT)
            Y.append(lastY)
            newY = calculateTan(lastT, lastY, stepSize)
            lastY = newY
            lastT = lastT + stepSize

        plt.plot(T, Y, color ="green")
        plt.show()