import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv


if __name__ == '__main__':
    # read the image
    # img = cv2.imread('lenaNoise.png', 0)
    img = cv2.imread('lenaNoise.png')
    img = rgb2gray(img)
    plt.imshow(img, cmap='gray')
    plt.show()

    # fft and shift
    freqImg = np.fft.fft2(img)
    freqImg = np.fft.fftshift(freqImg)

    # sepc of img
    absFreq = np.abs(freqImg)
    spec = np.log(1 + absFreq)
    plt.imshow(spec, cmap='gray')
    plt.show()

    ran = [5, 10, 20, 40, 256]
    # selection matrix
    selected = np.zeros(freqImg.shape)
    for n in ran:
        selected[512//2 - n : 512//2 + n, 512//2 - n : 512//2 + n] = 1
        print(selected)

        # filtered spec of img
        filteredSpec = selected * spec
        plt.imshow(filteredSpec, cmap='gray')
        plt.show()

        # filtered freq
        # inverse shift and inverse fft
        print(selected * freqImg)
        cleanImg = np.fft.ifftshift(selected * freqImg)
        cleanImg = np.fft.ifft2(cleanImg)
        print(cleanImg)
        cleanImg = np.real(cleanImg)
        plt.imshow(cleanImg, cmap='gray')
        plt.show()