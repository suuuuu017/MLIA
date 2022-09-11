import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import linalg as LA
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv


def gaussian_noise(img_gray):
    row, col = img_gray.shape
    mean = 0
    var = 0.01
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (row, col))
    noisy_img = img_gray + gaussian
    plt.figure()
    plt.title('Noisy Image')
    plt.imshow(noisy_img, cmap='gray')
    plt.show()

    return noisy_img

def forward_difference(image):
    rows, cols = image.shape
    d = np.zeros((rows, cols))
    d[:, 1:cols - 1] = image[:, 1:cols - 1] - image[:, 0:cols - 2];
    d[:, 0] = image[:, 0] - image[:, cols - 1];
    return d


def backward_difference(image):
    rows, cols = image.shape
    d = np.zeros((rows, cols))
    d[1:rows - 1, :] = image[1:rows - 1, :] - image[0:rows - 2, :];
    d[0, :] = image[0, :] - image[rows - 1, :];
    return d

def energy(noisy, clear, lam):
    l2 = -2 * lam * (noisy - clear)
    print(clear)
    # div = -1 * ((backward_difference(clear)) + (forward_difference(clear)))
    # div of dir
    # div = -1 * (backward_difference(backward_difference(clear)) + forward_difference(forward_difference(clear)))
    reg = cv2.Laplacian(clear, cv2.CV_64F, ksize=3)
    # norm
    # dummy = ((backward_difference(clear)) + (forward_difference(clear)))
    # norm = np.sqrt(np.sum(dummy * dummy))
    # print(dummy)
    # div = div / norm
    # print(div)
    # print(norm)
    # print(reg)
    return l2 - reg


if __name__ == '__main__':
    # img = cv2.imread('lena.jpg')
    # img_gray = rgb2gray(img)
    # plt.title('Input Image')
    # plt.imshow(img, cmap='gray')
    # plt.show()

    ## read image
    img = cv2.imread('lena.jpg')
    img_gray = rgb2gray(img)

    noisyImage = gaussian_noise(img_gray)
    print(noisyImage)

    # ## plot the image
    # plt.figure()
    # plt.title('Input Image')
    # plt.imshow(img_gray, cmap='gray')
    # plt.show()
    #
    # ## calculate forward difference and plot
    # forward_diff_img = forward_difference(img_gray)
    # plt.figure()
    # plt.title('Forward Difference')
    # plt.imshow(forward_diff_img, cmap='gray')
    # plt.show()
    #
    # ## calculate backward difference and plot
    # backward_diff_img = backward_difference(img_gray)
    # plt.figure()
    # plt.title('Backward Difference')
    # plt.imshow(backward_diff_img, cmap='gray')
    # plt.show()

    u = noisyImage.copy()

    stepSize = 0.01

    lam = 0.8

    loss = np.ones(noisyImage.shape)

    iter = 0

    while iter < 100:
        loss = energy(noisyImage, u, lam)
        l = np.linalg.norm(loss)
        if l < 0.001:
            break
        print(l)
        newU = u - stepSize * loss
        u = newU
        iter = iter + 1

    plt.imshow(u, cmap='gray')
    plt.show()