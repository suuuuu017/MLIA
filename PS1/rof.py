import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv

if __name__ == '__main__':
    def gaussian_noise(img_gray):
        row, col = img_gray.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        sigma = 0.01
        gaussian = np.random.normal(mean, sigma, (row, col))
        noisy_img = img_gray + gaussian
        plt.figure()
        plt.title('Noisy Image')
        plt.imshow(noisy_img, cmap='gray')


    img = cv2.imread('lena.jpg')
    img_gray = rgb2gray(img)
    plt.title('Input Image')
    plt.imshow(img, cmap='gray')
    plt.show()

    gaussian_noise(img_gray)

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


    ## read image
    img = cv2.imread('lena.jpg')
    img_gray = rgb2gray(img)

    ## plot the image
    plt.figure()
    plt.title('Input Image')
    plt.imshow(img_gray, cmap='gray')
    plt.show()

    ## calculate forward difference and plot
    forward_diff_img = forward_difference(img_gray)
    plt.figure()
    plt.title('Forward Difference')
    plt.imshow(forward_diff_img, cmap='gray')
    plt.show()

    ## calculate backward difference and plot
    backward_diff_img = backward_difference(img_gray)
    plt.figure()
    plt.title('Backward Difference')
    plt.imshow(backward_diff_img, cmap='gray')
    plt.show()
