import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import linalg as LA
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv


def gaussian_noise(img_gray):
    row, col = img_gray.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    sigma = 0.05
    gaussian = np.random.normal(mean, sigma, (row, col))
    noisy_img = img_gray + gaussian
    plt.figure()
    plt.title('Noisy Image')
    plt.imshow(noisy_img, cmap='gray')
    plt.show()

    return noisy_img

def x_difference(image):
    rows, cols = image.shape
    d = np.zeros((rows,cols))
    d[:,1:cols-1] = image[:,1:cols-1] - image[:,0:cols-2];
    d[:,0] = image[:,0] - image[:,cols-1];
    return d

def y_difference(image):
    rows, cols = image.shape
    d = np.zeros((rows,cols))
    d[1:rows-1, :] = image[1:rows-1, :] - image[0:rows-2, :];
    d[0,:] = image[0,:] - image[rows-1,:];
    return d

def energy(noisy, clear, lam):
    l2 = -2 * lam * (noisy - clear)
    sqrX = np.square(x_difference(clear))
    sqrY = np.square(y_difference(clear))
    devX = x_difference(clear)
    devY = y_difference(clear)
    mag = np.sqrt(sqrX + sqrY)
    mag = mag + 0.0000000001
    div = x_difference(devX / mag) + y_difference(devY / mag)
    # reg = cv2.Laplacian(clear, cv2.CV_64F, ksize=3)
    return l2 - div
    # return l2 - reg

if __name__ == '__main__':

    ## read image
    img = cv2.imread('lena.jpg')
    img_gray = rgb2gray(img)

    noisyImage = gaussian_noise(img_gray)

    u = noisyImage.copy()

    stepSize = 0.1

    lam = 1

    # h = 1

    loss = np.ones(noisyImage.shape)
    l = 10000000

    iter = 0
    fig, axs = plt.subplots(1, 2)

    gradVal = []
    iterRec = []
    oldL = 0
    while iter < 500:
        oldL = l
        loss = energy(noisyImage, u, lam)
        l = np.linalg.norm(loss)
        # TODO: change the criteria to not changing much
        if l < 0.00001:
            break
        print(l)
        # print("difference", np.linalg.norm(noisyImage - u))
        newU = u - stepSize * loss
        u = newU
        iter = iter + 1
        gradVal.append(l)
        iterRec.append(iter)

    axs[0].imshow(noisyImage, cmap='gray'); axs[0].set_title('Original Image')
    axs[1].imshow(u, cmap='gray'); axs[1].set_title('Denoised Image')
    # axs[0,2].imshow(np.abs(noisyImage - u), cmap='gray'); axs[0,2].set_title('Difference')
    # axs[0,3].imshow(np.abs(noisyImage - img_gray), cmap='gray'); axs[0, 3].set_title('Noise')
    #
    # axs[1,0].imshow(noisyImage[0:400, 0:400], cmap='gray')
    # axs[1,0].set_title('Original Image')
    # axs[1,1].imshow(u[0:400, 0:400], cmap='gray')
    # axs[1,1].set_title('Denoised Image')
    # axs[1,2].imshow(np.abs(noisyImage - u)[0:400, 0:400], cmap='gray')
    # axs[1,2].set_title('Difference')
    # axs[1,3].imshow(np.abs(noisyImage - img_gray)[0:400, 0:400] - 1, cmap='gray')
    # axs[1,3].set_title('Noise')

    # plt.imshow(u, cmap='gray')
    plt.show()

    plt.plot(iterRec, gradVal)
    plt.show()