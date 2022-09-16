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
    # sigma = 0.01
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
    # magx = np.sqrt(np.sum(np.square(x_difference(clear))))
    # magy = np.sqrt(np.sum(np.square(y_difference(clear))))
    # mag = np.sqrt(x_difference(clear) ** 2 + y_difference(clear) ** 2) + 0.0001
    magx = LA.norm(x_difference(clear)) + 0.00001
    magy = LA.norm(y_difference(clear)) + 0.00001
    # div =  (x_difference(clear) / mag + y_difference(clear) / mag)
    # div4 =-1 * (x_difference(div) + y_difference(div))
    # div2 = -1 * (x_difference(clear) / magx + y_difference(clear) / magy)
    # div3 = -1 * (x_difference(x_difference(clear) / magx) + y_difference(y_difference(clear) / magy))
    # innerPart = (x_difference(clear) + y_difference(clear)) / (magx + magy + 0.0001)
    innerPart = (x_difference(clear) / magx + y_difference(clear) / magy)
    div3 = -1 * (x_difference(innerPart) + y_difference(innerPart))
    print('div3 is', div3)
    # div3 = -1 * (x_difference(x_difference(clear)/ magx) + y_difference(y_difference(clear)/ magy))
    # return l2 + div3

    reg = cv2.Laplacian(clear, cv2.CV_64F, ksize=3)
    print("reg is ", reg)

    return l2 - reg

if __name__ == '__main__':

    ## read image
    img = cv2.imread('lena.jpg')
    img_gray = rgb2gray(img)

    noisyImage = gaussian_noise(img_gray)

    u = noisyImage.copy()

    stepSize = 0.1

    lam = 0.05

    loss = np.ones(noisyImage.shape)
    l = 10000000

    iter = 0
    fig, axs = plt.subplots(2,4)

    gradVal = []
    iterRec = []
    # oldL = 0
    while iter < 6:
        # oldL = l
        loss = energy(noisyImage, u, lam)
        l = np.linalg.norm(loss)
        # TODO: change the criteria to not changing much
        if l < 0.0001:
            break
        print(l)
        # print("difference", np.linalg.norm(noisyImage - u))
        newU = u - stepSize * loss
        u = newU
        iter = iter + 1
        gradVal.append(l)
        iterRec.append(iter)

    axs[0,0].imshow(noisyImage, cmap='gray'); axs[0,0].set_title('Original Image')
    axs[0,1].imshow(u, cmap='gray'); axs[0,1].set_title('Denoised Image')
    axs[0,2].imshow(np.abs(noisyImage - u), cmap='gray'); axs[0,2].set_title('Difference')
    axs[0,3].imshow(np.abs(noisyImage - img_gray), cmap='gray'); axs[0, 3].set_title('Noise')

    axs[1,0].imshow(noisyImage[0:400, 0:400], cmap='gray')
    axs[1,0].set_title('Original Image')
    axs[1,1].imshow(u[0:400, 0:400], cmap='gray')
    axs[1,1].set_title('Denoised Image')
    axs[1,2].imshow(np.abs(noisyImage - u)[0:400, 0:400], cmap='gray')
    axs[1,2].set_title('Difference')
    axs[1,3].imshow(np.abs(noisyImage - img_gray)[0:400, 0:400], cmap='gray')
    axs[1,3].set_title('Noise')

    # plt.imshow(u, cmap='gray')
    plt.show()

    plt.plot(iterRec, gradVal)
    plt.show()