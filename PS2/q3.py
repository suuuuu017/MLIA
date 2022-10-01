import numpy as np
from numpy.linalg import eig
from scipy import io
import numpy as np
from numpy.linalg import eig
from matplotlib import pyplot as plt
import SimpleITK as sitk
from scipy import ndimage

def x_difference(image):
    rows, cols = image.shape
    d = np.zeros((rows, cols))

    d1 = np.zeros((rows,cols))
    d1[:,1:cols-1] = image[:,1:cols-1] - image[:,0:cols-2]
    d1[:,0] = image[:,0] - image[:,cols-1]

    d2 = np.zeros((rows,cols))
    d2[:,0:cols-2] = image[:, 0:cols-2] - image[:, 1:cols-1]
    d2[:, cols-1] = image[:, cols-1] - image[:,0]

    #TODO: check if /2 if correct
    d = (d1 + d2) / 2

    return d

def y_difference(image):
    rows, cols = image.shape
    d = np.zeros((rows,cols))

    d1 = np.zeros((rows,cols))
    d1[1:rows-1, :] = image[1:rows-1, :] - image[0:rows-2, :]
    d1[0,:] = image[0,:] - image[rows-1,:]

    d2 = np.zeros((rows, cols))
    d2[0:rows-2, :] = image[0:rows-2, :] - image[1:rows-1, :]
    d2[rows-1, :] = image[rows-1, :] - image[0, :]

    d = (d1 + d2) / 2
    return d

def jacobian(vx, vy):
    topleft = x_difference(vx)
    topright = y_difference(vx)
    bottomleft = x_difference(vy)
    bottomright = y_difference(vy)
    top = np.concatenate((topleft, topright), axis = 1)
    bottom = np.concatenate((bottomleft, bottomright), axis = 1)
    dv = np.concatenate((top, bottom), axis = 0)
    dvt = dv.transpose()

    return dvt

def div(vx, vy):
    top = x_difference(np.multiply(vx, vx)) + x_difference(np.multiply(vx, vy))
    bottom = y_difference(np.multiply(vy, vx)) + y_difference((np.multiply(vy, vy)))
    divM = np.concatenate((top, bottom), axis = 0)
    return divM

if __name__ == '__main__':
    velocity = sitk.GetArrayFromImage(sitk.ReadImage('code+data_Q3/data/initialV/v0Spatial.mhd'))
    source = sitk.GetArrayFromImage(sitk.ReadImage('code+data_Q3/data/sourceImage/source.mhd'))
    print(source.shape)
    vx0 = velocity[0][:, :, 0]
    vy0 = velocity[0][:, :, 1]
    source = source[0]
    # print("source shape is", source.shape)
    # plt.imshow(vx0, cmap='gray')
    # plt.show()
    # print(vx0)

    iter = 0

    vx = vx0
    vy = vy0

    phi = np.arange(10000.).reshape((100, 100))

    stepSize = 0.01

    selected = np.zeros((100, 100))

    while iter < 100:
        iter = iter + 1
        #TODO: add smoothing K
        # print(jacobian(vx, vy).shape)
        # print(np.concatenate((vx, vy), axis = 0).shape)
        #TODO: dot product
        selected[100 // 2 - 8: 100 // 2 + 8, 100 // 2 - 8: 100 // 2 + 8] = 1

        dvdt = -1 * (np.dot(jacobian(vx, vy), np.concatenate((vx, vy), axis=0)) + div(vx, vy))
        # with np.printoptions(threshold=np.inf):
        #     print(dvdt)
        # plt.imshow(dvdt, cmap='gray')
        # plt.show()

        # plt.imshow(np.concatenate((vx, vy), axis=0), cmap='gray')
        # plt.show()

        dvxdt = dvdt[0:100, :]
        dvydt = dvdt[100:200, :]
        # with np.printoptions(threshold=np.inf):
        #     print(dvxdt)
        # plt.imshow(dvxdt, cmap='gray')
        # plt.show()

        filteredSpec = selected * dvxdt
        freqImg = np.fft.fft2(filteredSpec)
        freqImg = np.fft.fftshift(filteredSpec)
        cleanImg = np.fft.ifftshift(selected * freqImg)
        cleanImg = np.fft.ifft2(cleanImg)
        dvxdt = np.real(cleanImg)

        filteredSpec = selected * dvydt
        freqImg = np.fft.fft2(filteredSpec)
        freqImg = np.fft.fftshift(filteredSpec)
        cleanImg = np.fft.ifftshift(selected * freqImg)
        cleanImg = np.fft.ifft2(cleanImg)
        dvydt = np.real(cleanImg)

        # print(dvydt)

        dphidt = ndimage.map_coordinates(phi, [vx, vy], order=3)
        # with np.printoptions(threshold=np.inf):
        #     print(phi)

        vx = dvxdt * stepSize + vx
        vy = dvydt * stepSize + vy

        # print("dpidt size is ", dphidt.shape)
        # print("phi size is ", phi.shape)

        phi = dphidt * stepSize + phi
        ndimage.map_coordinates(phi, [vx, vy], order=3)

    fig, axs = plt.subplots(1, 3)
    print(phi.shape)
    axs[0].imshow(source, cmap='gray')
    axs[1].imshow(np.rint(phi), cmap='gray')

    newimg = np.zeros((100, 100))

    for x in range(100):
        for y in range(100):
            newCoor = int(phi[x][y])
            # print(newCoor)
            newX = newCoor//100
            newY = newCoor%100
            if newX>=100:
                newX = 99
            if newY>=100:
                newY = 99
            newimg[newX][newY] = source[x][y]

    axs[2].imshow(newimg, cmap='gray')
    plt.show()



