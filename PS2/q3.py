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
    # top = np.concatenate((topleft, topright), axis = 1)
    # bottom = np.concatenate((bottomleft, bottomright), axis = 1)
    # dv = np.concatenate((top, bottom), axis = 0)
    # dvt = dv.transpose()
    dvxt = np.multiply(topleft, vx) + np.multiply(bottomleft, vy)
    dvyt = np.multiply(topright, vx) + np.multiply(bottomright, vy)

    return dvxt, dvyt

def div(vx, vy):
    top = x_difference(np.multiply(vx, vx)) + x_difference(np.multiply(vx, vy))
    bottom = y_difference(np.multiply(vy, vx)) + y_difference((np.multiply(vy, vy)))
    return top, bottom

if __name__ == '__main__':
    velocity = sitk.GetArrayFromImage(sitk.ReadImage('code+data_Q3/data/initialV/v0Spatial.mhd'))
    source = sitk.GetArrayFromImage(sitk.ReadImage('code+data_Q3/data/sourceImage/source.mhd'))
    print(source.shape)
    vx0 = velocity[0][:, :, 0]
    vy0 = velocity[0][:, :, 1]
    source = source[0]
    # print("source shape is", source.shape)
    plt.imshow(vx0, cmap='gray')
    plt.show()
    # print(vx0)

    iter = 0

    vx = vx0
    vy = vy0

    # phix = np.arange(10000.).reshape((100, 100))
    # phi = source
    phi = np.mgrid[0:100, 0:100]
    phix = phi[0]
    phiy = phi[1]

    stepSize = 0.25

    selected = np.zeros((100, 100))

    while iter < 4:
        iter = iter + 1
        #TODO: add smoothing K
        # print(jacobian(vx, vy).shape)
        # print(np.concatenate((vx, vy), axis = 0).shape)
        #TODO: dot product
        selected[100 // 2 - 8: 100 // 2 + 8, 100 // 2 - 8: 100 // 2 + 8] = 1

        # dvdt = float(-1.0) * (jacobian(vx, vy) + div(vx, vy))
        # with np.printoptions(threshold=np.inf):
        #     print(dvdt)
        # plt.imshow(dvdt, cmap='gray')
        # plt.show()

        # plt.imshow(np.concatenate((vx, vy), axis=0), cmap='gray')
        # plt.show()
        dvxt, dvyt = jacobian(vx, vy)
        top, bottom = div(vx, vy)
        print("dvxt is", dvxt)
        print(top)
        dvxdt = (dvxt + top)
        dvydt = (dvyt + bottom)
        # with np.printoptions(threshold=np.inf):
        #     print(jacobian(vx, vy))

        print("shape of dvxdt is ", dvxdt.shape)

        plt.imshow(dvxdt, cmap='gray')
        plt.show()

        freqImg = np.fft.fft2(dvxdt)
        freqImg = np.fft.fftshift(freqImg)
        cleanImg = np.fft.ifftshift(selected * freqImg)
        cleanImg = np.fft.ifft2(cleanImg)
        dvxdt = np.real(cleanImg)
        dvxdt = -1 * dvxdt


        freqImg = np.fft.fft2(dvydt)
        freqImg = np.fft.fftshift(freqImg)
        cleanImg = np.fft.ifftshift(selected * freqImg)
        cleanImg = np.fft.ifft2(cleanImg)
        dvydt = np.real(cleanImg)
        dvydt = -1 * dvydt

        # print(dvydt)

        plt.imshow(dvxdt, cmap='gray')
        plt.show()


        # v = np.concatenate((vx, vy), axis=0)

        # with np.printoptions(threshold=np.inf):
        #     print(phi)

        vx = dvxdt * stepSize + vx
        vy = dvydt * stepSize + vy

        # dphidt = ndimage.map_coordinates(phi, [vx, vy], order=3)

        # dphixdt = ndimage.map_coordinates(phix, [vx, vy], order=3)
        # dphiydt = ndimage.map_coordinates(phiy, [vx, vy], order=3)

        dphixdt = ndimage.map_coordinates(vx, phi, order=3)
        dphiydt = ndimage.map_coordinates(vy, phi, order=3)

        plt.imshow(dphixdt, cmap='gray')
        plt.show()


        # print("dpidt size is ", dphidt.shape)
        # print("phi size is ", phi.shape)

        phix = dphixdt * stepSize + phix
        phiy = dphiydt * stepSize + phiy
        # ndimage.map_coordinates(phi, [vx, vy], order=3)

        # plt.imshow(dvdt, cmap='gray')
        # plt.show()
        # plt.imshow(dvxdt, cmap='gray')
        # plt.show()
        # plt.imshow(dvydt, cmap='gray')
        # plt.show()

    fig, axs = plt.subplots(1, 3)
    print(phi.shape)
    axs[0].imshow(source, cmap='gray')
    # axs[1].imshow(source - phi, cmap='gray')

    # newimg = np.zeros((100, 100))
    #
    # for x in range(100):
    #     for y in range(100):
    #         # newCoor = int(phi[x][y])
    #         # print(newCoor)
    #         newX = phix[x][y]
    #         newY = phiy[x][y]
    #         # if newX>=100:
    #         #     newX = 99
    #         # if newY>=100:
    #         #     newY = 99
    #         newimg[newX][newY] = source[x][y]


    newimg = ndimage.map_coordinates(source, [phix, phiy], order=3)

    # for x in range(100):
    #     for y in range(100):
    #         # newCoor = int(phi[x][y])
    #         # print(newCoor)
    #         newX = phix[x][y]
    #         newY = phiy[x][y]
    #         # if newX>=100:
    #         #     newX = 99
    #         # if newY>=100:
    #         #     newY = 99
    #         print(newX, newY)
    #         newimg[int(newX)][int(newY)] = source[x][y]
    axs[1].imshow(newimg, cmap='gray')
    # axs[1].imshow(phix, cmap='gray')
    axs[2].imshow(source - newimg, cmap='gray')
    plt.show()



