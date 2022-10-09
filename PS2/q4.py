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
    d[:, 1:cols - 2] = image[:, 2:cols-1] - image[:, 0:cols-3]
    return d/2

def y_difference(image):
    rows, cols = image.shape
    d = np.zeros((rows,cols))
    d[1:(rows - 2), :] = image[2:(rows-1), :] - image[0:(rows-3), :]
    return d/2

def jacobian(vx, vy):
    topleft = x_difference(vx)
    topright = y_difference(vx)
    bottomleft = x_difference(vy)
    bottomright = y_difference(vy)
    dvxt = np.multiply(topleft, vx) + np.multiply(bottomleft, vy)
    dvyt = np.multiply(topright, vx) + np.multiply(bottomright, vy)

    return dvxt, dvyt

def div(vx, vy):
    top = x_difference(np.multiply(vx, vx)) + x_difference(np.multiply(vx, vy))
    bottom = y_difference(np.multiply(vy, vx)) + y_difference((np.multiply(vy, vy)))
    return top, bottom

def getdphi(vx, vy, phix, phiy):
    dphixdt = np.multiply(x_difference(phix), vx) + np.multiply(y_difference(phix), vy)
    dphiydt = np.multiply(x_difference(phiy), vx) + np.multiply(y_difference(phiy), vy)
    return -1 * dphixdt, -1 * dphiydt


if __name__ == '__main__':
    velocity = sitk.GetArrayFromImage(sitk.ReadImage('code+data_Q3/data/initialV/v0Spatial.mhd'))
    source = sitk.GetArrayFromImage(sitk.ReadImage('code+data_Q3/data/sourceImage/source.mhd'))
    print(source.shape)
    vx0 = velocity[0][:, :, 0]
    vy0 = velocity[0][:, :, 1]
    source = source[0]

    iter = 0

    vx = vx0
    vy = vy0

    phi = np.mgrid[0:100, 0:100]
    phix = phi[0]
    phiy = phi[1]

    stepSize = 0.25

    selected = np.zeros((100, 100))

    while iter < 4:
        iter = iter + 1
        selected[100 // 2 - 8: 100 // 2 + 8, 100 // 2 - 8: 100 // 2 + 8] = 1

        dvxt, dvyt = jacobian(vx, vy)
        top, bottom = div(vx, vy)
        dvxdt = (dvxt + top)
        dvydt = (dvyt + bottom)

        freqImg = np.fft.fft2(dvxdt)
        freqImg = np.fft.fftshift(freqImg)
        cleanImg = np.fft.ifftshift(selected * freqImg)
        cleanImg = np.fft.ifft2(cleanImg)
        dvxdt = np.real(cleanImg)
        dvxdt = 1 * dvxdt

        freqImg = np.fft.fft2(dvydt)
        freqImg = np.fft.fftshift(freqImg)
        cleanImg = np.fft.ifftshift(selected * freqImg)
        cleanImg = np.fft.ifft2(cleanImg)
        dvydt = np.real(cleanImg)
        dvydt = 1 * dvydt

        vx = dvxdt * stepSize + vx
        vy = dvydt * stepSize + vy

        dphixdt, dphiydt = getdphi(vx, vy, phix, phiy)

        phix = dphixdt * stepSize + phix
        phiy = dphiydt * stepSize + phiy

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(source, cmap='gray')
    newimg = ndimage.map_coordinates(source, [phix, phiy], order=3)
    axs[1].imshow(newimg, cmap='gray')
    plt.show()



