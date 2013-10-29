from skimage import filter
from scipy import ndimage
import numpy as np
import random


def makeSegmentKernel(kernelShape, edge=1):

    K = np.ones(kernelShape, dtype=np.float)
    K[edge: -edge, edge:-edge] *= -1
    K[K > 0] /= (K > 0).sum()
    K[K < 0] /= (K < 0).sum()

    return K

def getMask(mShape, center, angle, threshold=1.0):

    uhat = np.array((np.sin(angle), np.cos(angle)))
    M = np.zeros(mShape, dtype=np.bool)
    aa, bb = center
    for a in range(mShape[0]):
        for b in range(mShape[1]):
            v = (a - aa, b - bb)
            if (np.linalg.norm(v - np.dot(v, uhat) * uhat) < threshold):
                M[a, b] = True
    return M


def getSegmentOrientation(imSlice, degreeStep=np.pi / 45, threshold=1.0):

    center = [d / 2 for d in imSlice.shape]
    angle = 0
    bestAngle = None
    bestValue = None

    while angle < np.pi:

        M = getMask(imSlice.shape, center, angle, threshold)
        v = imSlice[M].mean() / imSlice[M].std()
        if bestValue is None or v > bestValue:
            bestAngle = angle
            bestValue = v
            bestM = M

        angle += degreeStep

    return bestAngle


def getCenterAndRotation(I, orientationSliceSize=10):

    D1, D2 = random.choice(zip(np.where(I == I.max())))

    imSlice = I[D1 - orientationSliceSize: D1 + orientationSliceSize + 1,
                D2 - orientationSliceSize: D2 + orientationSliceSize + 1]

    a = getSegmentOrientation(imSlice)

    return (D1, D2), a


def getNeighbourKernel(segmentKernelSize, rotation, springCost, rotationCost):

    pass

def getNeighbourSuggestion(modelList, I):

    pass

gs = np.load("dev/gs.npy")

#Edgekernel
K = np.ones((4, 6))
K[:2] = -1

#Segment kernel
R = makeSegmentKernel((28, 61), edge=2)

#Vertical features
V = np.abs(ndimage.convolve(gs.astype(np.float), K.T))
V = ndimage.binary_erosion(np.logical_and(V < np.mean(V) * 15, V > np.mean(V) * 1.5))

#Horizontal features
H = filter.median_filter(ndimage.binary_erosion(ndimage.convolve(gs.astype(np.float), K) > 0), 3)

#Composite features
D = ndimage.convolve((V + H).astype(np.float), R)
