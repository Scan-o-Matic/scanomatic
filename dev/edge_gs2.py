from skimage import filter
from scipy import ndimage
from scipy import stats
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
            #bestM = M

        angle += degreeStep

    return bestAngle


def getCenterAndRotation(I, orientationSliceSize=10, position=None):

    if position is not None:
        D1, D2 = position
    else:
        D1, D2 = random.choice(zip(np.where(I == I.max())))

    imSlice = I[D1 - orientationSliceSize: D1 + orientationSliceSize + 1,
                D2 - orientationSliceSize: D2 + orientationSliceSize + 1]

    a = getSegmentOrientation(imSlice)

    return (D1, D2), a


def getNeighbourKernel(segmentDistance, rotationAngle,
                       springCost=2,
                       rotationCost=0.25,
                       kernelExtensionFactor=1.25):

    def vAngle(v1, v2):

        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)

        angle = np.arccos(np.dot(v1_u, v2_u))

        if np.isnan(angle):
            if (v1_u == v2_u).all():
                return 0.0
            else:
                return np.pi
        return angle

    Nspring = stats.norm(scale=springCost)
    Nrotation = stats.norm(scale=rotationCost)

    refVector = np.array((np.sin(rotationAngle), np.cos(rotationAngle)))

    Ksize = [int(np.round(2 * v * kernelExtensionFactor)) for v in
             (segmentDistance, segmentDistance)]
    Ksize = [v % 2 == 1 and v or v + 1 for v in Ksize]
    Kernel = np.zeros(Ksize, dtype=np.float)
    c1, c2 = [v / 2 for v in Ksize]

    for i in range(Ksize[0]):
        for j in range(Ksize[1]):

            if (i != c1 or j != c2):
                Kernel[i, j] = (Nspring.pdf(np.linalg.norm((i - c1, j - c2)) -
                                            segmentDistance) *
                                Nrotation.pdf(vAngle((i - c1, j - c2),
                                                     refVector)))

    return Kernel / Kernel.max()


def getNeighbourSuggestion(modelList, I, segmentDistance,
                           upCandidate=None, downCandidate=None):

    def getBestCandidate(I, currentPos, kernel):

        kSize = np.array(kernel.size)
        kCenter = (kSize - 1) / 2

        lSlice = [slice(currentPos[0] + kCenter - kSize),
                  slice(currentPos[1] + kCenter + kSize)]
        sSlice = [slice(None, None), slice(None, None)]

        for i in range(2):
            if lSlice[i].start < 0:
                sSlice[i] = slice(abs(lSlice[i].start), None)
                lSlice[i] = slice(None, lSlice[i].end)
            if lSlice[i].end > I.shape[i]:
                sSlice[i] = slice(sSlice[i].start, lSlice[i].end - I.shape[i])
                lSlice[i] = slice(lSlice[i].start, None)

        testI = I[lSlice] * kernel[sSlice]
        relPos = kCenter - np.array(
            [v[0] for v in np.where(testI == testI.max())])
        return tuple(currentPos + relPos), testI.max()

    upAngle = modelList[0][1] + np.pi
    upOrigin = modelList[0][0]
    downAngle = modelList[-1][1]
    downOrigin = modelList[-1][0]

    if upCandidate is None:
        upK = getNeighbourKernel(segmentDistance, upAngle)
        upNewPos, upNewPosVal = getBestCandidate(I, upOrigin, upK)
    else:
        upNewPos, upAngle, upNewPosVal = upCandidate

    if downCandidate is None:
        downK = getNeighbourKernel(segmentDistance, downAngle)
        downNewPos, downNewPosVal = getBestCandidate(I, downOrigin, downK)
    else:
        downNewPos, downAngle, downNewPosVal = downCandidate

    if upNewPosVal > downNewPosVal:
        newPos = (upNewPos, upAngle, upNewPosVal)
        altPos = (downNewPos, downAngle, downNewPosVal)
    else:
        newPos = (downNewPos, downAngle, downNewPosVal)
        altPos = (upNewPos, upAngle, upNewPosVal)

    return newPos, altPos

"""
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
"""
