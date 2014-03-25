#
#   DEPENDENCIES
#

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, sobel, laplace, convolve, generic_filter, median_filter

#
#   INTERNAL DEPENDENCIES
#

from scanomatic.dataProcessing.dataBridge import Data_Bridge

#
#   STATIC GLOBALS
#

DEFAULT_CONTROL_POSITION_KERNEL = np.array([[0, 0], [0, 1]], dtype=np.bool)

#
#   METHODS: Math support functions
#


def IQRmean(dataVector):
    """Returns the mean of the inter quartile range of an array of
    any shape (treated as a 1D vector"""

    dSorted = np.sort(dataVector.ravel())
    cutOff = dSorted.size / 4
    return dSorted[cutOff: -cutOff].mean()

#
#   METHODS: Normalisation methods
#


def getDownSampledPlates(dataObject, subSampling="BR"):
    """

        The subsampling is either supplied as a generic position for all plates
        or as a list of individual positions for each plate of the dataBridge
        or object following the same interface (e.g. numpy array of plates).

        The subsampling will attempt to extract one of the four smaller sized
        plates that made up the current plate. E.g. one of the four 384 plates
        that makes up a 1536 as the 384 looked before being pinned over.

        The returned array will work on the same memory as the original so
        any changes will affect the original data.

        The subSampling should be one of the following four expressions:

            TL:     Top left
            TR:     Top right
            BL:     Bottom left
            BR:     Bottom right
    """

    #How much smaller the cut should be than the original
    #(e.g. take every second)
    subSamplingSize = 2

    #Number of dimensions to subsample
    subSampleFirstNDim = 2

    #Lookup to translate subsamplingexpressions to coordinates
    subSamplingLookUp = {'TL': (0, 0), 'TR': (0, 1), 'BL': (1, 0), 'BR': (1, 1)}

    #Generic -> Per plate
    if isinstance(subSampling, str):
        subSampling = [subSampling for i in range(dataObject.shape[0])]

    #Name to offset
    subSampling = [subSamplingLookUp[s] for s in subSampling]

    #Create a new container for the plates. It is important that this remains
    #a list and is not converted into an array if both returned memebers of A
    #and original plate values should operate on the same memory
    A = []
    for i, plate in enumerate(dataObject):
        offset = subSampling[i]
        newShape = (tuple(plate.shape[d] / subSamplingSize for d in
                          xrange(subSampleFirstNDim)) +
                    plate.shape[subSampleFirstNDim:])
        newStrides = (tuple(plate.strides[d] * subSamplingSize for d in
                            xrange(subSampleFirstNDim)) +
                      plate.strides[subSampleFirstNDim:])
        A.append(np.lib.stride_tricks.as_strided(plate[offset[0]:, offset[1]:],
                                                 shape=newShape,
                                                 strides=newStrides))

    return A


def getControlPositionsArray(dataBridge,
                             controlPositionKernel=None,
                             experimentPositionsValue=np.nan):

    """Support method that returns array in the shape corresponding
    to the data in the DataBridge such that only the values reported
    in the control positions are maintained (without affecting the contents
    of the Databridge)."""

    data = dataBridge.getAsArray()
    nPlates = data.shape[0]
    tmpCtrlPosPlateHolder = []

    if controlPositionKernel is None:
        controlPositionKernel = [DEFAULT_CONTROL_POSITION_KERNEL] * nPlates

    for plateIndex in xrange(nPlates):

        tmpPlateArray = data[plateIndex].copy()
        tmpCtrlPosPlateHolder.append(tmpPlateArray)

        controlKernel = controlPositionKernel[plateIndex]
        kernelD1, kernelD2 = controlKernel.shape
        experimentPos = np.array([experimentPositionsValue] *
                                 data[plateIndex].shape[2])

        for idx1 in xrange(tmpPlateArray.shape[0]):

            for idx2 in xrange(tmpPlateArray.shape[1]):

                if controlKernel[idx1 % kernelD1, idx2 % kernelD2]:

                    tmpPlateArray[idx1, idx2] = data[plateIndex][idx1, idx2]

                else:

                    tmpPlateArray[idx1, idx2] = experimentPos

    return np.array(tmpCtrlPosPlateHolder)


def _getPositionsForKernelTrue(dataObject, positionKernels):

    platesCoordinates = []

    for plateIndex in range(len(dataObject)):

        plateCoordinates = [[], []]
        kernel = positionKernels[plateIndex]
        kernelD1, kernelD2 = kernel.shape

        for idx1 in xrange(dataObject[plateIndex].shape[0]):

            for idx2 in xrange(dataObject[plateIndex].shape[1]):

                if kernel[idx1 % kernelD1, idx2 % kernelD2]:

                    plateCoordinates[0].append(idx1)
                    plateCoordinates[1].append(idx2)

        platesCoordinates.append(map(np.array, plateCoordinates))

    return platesCoordinates


def getControlPositionsCoordinates(dataObject, controlPositionKernel=None):
    """Returns list of tuples that emulates the results of running np.where"""

    nPlates = len(dataObject)

    if controlPositionKernel is None:
        controlPositionKernel = [DEFAULT_CONTROL_POSITION_KERNEL] * nPlates

    return _getPositionsForKernelTrue(dataObject, controlPositionKernel)


def getExperimentPosistionsCoordinates(dataObject, controlPositionKernels=None):

    nPlates = len(dataObject)

    if controlPositionKernels is None:
        controlPositionKernels = [DEFAULT_CONTROL_POSITION_KERNEL] * nPlates

    experimentPositionKernels = [k == False for k in controlPositionKernels]

    return _getPositionsForKernelTrue(dataObject, experimentPositionKernels)


def getCoordinateFiltered(dataObject, coordinates, measure=1,
                          requireFinite=True,
                          requireCorrelated=False):

    if isinstance(dataObject, Data_Bridge):
        dataObject.getAsArray()

    filtered = []
    for i in range(len(dataObject)):

        p = dataObject[i][..., measure]
        filteredP = p[coordinates[i]]

        if requireFinite and not requireCorrelated:
            filteredP = filteredP[np.isfinite(filteredP)]

        filtered.append(filteredP)

    filtered = np.array(filtered)

    if requireCorrelated:

        filtered = filtered[:, np.isfinite(filtered).all(axis=0)]

    return filtered


def getCenterTransformedControlPositions(controlPositionCoordinates,
                                         dataObject):

    """Remaps coordinates so they are relative to the plates' center"""

    centerTransformed = []

    if isinstance(dataObject, Data_Bridge):
        dataObject = dataObject.getAsArray()

    for plateIndex, plate in enumerate(controlPositionCoordinates):

        center = dataObject[plateIndex].shape[:2] / 2.0
        centerTransformed.append(
            (plate[0] - center[0],  plate[1] - center[1]))

    return centerTransformed


def getControlPositionsAverage(controlPositionsDataArray,
                               experimentPositionsValue=np.nan,
                               averageMethod=IQRmean):
    """Returns the average per measure of each measurementtype for
    the control positions. Default is to return the mean of the
    inter quartile range"""

    plateControlAverages = []

    for plate in controlPositionsDataArray:

        measureVector = []
        plateControlAverages.append(measureVector)

        for measureIndex in xrange(plate.shape[2]):

            if experimentPositionsValue in (np.nan, np.inf):

                if np.isnan(experimentPositionsValue):

                    expPosTest = np.isnan

                else:

                    expPosTest = np.isinf

                measureVector.append(
                    averageMethod(plate[..., measureIndex][
                        expPosTest(plate[..., measureIndex]) == False]))

            else:

                measureVector.append(
                    averageMethod(plate[..., measureIndex][
                        plate[..., measureIndex] != experimentPositionsValue]))

    return np.array(plateControlAverages)


def getNormalisationSurfaceWithGridData(
        controlPositionsDataArray,
        controlPositionsCoordinates=None,
        normalisationSequence=('cubic', 'linear', 'nearest'),
        useAccumulated=False,
        missingDataValue=np.nan,
        controlPositionKernel=None,
        medianSmoothing=None,
        gaussSmoothing=None):
    """Constructs normalisation surface using iterative runs of
    scipy.interpolate's gridddata based on sequence of supplied
    method preferences.

        controlPositionDataArray
            An array with only control position values intact.
            All other values should be missingDataValue or they won't be
            calculated.

        controlPositionsCoordinates (None)
            Optional argument to supply already constructed
            per plate control positions vector. If not supplied
            it is constructed using controlPositionKernel

        normalisationSequence ('cubic', 'linear', 'nearest')
            The griddata method order to be invoked.

        useAccumulated (False)
            If later stage methods should use information obtained in
            earlier stages or only work on original control positions.

        missingDataValue (np.nan)
            The value to be used to indicate that normalisation value
            for a position is not known

        controlPositionKernel (None)
            Argument passed on when constructing the
            controlPositionsCoordinates if it is not supplied.

    """
    normInterpolations = []
    nPlates = controlPositionsDataArray.shape[0]

    if controlPositionsCoordinates is None:
        controlPositionsCoordinates = getControlPositionsCoordinates(
            controlPositionsDataArray, controlPositionKernel)

    if np.isnan(missingDataValue):

        missingTest = np.isnan

    else:

        missingTest = np.isinf

    for plateIndex in xrange(nPlates):

        points = controlPositionsCoordinates[plateIndex]
        plate = controlPositionsDataArray[plateIndex].copy()
        normInterpolations.append(plate)
        grid_x, grid_y = np.mgrid[0:plate.shape[0], 0:plate.shape[1]]

        for measureIndex in xrange(plate.shape[2]):

            for method in normalisationSequence:

                if useAccumulated:
                    points = np.where(missingTest(plate[..., measureIndex]) ==
                                      False)

                values = plate[..., measureIndex][points]

                finitePoints = np.isfinite(values)
                if (finitePoints == False).any():

                    points = tuple(p[finitePoints] for p in points)
                    values = values[finitePoints]

                res = griddata(
                    tuple(points),
                    #np.array(tuple(p.ravel() for p in points)).T.shape,
                    values,
                    (grid_x, grid_y),
                    method=method,
                    fill_value=missingDataValue)

                accPoints = np.where(
                    missingTest(plate[..., measureIndex]))

                """
                print method
                print "Before:", missingTest(plate[..., measureIndex]).sum()
                print "Change size:", (missingTest(res[accPoints]) == False).sum()
                print "After:", missingTest(res).sum()
                """

                plate[..., measureIndex][accPoints] = res[accPoints]

                """
                print "True After:", missingTest(plate[..., measureIndex]).sum()
                print "--"
                """

                if not missingTest(plate[..., measureIndex]).any():
                    break

            #print "***"

    normInterpolations = np.array(normInterpolations)

    if medianSmoothing is not None:
        for measureIndex in xrange(plate.shape[2]):
            applyMedianSmoothing(
                normInterpolations,
                filterShape=medianSmoothing,
                measure=measureIndex)

    if gaussSmoothing is not None:
        for measureIndex in xrange(plate.shape[2]):
            applyGaussSmoothing(
                normInterpolations,
                sigma=gaussSmoothing,
                measure=measureIndex)

    return normInterpolations

#
#   METHODS: Apply functions
#
#   Apply functions update the dataArray/Bridge values!
#


def applyOutlierFilter(dataArray, nanFillSize=(3, 3), measure=1,
                       k=2.0, p=10, maxIterations=10):
    """Checks all positions in each array and filters those outside
    set boundries based upon their peak/valey properties using
    laplace and normal distribution assumptions.

    Args:
        dataArray (numpy.array):    Array of platewise values

    Kwargs:

        nanFillSize (tuple):    Used in median filter that is nan-safe as
                                first smoothing step before testing outliers.
                                If set to None, step is skipped

        measure (int):  The measure to be outlier filtered

        k (float) : Distance in sigmas for setting nan-threshold

        p (int) :   Estimate number of positions to be qualified as outliers

        maxIterations (int) :   Maximum number of iterations filter may be
                                applied
    """

    def _nanFiller(X):
        #X = X.reshape(nanFillSize)
        if (np.isnan(X[nanFillerKernelCenter])):

            return np.median(X[np.isfinite(X)])

        else:

            return X[nanFillerKernelCenter]

    if nanFillSize is not None:

        nanFillerKernelCenter = (np.prod(nanFillSize) - 1) / 2

        assert np.array([v % 2 == 1 for v in nanFillSize]).all(), (
            "nanFillSize can only have odd values")

    laplaceKernel = np.array([
        [0.5, 1, 0.5],
        [1, -6, 1],
        [0.5, 1, 0.5]], dtype=dataArray[0].dtype)

    oldNans = -1
    newNans = 1
    iterations = 0
    while (newNans != oldNans and iterations < maxIterations):

        oldNans = newNans
        newNans = 0
        iterations += 1

        for plate in dataArray:

            #We need a copy because we'll modify it
            aPlate = plate[..., measure].copy()

            if nanFillSize is not None:

                #Apply median filter to fill nans
                aPlate = generic_filter(aPlate, _nanFiller, size=nanFillSize,
                                        mode="nearest")

            #Apply laplace
            aPlate = convolve(aPlate, laplaceKernel, mode="nearest")

            # Make normalness analysis to find lower and upper threshold
            # Rang based to z-score, compare to threshold adjusted by expected
            # fraction of removed positions
            rAPlate = aPlate.ravel()
            rPlate = plate[..., measure].ravel()
            sigma = np.sqrt(np.var(rAPlate))
            mu = np.mean(rAPlate)
            zScores = np.abs(rAPlate - mu)

            for idX in np.argsort(zScores)[::-1]:
                if (np.isnan(rPlate[idX]) or zScores[idX] > k * sigma /
                        np.exp(-(np.isfinite(rPlate).sum() /
                                 float(rPlate.size)) ** p)):

                    plate[idX / plate.shape[1], idX % plate.shape[1],
                          measure] = np.nan

                else:

                    break

            newNans += np.isnan(plate[..., measure]).sum()


def applyLog2Transform(dataArray, measures=None):
    """Log2 Transformation of dataArray values.

    If required, a filter for which measures to be log2-transformed as
    either an array or tuple of measure indices. If left None, all measures
    will be logged
    """

    if measures is None:
        measures = np.arange(dataArray[0].shape[-1])

    for plateIndex in range(len(dataArray)):
        dataArray[plateIndex][..., measures] = np.log2(
            dataArray[plateIndex][..., measures])


def applySobelFilter(dataArray, measure=1, threshold=1, **kwargs):
    """Applies a Sobel filter to the arrays and then compares this to a
    threshold setting all positions greater than said absolute threshold to NaN.

    measure     The measurement to evaluate
    threshold   The maximum absolute value allowed

    Further arguments of scipy.ndimage.sobel can be supplied
    """

    if ('mode' not in kwargs):
        kwargs['mode'] = 'nearest'

    for plateIndex in range(len(dataArray)):

        filt = (np.sqrt(sobel(
            dataArray[plateIndex][..., measure], axis=0, **kwargs) ** 2 +
            sobel(dataArray[plateIndex][..., measure], axis=1, **kwargs) ** 2)
            > threshold)

        dataArray[plateIndex][..., measure][filt] = np.nan


def applyLaplaceFilter(dataArray, measure=1, threshold=1, **kwargs):
    """Applies a Laplace filter to the arrays and then compares the absolute
    values of those to a threshold, discarding those exceeding it.

    measure     The measurement to evaluate
    threshold   The maximum absolute value allowed

    Further arguments of scipy.ndimage.laplace can be supplied
    """
    if ('mode' not in kwargs):
        kwargs['mode'] = 'nearest'

    for plateIndex in range(len(dataArray)):

        filt = (np.abs(laplace(dataArray[plateIndex][..., measure], **kwargs))
                > threshold)

        dataArray[plateIndex][..., measure][filt] = np.nan


def applyGaussSmoothing(dataArray, measure=1, sigma=3.5, **kwargs):
    """Applies a Gaussian Smoothing filter to the values of a plate (or norm
    surface).

    Note that this will behave badly if there are NaNs on the plate.

    measure     The measurement ot evaluate
    sigma       The size of the gaussian kernel
    """

    if ('mode' not in kwargs):
        kwargs['mode'] = 'nearest'

    for plateIndex in range(len(dataArray)):

        dataArray[plateIndex][..., measure] = gaussian_filter(
            dataArray[plateIndex][..., measure], sigma=sigma, **kwargs)


def applyMedianSmoothing(dataArray, measure=1, filterShape=(3, 3), **kwargs):

    if ('mode' not in kwargs):
        kwargs['mode'] = 'nearest'

    for plateIndex, plate in enumerate(dataArray):

        dataArray[plateIndex][..., measure] = median_filter(
            plate[..., measure], size=filterShape, **kwargs)


def applySigmaFilter(dataArray, nSigma=3):
    """Applies a per plate global sigma filter such that those values
    exceeding the absolute sigma distance to the mean are discarded.

    nSigma      Threshold distance from mean
    """
    for plateIndex in range(len(dataArray)):

        for measure in range(dataArray[plateIndex].shape[-1]):

            values = dataArray[plateIndex][..., measure]
            cleanValues = values[np.isfinite(values)]
            vBar = cleanValues.mean()
            vStd = cleanValues.std()
            values[np.logical_or(values < vBar - nSigma * vStd,
                                 values > vBar + nSigma * vStd)] = np.nan


#
#   METHODS: Normalisation method
#


def normalisation(dataBridge, normalisationSurface, updateBridge=True,
                  log=False):

    normalData = []
    bridgeArray = dataBridge.getAsArray()
    for plateIndex in range(normalisationSurface.shape[0]):

        if (bridgeArray[plateIndex] is None or
                normalisationSurface[plateIndex] is None):
            normalData.append(None)
        elif log:
            normalData.append(
                np.log2(bridgeArray[plateIndex]) -
                np.log2(normalisationSurface[plateIndex]))
        else:
            normalData.append(
                bridgeArray[plateIndex] -
                normalisationSurface[plateIndex])

    normalData = np.array(normalData)

    if updateBridge:
        dataBridge.setArrayRepresentation(normalData)
        dataBridge.updateBridge()

    return normalData
