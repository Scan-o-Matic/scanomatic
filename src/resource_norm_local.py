
#
#   DEPENDENCIES
#

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, sobel, laplace
from scipy.stats import probplot, linregress, pearsonr
import matplotlib.pyplot as plt
import itertools

#
#   INTERNAL DEPENDENCIES
#

import resource_xml_read
import resource_color_palette

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


def getControlPositionsCoordinates(dataObject, controlPositionKernel=None):
    """Returns list of tuples that emulates the results of running np.where"""

    platesCoordinates = []

    if isinstance(dataObject, DataBridge):
        dataObject = dataObject.getAsArray()

    nPlates = dataObject.shape[0]

    if controlPositionKernel is None:
        controlPositionKernel = [DEFAULT_CONTROL_POSITION_KERNEL] * nPlates

    for plateIndex in xrange(nPlates):

        plateCoordinates = [[], []]
        controlKernel = controlPositionKernel[plateIndex]
        kernelD1, kernelD2 = controlKernel.shape

        for idx1 in xrange(dataObject[plateIndex].shape[0]):

            for idx2 in xrange(dataObject[plateIndex].shape[1]):

                if controlKernel[idx1 % kernelD1, idx2 % kernelD2]:

                    plateCoordinates[0].append(idx1)
                    plateCoordinates[1].append(idx2)

        platesCoordinates.append(map(np.array, plateCoordinates))

    return platesCoordinates


def getExperimentPosistionsCoordinates(dataObject, controlPositionKernel=None):

    platesCoordinates = []

    if isinstance(dataObject, DataBridge):
        dataObject = dataObject.getAsArray()

    nPlates = dataObject.shape[0]

    if controlPositionKernel is None:
        controlPositionKernel = [DEFAULT_CONTROL_POSITION_KERNEL] * nPlates

    experimentPositionKernel = []
    for k in controlPositionKernel:
        experimentPositionKernel.append(k == False)

    for plateIndex in xrange(nPlates):

        plateCoordinates = [[], []]
        kernel = experimentPositionKernel[plateIndex]
        kernelD1, kernelD2 = kernel.shape

        for idx1 in xrange(dataObject[plateIndex].shape[0]):

            for idx2 in xrange(dataObject[plateIndex].shape[1]):

                if kernel[idx1 % kernelD1, idx2 % kernelD2]:

                    plateCoordinates[0].append(idx1)
                    plateCoordinates[1].append(idx2)

        platesCoordinates.append(map(np.array, plateCoordinates))

    return platesCoordinates


def getCoordinateFiltered(dataObject, coordinates, measure=1,
                          requireFinite=True,
                          requireCorrelated=False):

    if isinstance(dataObject, DataBridge):
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

    if isinstance(dataObject, DataBridge):
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
        controlPositionKernel=None, smoothing=None):
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

    if smoothing is not None:
        for measureIndex in xrange(plate.shape[2]):
            applyGaussSmoothing(
                normInterpolations,
                sigma=smoothing,
                measure=measureIndex)

    return normInterpolations

#
#   METHODS: Apply functions
#
#   Apply functions update the dataArray/Bridge values!
#


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
    for plateIndex in range(len(dataArray)):

        filt = (np.abs(laplace(dataArray[plateIndex][..., measure], **kwargs))
                > threshold)
        dataArray[plateIndex][..., measure][filt] = np.nan


def applyGaussSmoothing(dataArray, measure=1, sigma=3.5):
    """Applies a Gaussian Smoothing filter to the values of a plate (or norm
    surface).

    Note that this will behave badly if there are NaNs on the plate.

    measure     The measurement ot evaluate
    sigma       The size of the gaussian kernel
    """
    for plateIndex in range(len(dataArray)):

        dataArray[plateIndex][..., measure] = gaussian_filter(
            dataArray[plateIndex][..., measure], sigma=sigma, mode='nearest')


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

#
#   METHODS: Benchmarking
#


def getWithinBetweenCorrelation(dataObject, controlPositionKernel=None,
                                measuere=1):

    if controlPositionKernel is None:
        controlPositionKernel = [DEFAULT_CONTROL_POSITION_KERNEL] * len(
            dataObject)

    kernelKernel = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]], dtype=np.bool)
    benchmarkValues = []

    for plateIndex, plate in enumerate(dataObject):

        benchmarkPlateValues = [[], [], []]

        kernel = controlPositionKernel[plateIndex]
        kernelExtended = np.c_[(kernel == False).astype(np.int) * 1,
                               (kernel == False).astype(np.int) * 2,
                               (kernel == False).astype(np.int) * 3]
        kernelExtended = np.r_[kernelExtended,
                               kernelExtended * 4,
                               kernelExtended * 13]

        wLookup = {}
        hD1, hD2 = (int(v / 2.0 - 0.5) for v in kernelKernel.shape)

        for kD1 in range(kernel.shape[0], 2 * kernel.shape[0]):

            for kD2 in range(kernel.shape[1], 2 * kernel.shape[1]):

                localS = kernelExtended[kD1 - hD1: kD1 + hD1 + 1,
                                        kD2 - hD2: kD2 + hD2 + 1]

                whereSame = np.where(np.logical_and(
                    localS == localS[kernelKernel == 0],
                    kernelKernel != 0,
                    localS != 0))
                weightSame = kernelKernel[whereSame]
                whereOther = np.where(np.logical_and(
                    localS != localS[kernelKernel == 0],
                    localS != 0))
                weightOther = kernelKernel[whereOther]
                wLookup[(kD1 - kernel.shape[0], kD2 - kernel.shape[1])] = (
                    whereSame, weightSame, whereOther, weightOther)

        kD1, kD2 = kernel.shape[:2]

        for pD1 in range(plate.shape[0]):

            for pD2 in range(plate.shape[1]):

                whereSame, weightSame, whereOther, weightOther = wLookup[
                    (pD1 % kD1, pD2 % kD2)]

                lSlice = np.ones(kernelKernel.shape) * np.nan

                lSD1 = pD1 - hD1
                uSD1 = pD1 + hD1 + 1
                if lSD1 < 0:
                    lTD1 = -lSD1
                    lSD1 = 0
                else:
                    lTD1 = 0
                if uSD1 > plate.shape[0]:
                    uTD1 = kernelKernel.shape[0] + (plate.shape[0] - uSD1)
                    uSD1 = plate.shape[0]
                else:
                    uTD1 = kernelKernel.shape[0]

                lSD2 = pD2 - hD2
                uSD2 = pD2 + hD2 + 1
                if lSD2 < 0:
                    lTD2 = -lSD2
                    lSD2 = 0
                else:
                    lTD2 = 0
                if uSD2 > plate.shape[1]:
                    uTD2 = kernelKernel.shape[1] + (plate.shape[1] - uSD2)
                    uSD2 = plate.shape[1]
                else:
                    uTD2 = kernelKernel.shape[1]

                """
                print pD1, pD2
                print lTD1, uTD1, lTD2, uTD2
                print lSD1, uSD1, lSD2, uSD2
                """

                lSlice[lTD1:uTD1, lTD2:uTD2] = plate[lSD1:uSD1, lSD2:uSD2,
                                                     measuere]

                sameGroup = lSlice[whereSame]
                sameGroupFitite = np.isfinite(sameGroup)
                if sameGroupFitite.size > 0:
                    sameVal = ((sameGroup[sameGroupFitite] *
                                weightSame[sameGroupFitite]).sum() /
                               weightSame[sameGroupFitite].sum())
                else:
                    sameVal = np.nan

                otherGroup = lSlice[whereOther]
                otherGroupFitite = np.isfinite(otherGroup)
                if otherGroupFitite.size > 0:
                    otherVal = ((otherGroup[otherGroupFitite] *
                                 weightOther[otherGroupFitite]).sum() /
                                weightOther[otherGroupFitite].sum())
                else:
                    otherVal = np.nan

                benchmarkPlateValues[0].append(plate[pD1, pD2, measuere])
                benchmarkPlateValues[1].append(sameVal)
                benchmarkPlateValues[2].append(otherVal)

        #benchmarkValues.append(benchmarkPlateValues)
        benchmarkValues.append(np.array(benchmarkPlateValues, dtype=np.float))

    return benchmarkValues
#
#   METHODS: TimeSeries Helpers
#


def iterativeDataBridge(xmlReaderObject):

    for timeIndex in range(len(xmlReaderObject.get_scan_times())):
        yield DataBridge(xmlReaderObject, time=timeIndex)

#
#   METHODS: Plotting methods
#


def _plotLayout(plots):

    pC = plots / int(np.sqrt(plots))
    if plots % pC:
        pC += 1
    pR = plots / pC
    if pC * pR < plots:
        pR += 1

    return pC, pR


def getPlotValueSpans(measure, *dataObjects, **kwargs):

    #TODO: Broken
    if 'vmin' not in kwargs:
        kwargs['vmin'] = None

    if 'vmax' not in kwargs:
        kwargs['vmax'] = None

    for dataObject in dataObjects:
        if isinstance(dataObjects, DataBridge):
            dataObject = dataObject.getAsArray()
        for plate in dataObject:
            if plate is not None:
                if (kwargs['vmin'] is None or
                        plate[..., measure].min() < kwargs['vmin']):

                    kwargs['vmin'] = plate[..., measure].min()

                if (kwargs['vmax'] is None or
                        plate[..., measure].max() > kwargs['vmax']):

                    kwargs['vmax'] = plate[..., measure].max()
    return kwargs


def plotControlCurves(xmlReaderObject, averageMethod=IQRmean, phenotype=0,
                      title="Plate {0}"):

    Avgs = []
    Ctrls = []
    times = xmlReaderObject.get_scan_times()

    for dB in iterativeDataBridge(xmlReaderObject):

        ctrlArray = getControlPositionsArray(dB)
        ctrlCoord = getControlPositionsCoordinates(dB)

        Avgs.append(getControlPositionsAverage(ctrlArray))
        Ctrls.append([cAp[ctrlCoord[i]] for i, cAp in enumerate(ctrlArray)])

    A = [np.array(p).T for p in zip(*Avgs)]
    C = [np.array(p).T for p in zip(*Ctrls)]
    fig = plt.figure()

    pC, pR = _plotLayout(len(C))

    for plateIndex in range(len(C)):

        ax = fig.add_subplot(pR, pC, plateIndex + 1)
        ax.set_title(title.format(plateIndex + 1))

        for c in C[plateIndex][phenotype]:
            ax.semilogy(times, c, '-g', basey=2)

        ax.semilogy(times, A[plateIndex][phenotype], '-r', basey=2)
    return fig


def plotControlPhenotypesStats(dataObject, measure=1,
                               controlPositionsCoordinates=None,
                               controlPositionKernel=None, log=False):

    if isinstance(dataObject, DataBridge):
        dataObject = dataObject.getAsArray()

    if controlPositionsCoordinates is None:
        controlPositionsCoordinates = getControlPositionsCoordinates(
            dataObject, controlPositionKernel)

    data = []
    for plateIndex in range(len(dataObject)):
        plate = dataObject[plateIndex][..., measure][
            controlPositionsCoordinates[plateIndex]]
        if log:
            data.append(np.log2(plate[np.isfinite(plate)]))
        else:
            data.append(plate[np.isfinite(plate)])

    pC, pR = _plotLayout(len(dataObject) + 1)

    f = plt.figure()
    ax = f.add_subplot(pC, pR, 1)
    ax.boxplot(data)
    ax.set_title("Control Position Phenotype {0}".format(measure))
    ax.set_xticklabels(
        ["Plate {0}".format(i + 1) for i in range(len(dataObject))])

    for plateIndex in range(len(dataObject)):

        ax = f.add_subplot(pC, pR, 2 + plateIndex)
        probplot(data[plateIndex], plot=plt)
        ax.set_title("Plate {0} {1}".format(plateIndex + 1,
                                            ax.title.get_text()))

        ax.text(0.05, 0.9, "N = {0}".format(data[plateIndex].size),
                transform=ax.transAxes)

    f.tight_layout()

    return f


def plotHeatMaps(dataObject, showArgs=tuple(), showKwargs=dict(),
                 measure=1, title="Plate {0}", equalVscale=True):

    """
    if isinstance(dataObject, DataBridge):

        dataObject = dataObject.getAsArray()
    """
    pC, pR = _plotLayout(len(dataObject))
    fig = plt.figure()

    vMin = None
    vMax = None

    if 'vmax' in showKwargs or 'vmin' in showKwargs:

        pass

    elif equalVscale:

        for plate in dataObject:
            finPlate = plate[..., measure][np.isfinite(plate[..., measure])]
            if vMin is None or finPlate.min() < vMin:
                vMin = finPlate.min()
            if vMax is None or finPlate.max() > vMax:
                vMax = finPlate.max()

    for plateIndex in range(len(dataObject)):

        if (dataObject is not None):
            ax = fig.add_subplot(pR, pC, plateIndex + 1)
            ax.set_title(title.format(plateIndex + 1))
            if None not in (vMax, vMin):
                I = ax.imshow(
                    dataObject[plateIndex][..., measure],
                    interpolation="nearest", vmin=vMin, vmax=vMax,
                    *showArgs, **showKwargs)
            else:
                I = ax.imshow(
                    dataObject[plateIndex][..., measure],
                    interpolation="nearest", *showArgs, **showKwargs)

            if not equalVscale:

                cbar = plt.colorbar(I, orientation='vertical')
                cbar.ax.tick_params(labelsize='xx-small')

            ax.axis("off")

    if equalVscale:
        fig.subplots_adjust(bottom=0.85)
        cbar_ax = fig.add_axes([0.175, 0.035, 0.65, 0.02])
        cbar = plt.colorbar(I, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize='xx-small')
        fig.subplots_adjust(left=0.004, right=0.005)
    fig.tight_layout()
    return fig


def _correlationAxPlot(ax, X, Y, plotMethod, equalAxis=True,
                       pearsonText=True, linearRegText=True,
                       linearReg=True, oneToOneLine=True,
                       linearRegCorrValue=False,
                       scatterColor=(0, 0, 1, 0.2),
                       **plotKW):

    def nullMethod(X, base=0):
        return X

    def invTransform(X, base=10):
        return base ** X

    getattr(ax, plotMethod)(X, Y, ',', color=scatterColor, **plotKW)

    if plotMethod in ("loglog", "semilogx"):
        iTX = invTransform
        if 'basex' in plotKW:
            baseX = plotKW['basex']
        else:
            baseX = 10
    else:
        iTX = nullMethod
        baseX = 0
    if plotMethod in ("loglog", "semilogy"):
        iTY = invTransform
        if 'basey' in plotKW:
            baseY = plotKW['basey']
        else:
            baseY = 10
    else:
        iTY = nullMethod
        baseY = 0

    if plotMethod in ("plot", "semilogy"):
        tX = X
    else:
        tX = np.log(X) / np.log(baseX)
    if plotMethod in ("plot", "semilogx"):
        tY = Y
    else:
        tY = np.log(Y) / np.log(baseY)

    sliceXY = np.logical_and(np.isfinite(tX), np.isfinite(tY))

    if linearReg or linearRegText:
        slope, intercept, r_value, p_value, std_err = linregress(
            tX[sliceXY], tY[sliceXY])

    if pearsonText or linearRegCorrValue:
        pearsonCorr, pearsonP = pearsonr(tX[sliceXY], tY[sliceXY])

    xMin = tX[np.logical_and(tX != 0, np.isfinite(tX))].min()
    xMax = tX[np.logical_and(tX != 0, np.isfinite(tX))].max()

    if linearReg:
        getattr(ax, plotMethod)([iTX(xMin, base=baseX),
                                 iTX(xMax, base=baseX)],
                                [iTY(xMin * slope + intercept, base=baseY),
                                 iTY(xMax * slope + intercept, base=baseY)],
                                '-r', **plotKW)
    if linearRegCorrValue:
        cX = tX[sliceXY].mean()
        cY = tY[sliceXY].mean()
        dX = (tX[sliceXY].min() - cX, tX[sliceXY].max() - cX)
        dY = (tY[sliceXY].min() - cY, tY[sliceXY].max() - cY)
        oX = dX[np.abs(dX).argmax()] * 0.5
        oY = dY[np.abs(dY).argmax()] * 0.5

        ax.text(
            iTX(cX + oX, base=baseX),
            iTY(cY + oY, base=baseY),
            "{0:.2f}".format(pearsonCorr),
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10)

    if linearRegText:
        ax.text(
            0.05, 0.9,
            "r2 {0:.2f}, p {1:.2f}, stdErr {2:.2f}".format(
                r_value ** 2, p_value, std_err),
            transform=ax.transAxes,
            fontsize=8)

    if equalAxis:
        lB, uB = zip(ax.get_ylim(), ax.get_xlim())
        ax.set_xlim(min(lB), max(uB))
        ax.set_ylim(min(lB), max(uB))

    if oneToOneLine:
        getattr(ax, plotMethod)([iTX(xMin, base=baseX),
                                 iTX(xMax, base=baseX)],
                                [iTY(xMin, base=baseY),
                                 iTY(xMax, base=baseY)], '-g', **plotKW)

    if pearsonText:
        ax.text(0.9, 0.05,
                "Pearson {0:.2f}, p {1:.2f}".format(pearsonCorr, pearsonP),
                fontsize=8,
                horizontalalignment='right',
                transform=ax.transAxes)

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=6)


def plotClassifiedCorrelation(dataObject1, dataObject2, classifierObject,
                              measure1=1, measure2=1,
                              plateHeader="Plate {0}",
                              xLabel="Measure 1",
                              yLabel="Measure 2",
                              plotMethod="plot",
                              colorPaletteBase=None,
                              **plotKW):

    if (dataObject1.shape[0] != dataObject2.shape[0]):
        raise Exception(
            "DataObjects don't share first dimension ({0}, {1})".format(
                dataObject1.shape, dataObject2.shape))

    pC, pR = _plotLayout(len(dataObject1))
    f = plt.figure()
    for plateIndex in range(len(dataObject1)):

        if (dataObject1[plateIndex].shape[:-1] ==
                dataObject2[plateIndex].shape[:-1]):

            X = dataObject1[plateIndex][..., measure1].ravel()
            Y = dataObject2[plateIndex][..., measure2].ravel()
            C = classifierObject[plateIndex].ravel()
            cValues = np.unique(C)

            ax = f.add_subplot(pR, pC, plateIndex + 1)
            ax.set_title(plateHeader.format(plateIndex + 1))
            for i, color in enumerate(
                    resource_color_palette.get(cValues.size,
                                               base=colorPaletteBase)):

                cVal = cValues[i]

                _correlationAxPlot(
                    ax, X[C == cVal], Y[C == cVal], plotMethod,
                    pearsonText=False, linearRegText=False,
                    linearReg=True, oneToOneLine=False,
                    linearRegCorrValue=True,
                    equalAxis=False,
                    scatterColor=color,
                    **plotKW)

            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)

    f.tight_layout()
    return f


def plotCorrelation(dataObject1, dataObject2, measure1=1, measure2=1,
                    plateHeader="Plate {0}",
                    xLabel="Measure 1",
                    yLabel="Measure 2",
                    plotMethod="plot",
                    **plotKW):

    if (dataObject1.shape[0] != dataObject2.shape[0]):
        raise Exception(
            "DataObjects don't share first dimension ({0}, {1})".format(
                dataObject1.shape, dataObject2.shape))

    pC, pR = _plotLayout(len(dataObject1))
    f = plt.figure()

    for plateIndex in range(len(dataObject1)):

        if (dataObject1[plateIndex].shape[:-1] ==
                dataObject2[plateIndex].shape[:-1]):

            X = dataObject1[plateIndex][..., measure1].ravel()
            Y = dataObject2[plateIndex][..., measure2].ravel()

            ax = f.add_subplot(pR, pC, plateIndex + 1)
            ax.set_title(plateHeader.format(plateIndex + 1))
            _correlationAxPlot(
                ax, X, Y, plotMethod,
                oneToOneLine=plotMethod == "plot",
                **plotKW)
            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)
    return f


def plotPairWiseCorrelation(dataPairs, alpha=0.3, dataPadding=0.1):

    plates = dataPairs.shape[0]
    f = plt.figure()
    pColor = (0, 0, 1, alpha)

    for plateA in range(plates):

        for plateB in range(plates):

            if plateA < plateB:

                ax = f.add_subplot(plates, plates,
                                   plateA * plates + plateB + 1)

                X = dataPairs[plateA]
                Y = dataPairs[plateB]
                _correlationAxPlot(ax, X, Y, "plot", scatterColor=pColor)

    f.tight_layout()

    step = 1.0 / plates
    #step -= step / 2.0

    for i in range(1, plates + 1):

        f.text(step * i - step / 2, 0.05,
               "Plate {0}".format(i),
               horizontalalignment='center',
               verticalalignment='center',
               rotation='horizontal',
               fontsize=16,
               transform=f.transFigure)

        f.text(0.05, step * i - step / 2,
               "Plate {0}".format(plates + 1 - i),
               horizontalalignment='center',
               verticalalignment='center',
               rotation='vertical',
               fontsize=16,
               transform=f.transFigure)

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    f.text(0.5, 0.5 * step,
           "Green line is 1:1. Red regression. Upper text regression info."
           " Lower text correlation info.", bbox=props, fontsize=10,
           horizontalalignment='center',
           transform=f.transFigure)

    return f


def plotBenchmark(B, dataBridge):

    pC, pR = _plotLayout(len(B))
    f = plt.figure()

    for plateIndex, plate in enumerate(B):

        ax = f.add_subplot(pR, pC, plateIndex + 1)
        ax.plot(plate[0], plate[1], 'g,')
        ax.plot(plate[0], plate[2], 'r,')
        ax.set_ylabel("Local group mean phenotype")
        ax.set_xlabel("Position phenotype")
        lB, uB = zip(ax.get_ylim(), ax.get_xlim())
        ax.set_xlim(min(lB), max(uB))
        ax.set_ylim(min(lB), max(uB))
        p0 = np.array(plate[0])
        p1 = np.array(plate[1])
        p2 = np.array(plate[2])
        #r1 = np.abs(np.log(np.array(plate[1]) / np.array(plate[0])))
        r1 = (p1 ** 2 + p0 ** 2) * np.abs(p1 - p0)
        r2 = (p2 ** 2 + p0 ** 2) * np.abs(p2 - p0)
        #r2 = np.abs(np.log(np.array(plate[2]) / np.array(plate[0])))
        for r in (r1, r2):
            for v in np.sort(r[np.isfinite(r)])[-5:]:  # random.sample(r[np.isfinite(r)], 5):
                pos = [d[0] for d in np.where(r == v)][0]
                d1 = pos % dataBridge[plateIndex].shape[0]
                d2 = pos / dataBridge[plateIndex].shape[0]
                ax.annotate(
                    "({0}, {1})".format(d1, d2),
                    xy=(plate[0][pos], plate[1][pos]), xytext=(-5, 5),
                    textcoords='offset points', ha='right', va='bottom',
                    fontsize=8,
                    #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                    arrowprops=dict(arrowstyle='-'))

        finR1 = np.logical_and(np.isfinite(plate[0]), np.isfinite(plate[1]))
        finR2 = np.logical_and(np.isfinite(plate[0]), np.isfinite(plate[2]))
        print "Pearson Within {0}".format(
            pearsonr(plate[0][finR1],
                     plate[1][finR1]))

        print "Pearson Between {0}".format(
            pearsonr(plate[0][finR2],
                     plate[2][finR2]))

    f.tight_layout()
    return f
#
#   CLASSES
#


class _NumpyArrayInterface(object):

    def __init__(self, dataObject):

        self._dataObject = dataObject

    def __getitem__(self, key):

        return self._dataObject[key]

    def __iter__(self):

        for i in xrange(len(self._dataObject)):

            yield self.__getitem__(i)

    def __len__(self):

        return self._dataObject.shape[0]

    @property
    def shape(self):
        return self._dataObject.shape

    @property
    def ndim(self):
        return self._dataObject.ndim


class DataBridge(_NumpyArrayInterface):

    def __init__(self, source, **kwargs):
        """The data structure is expected to be convertable into
        a four dimensional array where dimensions are as follows:

            1:   Plate
            2,3: Positional Coordinates
            4:   Data Measure

        For importing primary analysis data, this means that measures
        for the different compartments will be enumerated after each other
        along the 4th dimension as they come.

        In the case of working directly on an XML file such as the
        analysis.xml or the analysis_slim.xml a keyword argument 'time'
        is required:

            time:   The time index  for which to bridge the data

        """

        self._source = source
        super(DataBridge, self).__init__(None)

        #This method is assigned dynamically based on
        #type of data imported
        self.updateSource = None

        self._createArrayRepresentation(**kwargs)

    def _createArrayRepresentation(self, **kwargs):

        if isinstance(self._source, np.ndarray):
            self._dataObject = self._source.copy()
            self.updateSource = self._updateToArray

        elif isinstance(self._source, dict):

            plates = [[]] * len(self._source)  # Creates plates and 1D pos

            for p in self._source.values():

                for d1 in p:
                    plates[-1].append([])  # Vector for 2D pos

                    for cell in d1:

                        plates[-1][-1].append([])

                        for compartment in cell.values():

                            for value in compartment.values():

                                plates[-1][-1][-1].append(value)

            self._dataObject = np.array(plates)
            self.updateSource = self._updateToFeatureDict

        elif isinstance(self._source, resource_xml_read.XML_Reader):

            if "time" not in kwargs:
                raise Exception(
                    "XML Reader objects can only be bridged for a time index"
                    " at a time, you must supply keyword argument 'time'")

            else:

                self._timeIndex = kwargs["time"]
                tmpD = []
                for p in self._source.get_data().values():
                    tmpD.append(p[..., self._timeIndex, :].copy())
                self._dataObject = np.array(tmpD)

        else:

            raise Exception(
                "Unknown data format {0}".format(type(self._source)))

    def _updateToFeatureDict(self):
        """Updates the source inplace"""

        plateIndex = 0
        for p in self._source:

            for d1Index, d1 in enumerate(p):

                for d2Index, cell in enumerate(d1):

                    measureIndex = 0

                    for compartment in cell.values():

                        for valueKey in compartment:

                            compartment[valueKey] = self._dataObject[
                                plateIndex, d1Index, d2Index, measureIndex]

                            measureIndex += 1
            plateIndex += 1

    def _updateToArray(self):
        """Updates the source inplace"""

        for i, p in enumerate(self._source):

            p[...] = self._dataObject[i]

    def _updateToXMLreader(self):
        """Updates the source inplace"""

        for plateIndex in self._dataObject.shape[0]:

            for d1 in self._dataObject[plateIndex].shape[0]:

                for d2 in self._dataObject[plateIndex].shape[1]:

                    self._source.set_data_value(
                        plateIndex, d1, d2, self._timeIndex,
                        self._dataObject[plateIndex][d1, d2])

    def getSource(self):
        """Returns a reference to the source"""

        return self._source

    def getAsArray(self):
        """Returns the data as a normalisations compatible array"""

        return self._dataObject

    def setArrayRepresentation(self, array):
        """Method for overwriting the array representation of the data"""

        if (array.shape == self._dataObject.shape):
            self._dataObject = array
        else:
            raise Exception(
                "New representation must match current shape: {0}".format(
                    self._dataObject.shape))


class SubSample(_NumpyArrayInterface):

    def __init__(self, dataObject, kernels=None):
        """This class puts an interchangeable subsampling level
        onto any applicable dataObject.

        If no kernel is set the layer is transparent and the original
        data in its original conformation can be directly accessed.

        If a kernel is in place, that plate will become strided such
        that it will expose an array as it would have looked before
        it got interleaved into the current plate.

        The

        Parameters:
            dataObject      An object holding several plates

            kernels         An array of kernels or None(s)

        """
        self._dataObject = dataObject
        self._kernels = None
        self.kernels = kernels

    @property
    def kernels(self):
        return self._kernels

    @kernels.setter
    def kernels(self, kernels):

        if (kernels is not None):

            assert len(kernels) == len(self._dataObject), (
                "Must have exactly as many kernels {0} as plates {1}".format(
                    len(kernels), len(self._dataObject)))

            for i, kernel in enumerate(kernels):

                if (kernel is not None):

                    assert kernel.sum() == 1, (
                        "All kernels must have exactly one true value "
                        "(kernel {0} has {1})".format(i, kernel.sum()))

                    assert np.array(
                        [p % k == 0 for p, k in itertools.izip(
                            self._dataObject[0].shape[:2],
                            kernel.shape)]).all(), (
                                "Dimension missmatch between kernel and plate"
                                " ({0} not evenly divisable with {1})".format(
                                    self._dataObject.shape[:2], kernel.shape))

        self._kernels = kernels

    def __getitem__(self, value):

        plate = self._dataObject[0]

        if (self._kernels is None or self._kernels[value] is None):
            return plate

        else:

            kernel = self._kernels[0]
            kernelD1, kernelD2 = (v[0] for v in np.where(kernel))

            assert plate.ndim in (2, 3), (
                "Plate {0} has wrong number of dimensions {1}".format(
                    value, plate.ndim))

            if (plate.ndim == 2):

                ravelOffset = plate.shape[1] * kernelD1 + kernelD2
                plateShape = (plate.shape[0] / kernel.shape[0],
                              plate.shape[1] / kernel.shape[1])
                plateStrides = (plate.strides[0] * kernel.shape[0],
                                plate.strides[1] * kernel.shape[1])

            elif (plate.ndim == 3):

                ravelOffset = (plate.shape[2] * plate.shape[1] * kernelD1 +
                               plate.shape[2] * kernelD2)
                plateShape = (plate.shape[0] / kernel.shape[0],
                              plate.shape[1] / kernel.shape[1],
                              plate.shape[2])
                plateStrides = (plate.strides[0] * kernel.shape[0],
                                plate.strides[1] * kernel.shape[1],
                                plate.strides[2])

            return np.lib.stride_tricks.as_strided(
                plate.ravel()[ravelOffset:],
                shape=plateShape,
                strides=plateStrides)

''' Outdated since it doesn't keep memory pointers to numpy arrays.
Also overly complex

class SubSample(object):

    def __init__(self, dataObject,
                 controlPositionKernel=None):

        self._sourceObject = dataObject

        if isinstance(dataObject, DataBridge):
            self._sourceIsBridge = True
            self._sourceArray = dataObject.getAsArray()
        else:
            self._sourceIsBridge = False
            self._sourceArrays = dataObject

        self._kernels = controlPositionKernel
        self._length = len(self._sourceArrays)

        self._sourceReformatted = None
        if (self._kernels is not None):
            self._setupSubslicing()

    def _decomposePrepare(self, interleavedArray):
        """Method assumes 2D interleaved array composed of 4 subarrays.

        Details:

            First array is separated into odd and even rows

            Second first odd row and first even row are zipped into an
            array. Same for second row etc. So that the output looks like:

                [(odd[0], even[0]), (odd[1], even[1]), ...]

            Thirdly the element of each odd even tuple are zipped,
            using first row odd and even as example:

                [(odd[0][0], even[0][0]), (odd[0][1], even[0][1]) ... ]

            Fourthly, as the array currently has 3D the sequence of
            tuples per row in previous step are chained. For first row:

                (odd[0][0], even[0][0], odd[0][1], even[0][1] ... )

            Since numpy would make an object of generators and the state
            of the entire object currently is:

                [generatorRow0, generatorRow1, ... ]

            list is mapped to each row to instantiate a list from the
            generator.

            The resulting list of lists is finally converted into a 2D
            array.

        Parameters:

            interleavedArray    Any 2D iterable of rectangular shape

        Returns:

            2D numpy array in which every fourth object in a row
            would come from a specific subarray.

            Assuming subarrays having been interleaved using the
            following pattern:

                [[0, 1]
                 [2, 3]]

            Their position will be [0, 2, 1, 3]

        """

        return np.array(
            map(list, (itertools.chain(*itertools.izip(*zipV))
                       for zipV in itertools.izip(
                           interleavedArray[::2],
                           interleavedArray[1::2]))))

    def _recompose(self, decomposedPlates, decomposedKernels):
        """

        """

        #Make sure all kernels are correctly shaped
        assert np.array([k is None and True or k.size == 4
                         for k in decomposedKernels]).all()

        #Make sure the kernel fits evenly in both directions of
        #the plates
        assert np.array([(plate is None or kernel is None) and
                         True or (plate.shape[1] % kernel.shape[1] == 0)
                         for plate, kernel in
                         itertools.izip(
                             decomposedPlates,
                             decomposedKernels)]).all()

        #Make sure each kernel only has one array as it is only supported
        assert np.array(
            [k is None and True or k.sum() == 1
             for k in decomposedKernels]).all()

        plates = []
        for plate, kernel in itertools.izip(decomposedPlates,
                                            decomposedKernels):

            if (plate is None or kernel is None):
                plates.append(None)
            else:
                plates.append(plate[:, np.where(kernel.ravel())[0][0]::4])

        return np.array(plates)

    def _setupSubslicing(self):
        """ Method called to decompose source array and recompose
        sub arrays from source using kernels.

        Returns:

            An array of the sub-plates sliced from the original arrays

        """
        sourceReformatted = []
        kernelReformatted = []
        for plate, kernel in itertools.izip(self._sourceArrays, self._kernels):
            if (plate is None or kernel is None):
                sourceReformatted.append(None)
                kernelReformatted.append(None)
            else:
                sourceReformatted.append(self._decomposePrepare(plate))
                kernelReformatted.append(self._decomposePrepare(kernel))

        self._sourceReformatted = self._recompose(sourceReformatted,
                                                  kernelReformatted)

    def __getitem__(self, key):

        if isinstance(key, int) or isinstance(key[0], int):

            if (self._kernels is None):
                return self._sourceArrays[key]
            else:
                return self._sourceReformatted[key]

    def __iter__(self):

        for i in range(self._length):
            yield self.__getitem__(i)

    def __len__(self):

        return self._length

    @property
    def kernel(self):
        return self._kernels

    @kernel.setter
    def kernel(self, newKernels):

        self._kernels = newKernels
        if (self._kernels is not None):
            self._setupSubslicing()

    @property
    def shape(self):
        return self._sourceArray.shape
'''
