
#
#   DEPENDENCIES
#

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

#
#   INTERNAL DEPENDENCIES
#

import resource_xml_read

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


def getNormalisationWithGridData(
        controlPositionsDataArray,
        controlPositionsCoordinates=None,
        normalisationSequence=('cubic', 'linear', 'nearest'),
        useAccumulated=False,
        missingDataValue=np.nan,
        controlPositionKernel=None):
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

    for plateIndex in xrange(nPlates):

        points = controlPositionsCoordinates[plateIndex]
        plate = controlPositionsDataArray[plateIndex].copy()
        normInterpolations
        mgrid = np.mgrid[0:plate.shape[0], 0:plate.shape[1]]

        for measureIndex in xrange(plate.shape[2]):

            for method in normalisationSequence:

                if useAccumulated:
                    points = np.where(plate[..., measureIndex] !=
                                      missingDataValue)

                values = plate[..., measureIndex][points]

                res = griddata(points, values, mgrid, method=method,
                               fill_value=missingDataValue)

                accPoints = np.where(
                    plate[..., measureIndex] == missingDataValue)
                plate[..., measureIndex][accPoints] = res[accPoints]

                if not (plate[..., measureIndex] == missingDataValue).any():
                    break

    return np.array(normInterpolations)

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


def plotHeatMaps(dataObject, showArgs=tuple(), showKwargs=dict(),
                 measure=1, title="Plate {0}"):

    if isinstance(dataObject, DataBridge):

        dataObject = dataObject.getAsArray()

    pC, pR = _plotLayout(len(dataObject))
    fig = plt.figure()

    for plateIndex in range(len(dataObject)):

        ax = fig.add_subplot(pR, pC, plateIndex + 1)
        ax.set_title(title.format(plateIndex + 1))
        ax.imshow(dataObject[plateIndex][..., measure],
                  interpolation="nearest", *showArgs, **showKwargs)
        ax.axis("off")

    fig.tight_layout()
    return fig

#
#   CLASSES
#


class DataBridge(object):

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
        self._arrayRepresentation = None

        #This method is assigned dynamically based on
        #type of data imported
        self.updateSource = None

        self._createArrayRepresentation(**kwargs)

    def _createArrayRepresentation(self, **kwargs):

        if isinstance(self._source, np.ndarray):
            self._arrayRepresentation = self._source.copy()
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

            self._arrayRepresentation = np.array(plates)
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
                self._arrayRepresentation = np.array(tmpD)

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

                            compartment[valueKey] = self._arrayRepresentation[
                                plateIndex, d1Index, d2Index, measureIndex]

                            measureIndex += 1
            plateIndex += 1

    def _updateToArray(self):
        """Updates the source inplace"""

        for i, p in enumerate(self._source):

            p[...] = self._arrayRepresentation[i]

    def _updateToXMLreader(self):
        """Updates the source inplace"""

        for plateIndex in self._arrayRepresentation.shape[0]:

            for d1 in self._arrayRepresentation[plateIndex].shape[0]:

                for d2 in self._arrayRepresentation[plateIndex].shape[1]:

                    self._source.set_data_value(
                        plateIndex, d1, d2, self._timeIndex,
                        self._arrayRepresentation[plateIndex][d1, d2])

    def getSource(self):
        """Returns a reference to the source"""

        return self._source

    def getAsArray(self):
        """Returns the data as a normalisations compatible array"""

        return self._arrayRepresentation

    def setArrayRepresentation(self, array):
        """Method for overwriting the array representation of the data"""

        if (array.shape == self._arrayRepresentation.shape):
            self._arrayRepresentation = array
        else:
            raise Exception(
                "New representation must match current shape: {0}".format(
                    self._arrayRepresentation.shape))
