"""Contains basic aspects of numpy interface such that it in
basic aspect can be used in the same way while derived classes
can implement specific behaviours.
"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
#   DEPENDENCIES
#

import numpy as np
import os
import sys
import time
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.optimize import leastsq
from scipy.stats import linregress
import itertools
import matplotlib.pyplot as plt

#
#   INTERNAL DEPENDENCIES
#

import _mockNumpyInterface
import scanomatic.io.xml.reader as xmlReader


class Phenotyper(_mockNumpyInterface.NumpyArrayInterface):

    GT_VALUE = 0
    GT_VALUE_ERR = 1
    GT_VALUE_POS = 2
    GT_2ND_VALUE = 3
    GT_2ND_ERR = 4
    GT_2ND_POS = 5

    def __init__(self, dataObject, timeObject=None,
                 medianKernelSize=5, gaussSigma=1.5, linRegSize=5,
                 measure=None, baseName=None, itermode=False):

        self._source = dataObject
        self._generationTimes = None
        self._curveFits = None

        self._timeObject = None
        self._baseName = baseName

        if isinstance(dataObject, xmlReader.XML_Reader):
            arrayCopy = self._xmlReader2array(dataObject)

            if timeObject is None:
                timeObject = dataObject.get_scan_times()
        else:
            arrayCopy = dataObject.copy()

        self.times = timeObject

        #TODO: Could require selecting measurement if more than one instead
        assert (arrayCopy.shape[-1] == 1 or
                len(arrayCopy.shape) == 3), (
                    "Phenotype Strider only work with one phenotype")

        super(Phenotyper, self).__init__(arrayCopy)

        assert medianKernelSize % 2 == 1, "Median kernel size must be odd"
        self._medianKernelSize = medianKernelSize
        self._gaussSigma = gaussSigma
        self._linRegSize = linRegSize
        self._itermode = itermode

        if not self._itermode:
            self._analyse()

    @classmethod
    def LoadFromXML(cls, path, **kwargs):
        """Class Method used to create a Phenotype Strider directly
        from a path do an xml

        Parameters:

            path        The path to the xml-file

        Optional Parameters can be passed as keywords and will be
        used in instanciating the class.
        """

        xml = xmlReader.XML_Reader(path)
        if (path.lower().endswith(".xml")):
            path = path[:-4]

        return cls(xml, baseName=path, **kwargs)

    @classmethod
    def LoadFromNumPy(cls, path, timesPath=None, **kwargs):
        """Class Method used to create a Phenotype Strider from
        a saved numpy data array and a saved numpy times array.

        Parameters:

            path        The path to the data numpy file

        Optional parameter:

            timesPath   The path to the times numpy file
                        If not supplied both paths are assumed
                        to be named as:

                            some/path.data.npy
                            some/path.times.npy

                        And path parameter is expexted to be
                        'some/path' in this examply.

        Optional Parameters can be passed as keywords and will be
        used in instanciating the class.
        """
        dataPath = path
        if (path.lower().endswith(".npy")):
            path = path[:-4]
            if (path.endswith(".data")):
                path = path[:-5]

        if (timesPath is None):
            timesPath = path + ".times.npy"

        if (not os.path.isfile(dataPath)):
            if (os.path.isfile(timesPath + ".data.npy")):

                timesPath += ".data.npy"

            elif (os.path.isfile(timesPath + ".npy")):

                timesPath += ".npy"

        return cls(np.load(dataPath), np.load(timesPath), baseName=path,
                   **kwargs)

    @staticmethod
    def ChapmanRichards4ParameterExtendedCurve(X, b0, b1, b2, b3, D):

        return D + b0 * np.power(1.0 - b1 * np.exp(-b2 * X), 1.0 / (1.0 - b3))

    @staticmethod
    def RCResiduals(crParams, X, Y):

        return Y - Phenotyper.ChapmanRichards4ParameterExtendedCurve(
            X, *crParams)

    @staticmethod
    def CalculateFitRSquare(X, Y, p0=np.array([4.5, -50, 0.3, 3, -3],
                                              dtype=np.float)):

        """X and Y must be 1D, Y must be log2"""

        p = leastsq(Phenotyper.RCResiduals, p0, args=(X, Y))[0]
        Yhat = Phenotyper.ChapmanRichards4ParameterExtendedCurve(
            X, *p)
        return (1.0 - np.square(Yhat - Y).sum() /
                np.square(Yhat - Y[np.isfinite(Y)].mean()).sum()), p

    @property
    def source(self):

        return self._source

    def _xmlReader2array(self, dataObject):

        return np.array([dataObject.get_data()[k] for k in sorted(
            dataObject.get_data().keys())])

    def _analyse(self):

        self._smoothen()
        self._calculatePhenotypes()

    def iterAnalyse(self):

        if (self._itermode is False):
            raise Exception("Can't iterate when not in itermode")
        else:
            n = sum((p.shape[1] * p.shape[2] for p in self._dataObject)) + 1.0
            i = 0
            self._smoothen()
            yield i / n
            for x in self._calculatePhenotypes():
                i += 1
                yield i / n

        self._itermode = False

    def _smoothen(self):
        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S\t", time.localtime()) +
            "Smoothing Started\n")
        medianFootprint = np.ones((1, self._medianKernelSize))

        for plate in self._dataObject:

            stridedPlate = np.lib.stride_tricks.as_strided(
                plate,
                shape=(plate.shape[0] * plate.shape[1], plate.shape[2]),
                strides=(plate.strides[1], plate.strides[2]))

            stridedPlate[...] = median_filter(
                stridedPlate, footprint=medianFootprint, mode='reflect')

            stridedPlate[...] = gaussian_filter1d(
                stridedPlate, sigma=self._gaussSigma, mode='reflect', axis=-1)

        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S\t", time.localtime()) +
            "Smoothing Done\n")

    def _calculatePhenotypes(self):

        def _linReg(X, Y):
            return linregress(X, Y)[0::4]

        def _linReg2(*args):
            return linregress(args[:linRegSize], args[linRegSize:])[0::4]

        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S\t", time.localtime()) +
            "Phenotype Extraction Started\n")

        timesStrided = np.lib.stride_tricks.as_strided(
            self._timeObject,
            shape=(self._timeObject.shape[0] - (self._linRegSize - 1),
                   self._linRegSize),
            strides=(self._timeObject.strides[0],
                     self._timeObject.strides[0]))

        flatT = self._timeObject.ravel()

        allGT = []
        allFits = []
        linRegSize = self._linRegSize
        #linRegUFunc = np.frompyfunc(_linReg2, linRegSize * 2, 2)
        posOffset = (linRegSize - 1) / 2

        for plateI, plate in enumerate(self._dataObject):

            stridedPlate = np.lib.stride_tricks.as_strided(
                plate,
                shape=(plate.shape[0] * plate.shape[1],
                       plate.shape[2] - (linRegSize - 1),
                       linRegSize),
                strides=(plate.strides[1],
                         plate.strides[2], plate.strides[2]))

            """ THIS CODE WOULD BE NEATER BUT ISN'T FASTER NOR VERY CLEAR

            timesStrided = np.lib.stride_tricks.as_strided(
            self._timeObject,
            shape=(stridedPlate.shape[0],
            stridedPlate.shape[1],
            self._linRegSize),
            strides=(0,
            self._timeObject.strides[0],
            self._timeObject.strides[0]))
            t = ([timesStrided[..., i] for i in range(
            timesStrided.shape[-1])] +
            [stridedPlate[..., i] for i in range(
            stridedPlate.shape[-1])])

            plateData = np.array(linRegUFunc(*t))

            argSort = plateData[..., -1].argsort(axis=-1)
            vals = np.array((
            1.0 / plateData[argSort[..., -1]][..., 0],
            plateData[argSort[..., -1]][..., 1],
            argSort[..., -1] + posOffset))

            allGT.append(np.lib.stride_tricks.as_strided(
            vals,
            shape=(
            plate.shape[0], plate.shape[1], vals.shape[-1]),
            strides=(
            plate.shape[1] * vals.strides[-2], vals.strides[-2],
            vals.strides[-1])))

            """
            generationTimes = np.zeros((plate.shape[:2]) + (6,),
                                       dtype=plate.dtype)

            allGT.append(generationTimes)

            curveFits = np.zeros((plate.shape[:2]) + (6,), dtype=plate.dtype)

            allFits.append(curveFits)

            stridedPlate = np.lib.stride_tricks.as_strided(
                plate,
                shape=(plate.shape[0], plate.shape[1],
                       plate.shape[2] - (linRegSize - 1),
                       linRegSize),
                strides=(plate.strides[0], plate.strides[1],
                         plate.strides[2], plate.strides[2]))

            for idX, X in enumerate(np.log2(stridedPlate)):

                for idY, Y in enumerate(X):

                    vals = []

                    for V, T in itertools.izip(Y, timesStrided):

                        vals.append(_linReg(T, V))

                    vals = np.array(vals)
                    vArgSort = vals[..., 0].argsort()

                    generationTimes[idX, idY, ...] = (
                        1.0 / vals[vArgSort[-1], self.GT_VALUE],
                        vals[vArgSort[-1], self.GT_VALUE_ERR],
                        vArgSort[-1] + posOffset,
                        1.0 / vals[vArgSort[-2], self.GT_VALUE],
                        vals[vArgSort[-2], self.GT_VALUE_ERR],
                        vArgSort[-2] + posOffset,
                    )

                    Yobs = plate[idX, idY].ravel().astype(np.float64)

                    p = Phenotyper.CalculateFitRSquare(
                        flatT, np.log2(Yobs))

                    curveFits[idX, idY, ...] = (p[0], ) + tuple(p[1])

                    if self._itermode:
                        yield

            sys.stderr.write(
                time.strftime("%Y-%m-%d %H:%M:%S\t", time.localtime()) +
                "Plate {0} Done\n".format(plateI))

        self._generationTimes = np.array(allGT)
        self._curveFits = np.array(allFits)

        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S\t", time.localtime()) +
            "Phenotype Extraction Done\n")

    @property
    def curveFits(self):

        return self._curveFits

    @property
    def generationTimes(self):

        return self._generationTimes

    @property
    def times(self):

        return self._timeObject

    @times.setter
    def times(self, value):

        assert (isinstance(value, np.ndarray) or isinstance(value, list) or
                isinstance(value, tuple)), "Invalid time series format"

        if (isinstance(value, np.ndarray) is False):
            value = np.array(value, dtype=np.float)

        self._timeObject = value

    def getPositionListFiltered(self, posList, valueType=GT_VALUE):
        """Get phenotypes for the list of positions.

        Parameters:

            posList         List of position tuples or position strings
            or mix thereof.

                            For tuples, they should be

                                    (plate, x, y)

                            For strings they should be

                                plate:x,y

        Optional Parameters:

            valueType       The type of value to obtain. Default is
            the generation time value.

        """

        values = []
        for pos in posList:

            if isinstance(pos, str):
                plate, x, y = self._posStringToTuple(pos)
            else:
                plate, x, y = pos

            values.append(self._generationTimes[plate][x, y][valueType])

        return values

    def plotRandomSampesAndSave(self, pathPattern="fig__{0}.png", n=100,
                                figure=None, figClear=False, **kwargs):

        zpos = int(np.floor(np.log10(n)) + 1)

        for i in range(n):
            figure = self.plotACurve(fig=figure, figClear=figClear,
                                     **kwargs)
            figure.savefig(pathPattern.format(str(i + 1).zfill(zpos)))

    def plotACurve(self, position=None,
                   plotRaw=True, plotSmooth=True, plotRegLine=True,
                   plotFit=True,
                   annotateGTpos=True, annotateFit=True,
                   xMarkTimes=None, plusMarkTimes=None, altMeasures=None,
                   fig=None, figClear=True):
        """Plots a curve with phenotypes marked based on a position.

        Optional Parameters:

            position        Tuple containing (plate, x, y)
            If none is submitted a random position is plotted

            plotRaw         If the raw growth data should be plotted
            Default: True

            plotSmooth      If the smoothed data should be plotted
            Default: True

            plotRegLine     If the regression line used for the GT
            extraction should be plotted
            Default: True

            annotateGTpos   If GT position used should be marked with
            arrow
            Default: True

            xMarkTimes      Times on raw data to be marked with x
            Default: None

            plusMarkTimes   Times on raw data to be marked with +
            Default: None

            altMeasures     If comparision measures should be written out
            as an iterable of (textLabel, value) tuples.
            Default: None

            fig             A figure to continue drawing on rather than
            creating a new one


            figClear        If the supplied figure should be cleared
            """

        def _markCurve(positions, colorChar):

            markIndices = np.abs(np.subtract.outer(
                self._timeObject, positions)).argmin(axis=0)

            ax.semilogy(
                self._timeObject[markIndices],
                plotRaw and self._source[position[0]][position[0]][markIndices],
                colorChar)

        if fig is not None:
            f = fig
            if figClear:
                f.clf()
        else:
            f = plt.figure()

        ax = f.gca()

        if position is None:
            position = (np.random.randint(0, self._dataObject.shape[0]),
                        np.random.randint(0, self._dataObject[0].shape[0]),
                        np.random.randint(0, self._dataObject[0].shape[1]))
        if plotRaw:
            ax.semilogy(self._timeObject,
                        self._source[position[0]][position[1:]],
                        '-b', basey=2)

        if plotSmooth:
            ax.semilogy(self._timeObject,
                        self._dataObject[position[0]][position[1:]],
                        '-g', basey=2)

        if xMarkTimes is not None:

            _markCurve(xMarkTimes, 'xr')

        if plusMarkTimes is not None:

            _markCurve(plusMarkTimes, '+k')

        tId = int(self._generationTimes[position[0]][position[1:]][
            self.GT_VALUE_POS])

        gtY = self._dataObject[position[0]][position[1:]][tId]

        if plotFit:

            Yhat = np.power(
                2.0,
                Phenotyper.ChapmanRichards4ParameterExtendedCurve(
                    self._timeObject.ravel(),
                    *self._curveFits[position[0]][position[1:]][1:]))

            ax.semilogy(self._timeObject,
                        Yhat, '--', color=(0.1, 0.1, 0.1, 0.5), basey=2)

        if plotRegLine:

            t = self._timeObject[tId]

            a = 1.0 / self._generationTimes[position[0]][position[1:]][
                self.GT_VALUE]
            b = (np.log2(gtY) - a * t)

            dT = 0.1 * self._timeObject.max() - self._timeObject.min()
            ax.semilogy([t - dT, t + dT], np.power(2, [a * (t - dT) + b,
                                                       a * (t + dT) + b]),
                        '-.k', basey=2)

        if annotateGTpos:

            ax.annotate("GT", xy=(self._timeObject[tId], gtY), arrowprops=dict(
                arrowstyle="->", connectionstyle="arc3"),
                xytext=(30, 10), textcoords="offset points")

        if annotateFit:

            ax.text(0.1, 0.85, "$R^2 = {0:.5f}$".format(
                    self._curveFits[position[0]][position[1:]][0]),
                    transform=ax.transAxes)

        measureText = "Generation Time: {0:.2f}".format(
            self._generationTimes[position[0]][position[1:]][self.GT_VALUE])

        if altMeasures is not None:

            for label, value in altMeasures:

                measureText += "\n{0}: {1:.2f}".format(label, value)

        ax.text(0.6, 0.3, measureText, transform=ax.transAxes)

        ax.text(0.1, 0.9, "Plate {0}, Row {1} Col {2}".format(*position),
                transform=ax.transAxes)

        return f

    def savePhenotypes(self, path=None, delim="\t", newline="\n"):
        """Outputs the phenotypes as a csv type format."""

        if (path is None and self._baseName is not None):
            path = self._baseName + ".csv"

        if (os.path.isfile(path)):
            if ('y' not in raw_input("Overwrite existing file? (y/N)").lower()):
                return

        fh = open(path, 'w')

        curveFits = self.curveFits

        for plateI, plate in enumerate(self.generationTimes):

            curveFitsP = curveFits[plateI]

            for idX, X in enumerate(plate):

                for idY, Y in enumerate(X):

                    fh.write("{0}{1}".format(
                        delim.join(
                            ["{0}:{1}-{2}".format(plateI, idX, idY)] +
                            map(str, ["", "", "", Y[self.GT_VALUE], ""]
                                + Y[1:].tolist() +
                                curveFitsP[idX, idY].tolist())),
                        newline))

        fh.close()

    def saveInputData(self, path=None):

        if (path is None):

            assert self._baseName is not None, "Must give path some way"

            path = self._baseName

        if (path.endswith(".npy")):
            path = path[:-4]

        source = self._source
        if (isinstance(source, xmlReader.XML_Reader)):
            source = self._xmlReader2array(source)

        np.save(path + ".data.npy", source)
        np.save(path + ".times.npy", self._timeObject)
