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
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.optimize import leastsq
from scipy.stats import linregress
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#
#   INTERNAL DEPENDENCIES
#

import _mockNumpyInterface
import scanomatic.io.xml.reader as xmlReader
import scanomatic.io.logger as logger
import scanomatic.io.paths as paths


class Phenotyper(_mockNumpyInterface.NumpyArrayInterface):

    PHEN_GT_VALUE = 0
    PHEN_GT_ERR = 1
    PHEN_GT_POS = 2
    PHEN_GT_2ND_VALUE = 3
    PHEN_GT_2ND_ERR = 4
    PHEN_GT_2ND_POS = 5
    PHEN_FIT_VALUE = 6
    PHEN_FIT_PARAM1 = 7
    PHEN_FIT_PARAM2 = 8
    PHEN_FIT_PARAM3 = 9
    PHEN_FIT_PARAM4 = 10
    PHEN_FIT_PARAM5 = 11

    NAMES_OF_PHENOTYPES = {
        0:  "Generation Time",
        1:  "Error of GT-fit",
        2:  "Time for fastest growth",
        3:  "Generation Time (2nd place)",
        4:  "Error of GT (2nd place) - fit",
        5:  "Time of second fastest growth",
        6:  "Chapman Richards model fit",
        7:  "Chapman Richards b1 (untransformed)",
        8:  "Chapman Richards b2 (untransformed)",
        9:  "Chapman Richards b3 (untransformed)",
        10:  "Chapman Richards b4 (untransformed)",
        11:  "Chapman Richards extension D (initial cells)",
    }

    def __init__(self, dataObject, timeObject=None,
                 medianKernelSize=5, gaussSigma=1.5, linRegSize=5,
                 measure=None, baseName=None, itermode=False, runAnalysis=True):

        self._paths = paths.Paths()

        self._source = dataObject

        self._phenotypes = None
        self._timeObject = None
        self._baseName = baseName

        if isinstance(dataObject, xmlReader.XML_Reader):
            arrayCopy = self._xmlReader2array(dataObject)

            if timeObject is None:
                timeObject = dataObject.get_scan_times()
        else:
            arrayCopy = dataObject.copy()

        self.times = timeObject

        assert self._timeObject is not None, "A data series needs its times"

        for plate in arrayCopy:

            assert (plate is None or
                    plate.ndim == 4 and plate.shape[-1] == 1 or
                    plate.ndim == 3), (
                        "Phenotype Strider only work with one phenotype. "
                        + "Your shape is {0}".format(plate.shape))

        super(Phenotyper, self).__init__(arrayCopy)
        self._logger = logger.Logger("Phenotyper")

        assert medianKernelSize % 2 == 1, "Median kernel size must be odd"
        self._medianKernelSize = medianKernelSize
        self._gaussSigma = gaussSigma
        self._linRegSize = linRegSize
        self._itermode = itermode

        if not self._itermode and runAnalysis:
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
    def LoadFromSate(cls, dirPath):
        """Creates an instance based on previously saved phenotyper state
        in specified directory.

        Args:

            dirPath (str):  Path to the directory holding the relevant
                            files

        Returns:

            Phenotyper instance
        """
        _p = paths.Paths()

        phenotypes = np.load(os.path.join(dirPath, _p.phenotypes_raw_npy))

        source = np.load(os.path.join(dirPath,  _p.phenotypes_input_data))

        times = np.load(os.path.join(dirPath, _p.phenotype_times))

        dataObject = np.load(os.path.join(dirPath,
                                          _p.phenotypes_input_smooth))

        medianKernelSize, gaussSigma, linRegSize = np.load(
            os.path.join(dirPath, _p.phenotypes_extraction_params))

        phenotyper = cls(source, times, medianKernelSize=medianKernelSize,
                         gaussSigma=gaussSigma, linRegSize=linRegSize,
                         runAnalysis=False)

        phenotyper._dataObject = dataObject
        phenotyper._phenotypes = phenotypes

        return phenotyper

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
        """Reterns a Chapman-Ritchards 4 parameter curve exteneded with a
        Y-axis offset D parameter.

        ''Note: The parameters b0, b1, b2 and b3 have been transposed so
        that they stay within the allowed bounds of the model

        Args:

            X (np.array):   The X-data

            b0 (float): The first parameter. To ensure that it stays within
                        the allowed bounds b0 > 0, the input b0 is
                        transposed using ``np.power(np.e, b0)``.

            b1 (float): The second parameter. The bounds are
                        1 - b3 < b1 < 1 and thus it is scaled as follows::

                            ``np.power(np.e, b1) / (np.power(np.e, b1) + 1) *
                            b3 + (1 - b3)``

                        Where ``b3`` referes to the transformed version.


            b2 (float): The third parameter, has same bounds and scaling as
                        the first

            b3 (float): The fourth parameter, has bounds 0 < b3 < 1, thus
                        scaling is done with::

                            ``np.power(np.e, b3) / (np.power(np.e, b3) + 1)``

            D (float):  Any real number, used as the offset of the curve,
                        no transformation applied.

        Returns:

            np.array.       An array of matching size as X with the
                            Chapman-Ritchards extended curve for the
                            parameter set.

        """

        #Enusuring parameters stay within the allowed bounds
        b0 = np.power(np.e, b0)
        b2 = np.power(np.e, b2)
        v = np.power(np.e, b3)
        b3 = v / (v + 1.0)
        v = np.power(np.e, b1)
        b1 = v / (v + 1.0) * b3 + (1 - b3)

        return D + b0 * np.power(1.0 - b1 * np.exp(-b2 * X), 1.0 / (1.0 - b3))

    @staticmethod
    def RCResiduals(crParams, X, Y):

        return Y - Phenotyper.ChapmanRichards4ParameterExtendedCurve(
            X, *crParams)

    @staticmethod
    def CalculateFitRSquare(
            X, Y,
            #i1 p0=np.array([1.7, -50, -2.28, -261, 14.6],
            #i2 p0=np.array([1.64, -50, -2.46, -261, 15.18],
            #p0=np.array([1.622, 21.1, -2.38, -1.99, 15.36],
            p0=np.array([1.64, -0.1, -2.46, 0.1, 15.18],
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

        return np.array([k in dataObject.get_data().keys() and
                         dataObject.get_data()[k] or None for k in
                         range(max((dataObject.get_data().keys())) + 1)])

    def _analyse(self):

        self._smoothen()
        self._calculatePhenotypes()

    def iterAnalyse(self):

        self._logger.info(
            "Iteration started, will extract {0} phenotypes".format(
                self.nPhenotypeTypes))

        if (self._itermode is False):
            raise StopIteration("Can't iterate when not in itermode")
            return
        else:
            n = sum((p.shape[0] * p.shape[1] for p in self._dataObject)) + 1.0
            i = 0.0
            self._smoothen()
            self._logger.info("Smoothed")
            yield i / n
            for x in self._calculatePhenotypes():
                self._logger.info("Phenotype extraction iteration")
                i += x
                yield i / n

        self._itermode = False

    def _smoothen(self):
        self._logger.info("Smoothing Started")
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

        self._logger.info("Smoothing Done")

    def _calculatePhenotypes(self):

        def _linReg(X, Y):
            return linregress(X, Y)[0::4]

        def _linReg2(*args):
            return linregress(args[:linRegSize], args[linRegSize:])[0::4]

        self._logger.info("Phenotype Extraction Started")

        timesStrided = np.lib.stride_tricks.as_strided(
            self._timeObject,
            shape=(self._timeObject.shape[0] - (self._linRegSize - 1),
                   self._linRegSize),
            strides=(self._timeObject.strides[0],
                     self._timeObject.strides[0]))

        flatT = self._timeObject.ravel()

        allPhenotypes = []

        linRegSize = self._linRegSize
        #linRegUFunc = np.frompyfunc(_linReg2, linRegSize * 2, 2)
        posOffset = (linRegSize - 1) / 2
        nPhenotypes = self.nPhenotypeTypes

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
            phenotypes = np.zeros((plate.shape[:2]) + (nPhenotypes,),
                                  dtype=plate.dtype)

            allPhenotypes.append(phenotypes)

            stridedPlate = np.lib.stride_tricks.as_strided(
                plate,
                shape=(plate.shape[0], plate.shape[1],
                       plate.shape[2] - (linRegSize - 1),
                       linRegSize),
                strides=(plate.strides[0], plate.strides[1],
                         plate.strides[2], plate.strides[2]))

            for idX, X in enumerate(np.log2(stridedPlate)):

                for idY, Y in enumerate(X):

                    curPhenos = [None] * nPhenotypes

                    #CALCULATING GT
                    vals = []

                    for V, T in itertools.izip(Y, timesStrided):

                        vals.append(_linReg(T, V))

                    vals = np.array(vals)
                    vArgSort = vals[..., 0].argsort()

                    #REGISTRATING GT PHENOTYPES
                    curPhenos[self.PHEN_GT_VALUE] = 1.0 / vals[
                        vArgSort[-1], self.PHEN_GT_VALUE]

                    curPhenos[self.PHEN_GT_ERR] = vals[
                        vArgSort[-1], self.PHEN_GT_ERR]

                    curPhenos[self.PHEN_GT_POS] = vArgSort[-1] + posOffset

                    curPhenos[self.PHEN_GT_2ND_VALUE] = 1.0 / vals[
                        vArgSort[-2], self.PHEN_GT_VALUE]

                    curPhenos[self.PHEN_GT_2ND_ERR] = vals[
                        vArgSort[-2], self.PHEN_GT_ERR]

                    curPhenos[self.PHEN_GT_2ND_POS] = vArgSort[-2] + posOffset

                    #CALCULATING CURVE FITS
                    Yobs = plate[idX, idY].ravel().astype(np.float64)

                    p = Phenotyper.CalculateFitRSquare(
                        flatT, np.log2(Yobs))

                    #REGISTRATING CURVE FITS
                    curPhenos[self.PHEN_FIT_VALUE] = p[0]
                    curPhenos[self.PHEN_FIT_PARAM1] = p[1][0]
                    curPhenos[self.PHEN_FIT_PARAM2] = p[1][1]
                    curPhenos[self.PHEN_FIT_PARAM3] = p[1][2]
                    curPhenos[self.PHEN_FIT_PARAM4] = p[1][3]
                    curPhenos[self.PHEN_FIT_PARAM5] = p[1][4]

                    #STORING PHENOTYPES
                    phenotypes[idX, idY, ...] = curPhenos

                if self._itermode:
                    self._logger.info("Done plate {0} pos {1} {2}".format(
                        plateI, idX, idY))
                    yield idY + 1

            self._logger.info("Plate {0} Done".format(plateI))

        self._phenotypes = np.array(allPhenotypes)

        self._logger.info("Phenotype Extraction Done")

    @property
    def nPhenotypeTypes(self):

        return max(getattr(self, attr) for attr in dir(self) if
                   attr.startswith("PHEN_")) + 1

    @property
    def curveFits(self):

        return np.array(
            [plate[..., self.PHEN_FIT_VALUE] for plate in self._phenotypes])

    @property
    def generationTimes(self):

        return np.array(
            [plate[..., self.PHEN_GT_VALUE] for plate in self._phenotypes])

    @property
    def phenotypes(self):

        return self._phenotypes

    @property
    def times(self):

        return self._timeObject

    @times.setter
    def times(self, value):

        assert (isinstance(value, np.ndarray) or isinstance(value, list) or
                isinstance(value, tuple)), "Invalid time series {0}".format(
                    value)

        if (isinstance(value, np.ndarray) is False):
            value = np.array(value, dtype=np.float)

        self._timeObject = value

    def getPositionListFiltered(self, posList, valueType=PHEN_GT_VALUE):
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

            values.append(self._phenotypes[plate][x, y][valueType])

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

        tId = int(self._phenotypes[position[0]][position[1:]][
            self.PHEN_GT_POS])

        gtY = self._dataObject[position[0]][position[1:]][tId]

        if plotFit:

            Yhat = np.power(
                2.0,
                Phenotyper.ChapmanRichards4ParameterExtendedCurve(
                    self._timeObject.ravel(),
                    self._phenotypes[position[0]][position[1:]][
                        self.PHEN_FIT_PARAM1],
                    self._phenotypes[position[0]][position[1:]][
                        self.PHEN_FIT_PARAM2],
                    self._phenotypes[position[0]][position[1:]][
                        self.PHEN_FIT_PARAM3],
                    self._phenotypes[position[0]][position[1:]][
                        self.PHEN_FIT_PARAM4],
                    self._phenotypes[position[0]][position[1:]][
                        self.PHEN_FIT_PARAM5]))

            ax.semilogy(self._timeObject,
                        Yhat, '--', color=(0.1, 0.1, 0.1, 0.5), basey=2)

        if plotRegLine:

            t = self._timeObject[tId]

            a = 1.0 / self._phenotypes[position[0]][position[1:]][
                self.PHEN_GT_VALUE]
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
                    self._phenotypes[position[0]][position[1:]][
                        self.PHEN_FIT_VALUE]),
                    transform=ax.transAxes)

        measureText = "Generation Time: {0:.2f}".format(
            self._phenotypes[position[0]][position[1:]][self.PHEN_GT_VALUE])

        if altMeasures is not None:

            for label, value in altMeasures:

                measureText += "\n{0}: {1:.2f}".format(label, value)

        ax.text(0.6, 0.3, measureText, transform=ax.transAxes)

        ax.text(0.1, 0.9, "Plate {0}, Row {1} Col {2}".format(*position),
                transform=ax.transAxes)

        if (fig is not None):
            f.show()

        return f

    def plotPlateHeatmap(self, plateIndex,
                         markPositions=[],
                         measure=None,
                         useCommonValueAxis=True,
                         vmin=None,
                         vmax=None,
                         showColorBar=True,
                         horizontalOrientation=True,
                         cm=plt.cm.RdBu_r,
                         titleText=None,
                         hideAxis=False,
                         fig=None,
                         showFig=True):

        if measure is None:
            measure = self.PHEN_GT_VALUE

        if fig is None:
            fig = plt.figure()

        fig.clf()
        ax = fig.gca()

        if (titleText is not None):
            ax.set_title(titleText)

        plateData = self.phenotypes[plateIndex][..., measure]
        if not horizontalOrientation:
            plateData = plateData.T

        if (None not in (vmin, vmax)):
            pass
        elif (useCommonValueAxis):
            vmin, vmax = zip(*[
                (p[..., measure][np.isfinite(p[..., measure])].min(),
                 p[..., measure][np.isfinite(p[..., measure])].max())
                for p in self.phenotypes if p is not None])
            vmin = min(vmin)
            vmax = max(vmax)
        else:
            vmin = plateData[np.isfinite(plateData)].min()
            vmax = plateData[np.isfinite(plateData)].max()

        im = ax.imshow(
            plateData,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            cmap=cm)

        if (showColorBar):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", "5%", pad="3%")
            plt.colorbar(im, cax=cax)

        if (hideAxis):
            ax.set_axis_off()

        fig.tight_layout()
        if (showFig):
            fig.show()

        return fig

    def savePhenotypes(self, path=None, delim="\t", newline="\n",
                       askOverwrite=True):
        """Outputs the phenotypes as a csv type format."""

        if (path is None and self._baseName is not None):
            path = self._baseName + ".csv"

        if (os.path.isfile(path) and askOverwrite):
            if ('y' not in raw_input("Overwrite existing file? (y/N)").lower()):
                return False

        fh = open(path, 'w')

        for plateI, plate in enumerate(self.phenotypes):

            for idX, X in enumerate(plate):

                for idY, Y in enumerate(X):

                    fh.write("{0}{1}".format(
                        delim.join(
                            ["{0}:{1}-{2}".format(plateI, idX, idY),
                             "Unknown Strain",
                             "Unknown Condition"] +
                            map(str, ["None", Y[self.PHEN_GT_VALUE], "None",
                                      Y[self.PHEN_FIT_VALUE]])), newline))

        fh.close()
        return True

    @staticmethod
    def _saveOverwriteAsk(path):
        return raw_input("Overwrite '{0}' (y/N)".format(
            path)).strip().upper().startswith("Y")

    def saveState(self, dirPath, askOverwrite=True):

        p = os.path.join(dirPath, self._paths.phenotypes_raw_npy)
        if (not askOverwrite or not os.path.isfile(p) or
                self._saveOverwriteAsk(p)):
            np.save(p, self.phenotypes)

        p = os.path.join(dirPath, self._paths.phenotypes_input_data)
        if (not askOverwrite or not os.path.isfile(p) or
                self._saveOverwriteAsk(p)):
            np.save(p, self._source)

        p = os.path.join(dirPath, self._paths.phenotypes_input_smooth)
        if (not askOverwrite or not os.path.isfile(p) or
                self._saveOverwriteAsk(p)):
            np.save(p, self._dataObject)

        p = os.path.join(dirPath, self._paths.phenotype_times)
        if (not askOverwrite or not os.path.isfile(p) or
                self._saveOverwriteAsk(p)):
            np.save(p, self._timeObject)

        p = os.path.join(dirPath, self._paths.phenotypes_extraction_params)
        if (not askOverwrite or not os.path.isfile(p) or
                self._saveOverwriteAsk(p)):
            np.save(
                p,
                [self._medianKernelSize,
                 self._gaussSigma,
                 self._linRegSize])

        self._logger.info("State saved to '{0}'".format(dirPath))

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
