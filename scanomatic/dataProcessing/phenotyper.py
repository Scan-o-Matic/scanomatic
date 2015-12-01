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
from types import StringTypes
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.stats import linregress
import itertools
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

#
#   INTERNAL DEPENDENCIES
#

import _mockNumpyInterface
import scanomatic.io.xml.reader as xmlReader
import scanomatic.io.logger as logger
import scanomatic.io.paths as paths
import scanomatic.io.image_data as image_data
from scanomatic.dataProcessing.growth_phenotypes import Phenotypes, CalculateFitRSquare


class Phenotyper(_mockNumpyInterface.NumpyArrayInterface):
    """The Phenotyper class is a class for producing phenotypes
    based on growth curves as well as storing and easy displaying them.

    There are fource modes of instanciating a phenotyper instance:

    <code>
    #Generic instanciation
    p = Phenotyper(...)

    #Instanciation from xml based data
    p = Phenotyper.LoadFromXML(...)

    #Instanciation from numpy data
    p = Phenotyper.LoadFromNumPy(...)

    #Instanciation from a saved phenotyper-state
    #this method will not try to make new phenotypes
    p = Phenotyper.LoadFromState(...)
    </code>

    The names of the currently supported phenotypes are stored in static
    dictionary lookup <code>Phenotyper.NAMES_OF_PHENOTYPES</code>.

    The matching lookup-keys for accessing specific phenotype indices in the
    phenotypes array are stored as static integers on the class following
    the pattern <code>Phenotyper.PHEN_*</code>. 
    """

    def __init__(self, dataObject, timeObject=None,
                 medianKernelSize=5, gaussSigma=1.5, linRegSize=5,
                 measure=None, baseName=None, itermode=False, runAnalysis=True):

        self._paths = paths.Paths()

        self._raw_growth_data = dataObject

        self._phenotypes = None
        self._timeObject = None
        self._base_name = baseName

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

        self._removeFilter = np.array([None for _ in self._smooth_growth_data],
                                      dtype=np.object)

        self._logger = logger.Logger("Phenotyper")

        assert medianKernelSize % 2 == 1, "Median kernel size must be odd"
        self._median_kernel_size = medianKernelSize
        self._gaussSigma = gaussSigma
        self._linRegSize = linRegSize
        self._itermode = itermode
        self._metaData = None

        if not self._itermode and runAnalysis:
            self._analyse()

    @classmethod
    def phenotypeNames(cls):

        return (p for p in cls.__dict__ if p.startswith("PHEN_"))

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
    def LoadFromState(cls, dirPath):
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

        phenotyper._smooth_growth_data = dataObject
        phenotyper._phenotypes = phenotypes

        p = os.path.join(dirPath, _p.phenotypes_filter)
        if os.path.isfile(p):
            pFilter = np.load(p)
            if all(p.shape == pFilter[i].shape for i, p in enumerate(phenotypes)
                   if p is not None and pFilter[i] is not None):

                #phenotypes._removeFilter = pFilter
                pass

        return phenotyper

    @classmethod
    def LoadFromImageData(cls, path='.'):

        times, data = image_data.ImageData.read_image_data_and_time(path)
        return cls(data, times)

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
            if path.endswith(".data"):
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

    @property
    def metaData(self):

        return self._metaData

    @metaData.setter
    def metaData(self, val):

        self._metaData = val

    @property
    def raw_growth_data(self):

        return self._raw_growth_data

    @property
    def smooth_growth_data(self):

        return self._smooth_growth_data

    def _xmlReader2array(self, dataObject):

        return np.array([k in dataObject.get_data().keys() and
                         dataObject.get_data()[k] or None for k in
                         range(max((dataObject.get_data().keys())) + 1)])

    def _analyse(self):

        self._smoothen()
        self._calculate_phenotypes()

    def iterAnalyse(self):

        self._logger.info(
            "Iteration started, will extract {0} phenotypes".format(
                self.nPhenotypeTypes))

        if self._itermode is False:
            raise StopIteration("Can't iterate when not in itermode")
            return

        else:

            self._smoothen()
            self._logger.info("Smoothed")
            yield 0
            for x in self._calculate_phenotypes():
                self._logger.debug("Phenotype extraction iteration")
                yield x

        self._itermode = False

    def _smoothen(self):

        self._logger.info("Smoothing Started")
        median_kernel = np.ones((1, self._median_kernel_size))

        for plate in self._smooth_growth_data:

            plate_as_flat = np.lib.stride_tricks.as_strided(
                plate,
                shape=(plate.shape[0] * plate.shape[1], plate.shape[2]),
                strides=(plate.strides[1], plate.strides[2]))

            plate_as_flat[...] = median_filter(
                plate_as_flat, footprint=median_kernel, mode='reflect')

            plate_as_flat[...] = gaussian_filter1d(
                plate_as_flat, sigma=self._gaussSigma, mode='reflect', axis=-1)

        self._logger.info("Smoothing Done")

    def _calculate_phenotypes(self):

        def _linreg_helper(X, Y):
            return linregress(X, Y)[0::4]

        if self._timeObject.shape[0] - (self._linRegSize - 1) <= 0:
            self._logger.error(
                "Refusing phenotype extractions since number of scans are less than used in the linear regression")
            return

        times_strided = np.lib.stride_tricks.as_strided(
            self._timeObject,
            shape=(self._timeObject.shape[0] - (self._linRegSize - 1),
                   self._linRegSize),
            strides=(self._timeObject.strides[0],
                     self._timeObject.strides[0]))

        flat_times = self._timeObject.ravel()

        index_for_48h = np.abs(np.subtract.outer(self._timeObject, [48])).argmin()

        all_phenotypes = []

        regression_size = self._linRegSize
        position_offset = (regression_size - 1) / 2
        phenotypes_count = self.nPhenotypeTypes
        total_curves = float(
            sum(p.shape[0] * p.shape[1] if (p is not None and p.ndim > 1) else 0 for p in self._smooth_growth_data))

        self._logger.info("Phenotypes (N={0}) Extraction Started".format(
            phenotypes_count))

        for plateI, plate in enumerate(self._smooth_growth_data):

            plate_flat_regression_strided = np.lib.stride_tricks.as_strided(
                plate,
                shape=(plate.shape[0] * plate.shape[1],
                       plate.shape[2] - (regression_size - 1),
                       regression_size),
                strides=(plate.strides[1],
                         plate.strides[2], plate.strides[2]))

            phenotypes = np.zeros((plate.shape[:2]) + (phenotypes_count,),
                                  dtype=plate.dtype)

            all_phenotypes.append(phenotypes)

            for pos_index, pos_data in enumerate(np.log2(plate_flat_regression_strided)):

                position_phenotypes = [None] * phenotypes_count

                linreg_values= []

                for times, value_segment in itertools.izip(times_strided, pos_data):

                    linreg_values.append(_linreg_helper(times, value_segment))

                derivative_values, derivative_errors = np.array(linreg_values).T

                id0 = pos_index % plate.shape[0]
                id1 = pos_index / plate.shape[0]

                curve_data = {'curve_smooth_growth_data': np.ma.masked_invalid(plate[id0, id1]),
                              'index48h': index_for_48h,
                              'chapman_richards_fit': CalculateFitRSquare(
                                  flat_times, np.log2(plate[id0, id1].ravel().astype(np.float64))),
                              'derivative_values': derivative_values,
                              'derivative_errors': derivative_errors,
                              'linregress_extent': position_offset,
                              'flat_times': flat_times}

                for phenotype in Phenotypes:
                    position_phenotypes[phenotype.value] = phenotype(**curve_data)

                phenotypes[id0, id1, ...] = position_phenotypes

                if self._itermode:
                    self._logger.debug("Done plate {0} pos {1} {2} {3}".format(
                        plateI, id0, id1, list(position_phenotypes)))
                    yield (pos_index + 1.0) / total_curves

            self._logger.info("Plate {0} Done".format(plateI))

        self._phenotypes = np.array(all_phenotypes)

        self._logger.info("Phenotype Extraction Done")

    @property
    def nPhenotypeTypes(self):

        return max(getattr(self, attr) for attr in dir(self) if
                   attr.startswith("PHEN_")) + 1

    @property
    def nPhenotypesInData(self):

        return max((p is None and 0 or p.shape[-1]) for p in self._phenotypes)

    @property
    def curveFits(self):

        return np.array(
            [plate[..., self.PHEN_FIT_VALUE] for plate in self.phenotypes])

    @property
    def generationTimes(self):

        return np.array(
            [plate[..., self.PHEN_GT_VALUE] for plate in self.phenotypes])

    @property
    def phenotypes(self):

        ret = []
        if self._phenotypes is None:
            return None

        for i, p in enumerate(self._phenotypes):
            if (self._removeFilter[i] is not None and p is not None):
                ret.append(
                    np.ma.masked_array(p, self._removeFilter[i],
                                       fill_value=np.nan))
            else:
                ret.append(p)

        return ret

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

    def padPhenotypes(self):

        padding = self.nPhenotypeTypes - self.nPhenotypesInData

        if (padding):
            self._logger.info(
                "Padding phenotypes, adding" +
                " {0} to become {1}, current shape {2}".format(
                    padding,
                    self.nPhenotypeTypes,
                    self._phenotypes.shape))

            phenotypes = []
            removes = []
            for i, p in enumerate(self._phenotypes):

                if p is not None:
                    pad = np.zeros(p.shape[:-1] + (padding,))
                    phenotypes.append(np.dstack((p, pad * np.nan)))
                    removes.append(np.dstack((p, pad == 0)))
                else:
                    removes.append(None)
                    phenotypes.append(None)

            self._phenotypes = np.array(phenotypes)
            self._removeFilter = np.array(removes)

            self._logger.info(
                "New phenotypes shapes {0}".format(
                    self._phenotypes.shape))

        return padding

    def _checkFilterInit(self, plate):

        if (not(hasattr(self._removeFilter[plate], "shape")) or
                self._removeFilter[plate].shape !=
                self._phenotypes[plate].shape):

            self._removeFilter[plate] = np.zeros(
                self._phenotypes[plate].shape,
                dtype=np.bool)

    def add2RemoveFilter(self, plate, positionList, phenotype=None):
        """Adds positions as removed from data.

        Args:

            plate (int):    The plate

            positionList (iterable):    A list of X and Y coordinates as
                                        returned by np.where

        Kwargs:

            phenotype (int/None):   What phenotype to invoke filter on
                                    or if None to invoke on all
        """

        if (self._phenotypes is None or self._phenotypes[plate] is None):
            raise IndexError("No phenotypes known for plate {0}".format(plate))

        self._checkFilterInit(plate)
        self._removeFilter[plate][positionList] = True

    def getRemoveFilter(self, plate):
        """Get remove filter for plate.

        Args:

            plate (int)   Index of plate

        Returns:

            numpy.ndarray (dtype=np.bool)
                The per position status of removal
        """

        self._checkFilterInit(plate)
        return self._removeFilter[plate]

    def hasRemoved(self, plate):
        """Get if plate has anything removed.

        Args:

            plate (int)   Index of plate

        Returns:

            bool    The status of the plate removals
        """

        return self.getRemoveFilter(plate).any()

    def hasAnyRemoved(self):
        """If any plate has anything removed

        Returns:
            bool    The removal status
        """
        return any(self.hasRemoved(i) for i in
                   range(self._removeFilter.shape[0]))

    def get_position_list_filtered(self, position_list, value_type=Phenotypes.GenerationTime):


        values = []
        for pos in position_list:

            if isinstance(pos, StringTypes):
                plate, x, y = self._posStringToTuple(pos)
            else:
                plate, x, y = pos

            values.append(self.phenotypes[plate][x, y][value_type.value])

        return values

    def plot_plate_heatmap(self, plate_index,
                           measure=None,
                           data=None,
                           use_common_value_axis=True,
                           vmin=None,
                           vmax=None,
                           show_color_bar=True,
                           horizontal_orientation=True,
                           cm=plt.cm.RdBu_r,
                           title_text=None,
                           hide_axis=False,
                           fig=None,
                           show_figure=True):

        if measure is None:
            measure = Phenotypes.GenerationTime.value
        elif isinstance(measure, Phenotypes):
            measure = measure.value

        if fig is None:
            fig = plt.figure()

        cax = None

        if len(fig.axes):
            ax = fig.axes[0]
            if len(fig.axes) == 2:
                cax = fig.axes[1]
                cax.cla()
                fig.delaxes(cax)
                cax = None
            ax.cla()
        else:
            ax = fig.gca()

        if title_text is not None:
            ax.set_title(title_text)

        if data is None:
            data = self.phenotypes

        plate_data = data[plate_index][..., measure]

        if not horizontal_orientation:
            plate_data = plate_data.T

        if plate_data[np.isfinite(plate_data)].size == 0:
            self._logger.error("No finite data")
            return False

        if None not in (vmin, vmax):
            pass
        elif use_common_value_axis:
            vmin, vmax = zip(*[
                (p[..., measure][np.isfinite(p[..., measure])].min(),
                 p[..., measure][np.isfinite(p[..., measure])].max())
                for p in data if p is not None])
            vmin = min(vmin)
            vmax = max(vmax)
        else:
            vmin = plate_data[np.isfinite(plate_data)].min()
            vmax = plate_data[np.isfinite(plate_data)].max()

        font = {'family': 'sans',
                'weight': 'normal',
                'size': 6}

        matplotlib.rc('font', **font)

        im = ax.imshow(
            plate_data,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            cmap=cm)

        if show_color_bar:
            divider = make_axes_locatable(ax)
            if cax is None:
                cax = divider.append_axes("right", "5%", pad="3%")
            plt.colorbar(im, cax=cax)

        if hide_axis:
            ax.set_axis_off()

        fig.tight_layout()
        if show_figure:
            fig.show()

        return fig

    def save_phenotypes(self, path=None, data=None, data_headers=None, delim="\t", newline="\n", ask_if_overwrite=True):

        if path is None and self._base_name is not None:
            path = self._base_name + ".csv"

        if os.path.isfile(path) and ask_if_overwrite:
            if 'y' not in raw_input("Overwrite existing file? (y/N)").lower():
                return False

        headers = ('Plate', 'Row', 'Column')

        # USING RAW PHENOTYPE DATA
        if data is None:
            data_headers = tuple(phenotype.name for phenotype in Phenotypes)
            self._logger.info("Using raw phenotypes")
            data = self.phenotypes

        if data is None:
            self._logger.warning("Could not save data since there is no data")
            return False

        with open(path, 'w') as fh:

            # SAVES OUT DATA AS NPY AS WELL
            np.save(path + ".npy", data)

            # HEADER ROW
            meta_data = self._metaData
            all_headers_identical = True
            meta_data_headers = tuple()

            if meta_data is not None:
                self._logger.info("Using meta-data")
                meta_data_headers = meta_data.getHeaderRow(0)
                for plate_index in range(1, len(data)):
                    if meta_data_headers != meta_data.getHeaderRow(plate_index):
                        all_headers_identical = False
                        break
                meta_data_headers = tuple(meta_data_headers)

            if all_headers_identical:
                fh.write("{0}{1}".format(delim.join(
                    map(str, headers + meta_data_headers + data_headers)), newline))

            # DATA
            for plate_index, plate in enumerate(data):

                if not all_headers_identical:
                    fh.write("{0}{1}".format(delim.join(
                        map(str, headers + tuple(meta_data.getHeaderRow(plate_index)) +
                            data_headers)), newline))

                for idX, X in enumerate(plate):

                    for idY, Y in enumerate(X):

                        if meta_data is None:
                            fh.write("{0}{1}".format(delim.join(map(
                                str, [plate_index, idX, idY] + Y.tolist())), newline))
                        else:
                            fh.write("{0}{1}".format(delim.join(map(
                                str, [plate_index, idX, idY] + meta_data(plate_index, idX, idY) +
                                     Y.tolist())), newline))

        self._logger.info("Saved csv absolute phenotypes to {0}".format(path))

        return True

    @staticmethod
    def _do_ask_overwrite(path):
        return raw_input("Overwrite '{0}' (y/N)".format(
            path)).strip().upper().startswith("Y")

    def save_state(self, dir_path, ask_if_overwrite=True):

        p = os.path.join(dir_path, self._paths.phenotypes_raw_npy)
        if (not ask_if_overwrite or not os.path.isfile(p) or
                self._do_ask_overwrite(p)):
            np.save(p, self._phenotypes)

        p = os.path.join(dir_path, self._paths.phenotypes_input_data)
        if (not ask_if_overwrite or not os.path.isfile(p) or
                self._do_ask_overwrite(p)):
            np.save(p, self._raw_growth_data)

        p = os.path.join(dir_path, self._paths.phenotypes_input_smooth)
        if (not ask_if_overwrite or not os.path.isfile(p) or
                self._do_ask_overwrite(p)):
            np.save(p, self._smooth_growth_data)

        p = os.path.join(dir_path, self._paths.phenotypes_filter)
        if (not ask_if_overwrite or not os.path.isfile(p) or
                self._do_ask_overwrite(p)):
            np.save(p, self._removeFilter)

        p = os.path.join(dir_path, self._paths.phenotype_times)
        if (not ask_if_overwrite or not os.path.isfile(p) or
                self._do_ask_overwrite(p)):
            np.save(p, self._timeObject)

        p = os.path.join(dir_path, self._paths.phenotypes_extraction_params)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(
                p,
                [self._median_kernel_size,
                 self._gaussSigma,
                 self._linRegSize])

        self._logger.info("State saved to '{0}'".format(dir_path))

    def save_input_data(self, path=None):

        if path is None:

            assert self._base_name is not None, "Must give path some way"

            path = self._base_name

        if path.endswith(".npy"):
            path = path[:-4]

        source = self._raw_growth_data
        if isinstance(source, xmlReader.XML_Reader):
            source = self._xmlReader2array(source)

        np.save(path + ".data.npy", source)
        np.save(path + ".times.npy", self._timeObject)
