import numpy as np
import os
from types import StringTypes
from scipy.ndimage import median_filter, gaussian_filter1d
from collections import deque
from enum import Enum

#
#   INTERNAL DEPENDENCIES
#

import _mockNumpyInterface
import scanomatic.io.xml.reader as xml_reader_module
import scanomatic.io.logger as logger
import scanomatic.io.paths as paths
import scanomatic.io.image_data as image_data
from scanomatic.dataProcessing.growth_phenotypes import Phenotypes, get_preprocessed_data_for_phenotypes,\
    PhenotypeDataType, get_derivative


class PositionMark(Enum):

    OK = 0
    NoGrowth = 1
    BadData = 2


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

    def __init__(self, raw_growth_data, times_data=None,
                 median_kernel_size=5, gaussian_filter_sigma=1.5, linear_regression_size=5,
                 phenotypes=None, base_name=None, run_extraction=False,
                 phenotypes_inclusion=PhenotypeDataType.Trusted):

        self._paths = paths.Paths()

        self._raw_growth_data = raw_growth_data
        self._smooth_growth_data = None
        self._phenotypes = None
        self._times_data = None
        self._limited_phenotypes = phenotypes
        self._phenotypes_inclusion = phenotypes_inclusion
        self._base_name = base_name

        if isinstance(raw_growth_data, xml_reader_module.XML_Reader):
            array_copy = self._xml_reader_2_array(raw_growth_data)

            if times_data is None:
                times_data = raw_growth_data.get_scan_times()
        else:
            array_copy = raw_growth_data.copy()

        self.times = times_data

        assert self._times_data is not None, "A data series needs its times"

        for plate in array_copy:

            assert (plate is None or
                    plate.ndim == 4 and plate.shape[-1] == 1 or
                    plate.ndim == 3), (
                        "Phenotype Strider only work with one phenotype. "
                        + "Your shape is {0}".format(plate.shape))

        super(Phenotyper, self).__init__(array_copy)

        self._removed_filter = np.array([None for _ in self._smooth_growth_data], dtype=np.object)
        self._remove_actions = None
        self._init_remove_filter_and_undo_actions()

        self._logger = logger.Logger("Phenotyper")

        assert median_kernel_size % 2 == 1, "Median kernel size must be odd"
        self._median_kernel_size = median_kernel_size
        self._gaussian_filter_sigma = gaussian_filter_sigma
        self._linear_regression_size = linear_regression_size
        self._meta_data = None

        if run_extraction:
            self.extract_phenotypes()

    def set_phenotype_inclusion_level(self, value):
        if isinstance(value, PhenotypeDataType):
            self._phenotypes_inclusion = value
        else:
            self._logger.error("Value not a PhenotypeDataType!")

    def phenotype_names(self):

        return tuple(p.name for p in Phenotypes if not self._limited_phenotypes or p in self._limited_phenotypes)

    @classmethod
    def LoadFromXML(cls, path, **kwargs):
        """Class Method used to create a Phenotype Strider directly
        from a path do an xml

        Parameters:

            path        The path to the xml-file

        Optional Parameters can be passed as keywords and will be
        used in instanciating the class.
        """

        xml = xml_reader_module.XML_Reader(path)
        if path.lower().endswith(".xml"):
            path = path[:-4]

        return cls(xml, base_name=path, run_extraction=True, **kwargs)

    @classmethod
    def LoadFromState(cls, directory_path):
        """Creates an instance based on previously saved phenotyper state
        in specified directory.

        Args:

            dirPath (str):  Path to the directory holding the relevant
                            files

        Returns:

            Phenotyper instance
        """
        _p = paths.Paths()

        phenotypes = np.load(os.path.join(directory_path, _p.phenotypes_raw_npy))

        raw_growth_data = np.load(os.path.join(directory_path,  _p.phenotypes_input_data))

        times = np.load(os.path.join(directory_path, _p.phenotype_times))

        smooth_growth_data = np.load(os.path.join(directory_path, _p.phenotypes_input_smooth))

        median_filt_size, gauss_sigma, linear_reg_size = np.load(
            os.path.join(directory_path, _p.phenotypes_extraction_params))

        phenotyper = cls(raw_growth_data, times, median_kernel_size=median_filt_size,
                         gaussian_filter_sigma=gauss_sigma, linear_regression_size=linear_reg_size,
                         run_extraction=False, base_name=directory_path)

        phenotyper.set('smooth_growth_data',smooth_growth_data)
        phenotyper.set('phenotypes', phenotypes)

        filter_path = os.path.join(directory_path, _p.phenotypes_filter)
        if os.path.isfile(filter_path):
            # TODO: Need implementation to load remove_state correctly
            """
            phenotype_remove_filter = np.load(filter_path)
            if all(p.shape == phenotype_remove_filter[i].shape for i, p in enumerate(phenotypes)
                   if p is not None and phenotype_remove_filter[i] is not None):

                # phenotypes._removeFilter = phenotype_remove_filter
                pass
            """
        return phenotyper

    @classmethod
    def LoadFromImageData(cls, path='.'):

        times, data = image_data.ImageData.read_image_data_and_time(path)
        return cls(data, times, run_extraction=True)

    @classmethod
    def LoadFromNumPy(cls, path, times_data_path=None, **kwargs):
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
        data_directory = path
        if path.lower().endswith(".npy"):
            path = path[:-4]
            if path.endswith(".data"):
                path = path[:-5]

        if times_data_path is None:
            times_data_path = path + ".times.npy"

        if not os.path.isfile(data_directory):
            if os.path.isfile(times_data_path + ".data.npy"):

                times_data_path += ".data.npy"

            elif os.path.isfile(times_data_path + ".npy"):

                times_data_path += ".npy"

        return cls(np.load(data_directory), np.load(times_data_path), base_name=path, run_extraction=True
                   **kwargs)

    @property
    def meta_data(self):

        return self._meta_data

    @meta_data.setter
    def meta_data(self, val):

        self._meta_data = val

    @property
    def raw_growth_data(self):

        return self._raw_growth_data

    @property
    def smooth_growth_data(self):

        return self._smooth_growth_data

    @staticmethod
    def _xml_reader_2_array(data_object):

        return np.array([k in data_object.get_data().keys() and
                         data_object.get_data()[k] or None for k in
                         range(max((data_object.get_data().keys())) + 1)])

    def iterate_extraction(self):

        self._logger.info(
            "Iteration started, will extract {0} phenotypes".format(
                self.number_of_phenotypes))

        if not self.has_smooth_growth_data:
            self._smoothen()
        self._logger.info("Smoothed")
        yield 0
        for x in self._calculate_phenotypes():
            self._logger.debug("Phenotype extraction iteration")
            yield x

    def wipe_extracted_phenotypes(self):

        self._phenotypes = None

    def extract_phenotypes(self):

        self._logger.info("Extracting phenotypes. This will take a while...")

        if not self.has_smooth_growth_data:
            self._smoothen()

        for _ in self._calculate_phenotypes():
            pass

        self._logger.info("Phenotypes extracted")

    @property
    def has_smooth_growth_data(self):

        if self._smooth_growth_data is None or len(self._smooth_growth_data) != len(self._raw_growth_data):
            return False

        return all((a is None == b is None ) or a.shape == b.shape for a, b in zip(self._raw_growth_data, self._smooth_growth_data))

    def _smoothen(self):

        self.set("smooth_growth_data", self._raw_growth_data.copy())
        self._logger.info("Smoothing Started")
        median_kernel = np.ones((1, self._median_kernel_size))

        for plate in self._smooth_growth_data:

            if plate is None:
                continue

            plate_as_flat = np.lib.stride_tricks.as_strided(
                plate,
                shape=(plate.shape[0] * plate.shape[1], plate.shape[2]),
                strides=(plate.strides[1], plate.strides[2]))

            plate_as_flat[...] = median_filter(
                plate_as_flat, footprint=median_kernel, mode='reflect')

            plate_as_flat[...] = gaussian_filter1d(
                plate_as_flat, sigma=self._gaussian_filter_sigma, mode='reflect', axis=-1)

        self._logger.info("Smoothing Done")

    def _calculate_phenotypes(self):

        if self._times_data.shape[0] - (self._linear_regression_size - 1) <= 0:
            self._logger.error(
                "Refusing phenotype extractions since number of scans are less than used in the linear regression")
            return

        times_strided = self.times_strided

        flat_times = self._times_data

        index_for_48h = np.abs(np.subtract.outer(self._times_data, [48])).argmin()

        all_phenotypes = []

        regression_size = self._linear_regression_size
        position_offset = (regression_size - 1) / 2
        phenotypes_count = self.number_of_phenotypes
        total_curves = float(self.number_of_curves)

        self._logger.info("Phenotypes (N={0}), extraction started for {1} curves".format(
            phenotypes_count, int(total_curves)))

        curves_in_completed_plates = 0
        phenotypes_inclusion = self._phenotypes_inclusion

        if phenotypes_inclusion is not PhenotypeDataType.Trusted:
            self._logger.warning("Will extract phenotypes beyond those that are trusted, this is not recommended!" +
                                 " It is your responsibility to verify the validity of those phenotypes!")

        for plateI, plate in enumerate(self._smooth_growth_data):

            if plate is None:
                all_phenotypes.append(None)
                continue

            plate_flat_regression_strided = self._get_plate_linear_regression_strided(plate)

            phenotypes = np.zeros((plate.shape[:2]) + (phenotypes_count,),
                                  dtype=np.float)

            all_phenotypes.append(phenotypes)

            for pos_index, pos_data in enumerate(plate_flat_regression_strided):

                position_phenotypes = [None] * phenotypes_count

                id1 = pos_index % plate.shape[1]
                id0 = pos_index / plate.shape[1]

                curve_data = get_preprocessed_data_for_phenotypes(
                    curve=plate[id0, id1],
                    curve_strided=pos_data,
                    flat_times=flat_times,
                    times_strided=times_strided,
                    index_for_48h=index_for_48h,
                    position_offset=position_offset)

                for phenotype in Phenotypes:

                    if PhenotypeDataType.Scalar(phenotype) and phenotypes_inclusion(phenotype):

                        position_phenotypes[phenotype.value] = phenotype(**curve_data)

                phenotypes[id0, id1, ...] = position_phenotypes

                if id0 == 0:
                    self._logger.debug("Done plate {0} pos {1} {2} {3}".format(
                        plateI, id0, id1, list(position_phenotypes)))
                    yield (curves_in_completed_plates + pos_index + 1.0) / total_curves

            self._logger.info("Plate {0} Done".format(plateI + 1))
            curves_in_completed_plates += 0 if plate is None else plate_flat_regression_strided.shape[0]

        self._phenotypes = np.array(all_phenotypes)

        self._logger.info("Phenotype Extraction Done")

    def _get_plate_linear_regression_strided(self, plate):

        if plate is None:
            return None

        return np.lib.stride_tricks.as_strided(
            plate,
            shape=(plate.shape[0] * plate.shape[1],
                   plate.shape[2] - (self._linear_regression_size - 1),
                   self._linear_regression_size),
            strides=(plate.strides[1],
                     plate.strides[2], plate.strides[2]))

    @property
    def number_of_curves(self):

        return sum(p.shape[0] * p.shape[1] if (p is not None and p.ndim > 1) else 0 for p in self._raw_growth_data)

    @property
    def number_of_phenotypes(self):

        if self._phenotypes is not None:
            phenotypes = np.unique(tuple(p.shape[2] for p in self._phenotypes if p is not None and p.ndim == 3))
            if phenotypes.size == 1:
                return phenotypes[0]

        return len(Phenotypes) if not self._limited_phenotypes else len(self._limited_phenotypes)

    @property
    def generation_times(self):

        return np.array(
            [plate[..., Phenotypes.GenerationTime.value] for plate in self.phenotypes])

    @property
    def phenotypes(self):

        ret = []
        if self._phenotypes is None:
            return None

        for i, p in enumerate(self._phenotypes):
            if p is not None:
                filtered_plate = np.ma.masked_array(
                    p.copy(), self._removed_filter[i] == PositionMark.BadData.value, fill_value=np.nan)
                filtered_plate[self._removed_filter[i] == PositionMark.NoGrowth.value] = np.inf
                ret.append(filtered_plate)
            else:
                ret.append(p)

        return ret

    def get_phenotype(self, phenotype):

        def _plate_type_converter_vector(plate):

            out = np.zeros(plate.shape + plate[0, 0].shape, dtype=plate[0, 0].dtype)
            if out.dtype == np.floating:
                out *= np.nan

            for out_pos, in_pos in zip(out.reshape(out.shape[0] * out.shape[1], out.shape[2]), plate.ravel()):
                out_pos[...] = in_pos

            return out

        def _plate_type_converter_scalar(plate):

            dtype = type(plate[0, 0])
            out = np.zeros(plate.shape, dtype=dtype)

            if issubclass(type(out.dtype), np.floating):
                out *= np.nan

            out[...] = plate

            return out

        def _plate_type_converter(plate):

            if plate.ndim == 3:
                return _plate_type_converter_vector(plate)
            else:
                return _plate_type_converter_scalar(plate)

        if self._phenotypes is None or \
                    self._limited_phenotypes and phenotype not in self._limited_phenotypes or \
                    phenotype.value >= self.number_of_phenotypes:

            raise ValueError(
                "'{0}' has not been extracted, please re-run 'extract_phenotypes()' to include it.".format(
                phenotype.name))

        if not PhenotypeDataType.Trusted(phenotype):
            self._logger.warning("The phenotype '{0}' has not been fully tested and verified!".format(phenotype.name))

        return [p if p is None else _plate_type_converter(p[..., phenotype.value]) for p in self.phenotypes]

    @property
    def times(self):

        return self._times_data

    @times.setter
    def times(self, value):

        assert (isinstance(value, np.ndarray) or isinstance(value, list) or
                isinstance(value, tuple)), "Invalid time series {0}".format(
                    value)

        if isinstance(value, np.ndarray) is False:
            value = np.array(value, dtype=np.float)

        self._times_data = value

    @property
    def times_strided(self):

        return np.lib.stride_tricks.as_strided(
            self._times_data,
            shape=(self._times_data.shape[0] - (self._linear_regression_size - 1),
                   self._linear_regression_size),
            strides=(self._times_data.strides[0],
                     self._times_data.strides[0]))

    def get_derivative(self, plate, position):

        return get_derivative(
            self._get_plate_linear_regression_strided(
                self.smooth_growth_data[plate][position].reshape(1, 1, self.times.size))[0],
                self.times_strided)[0]

    def set(self, data_type, data):

        if data_type == 'phenotypes':

            self._phenotypes = data
            if isinstance(data, np.ndarray) and (data.size == 0 or not data.any()):
                self._phenotypes = None

            self._init_remove_filter_and_undo_actions()

        elif data_type == 'smooth_growth_data':

            self._smooth_growth_data = data

        else:

            self._logger.warning('Unknown type of data {0}'.format(data_type))

    def _init_remove_filter_and_undo_actions(self):

        for plate_index in range(self._removed_filter.shape[0]):
            if self._raw_growth_data[plate_index] is None:
                continue
            self._removed_filter[plate_index] = np.zeros(
                self._raw_growth_data[plate_index].shape[:2] + (self.number_of_phenotypes,), dtype=np.int8)

        self._remove_actions = tuple(deque() for _ in self._smooth_growth_data)

    def add_position_mark(self, plate, position_list, phenotype=None, position_mark=PositionMark.BadData):

        if position_mark is PositionMark.NoGrowth or phenotype is None:
            phenotype = slice(None, None, None)
        else:
            phenotype = phenotype.value

        previous_state = self._removed_filter[plate][position_list, phenotype]
        if isinstance(previous_state, np.array):
            previous_state = np.unique(previous_state)
            if previous_state.size > 1:
                previous_state = 0
            else:
                previous_state = previous_state[0]

        self._remove_actions[plate].append((position_list, phenotype, previous_state))
        self._removed_filter[plate][position_list, phenotype] = position_mark.value

    def plate_has_any_colonies_removed(self, plate):
        """Get if plate has anything removed.

        Args:

            plate (int)   Index of plate

        Returns:

            bool    The status of the plate removals
        """

        return self._removed_filter[plate].any()

    def has_any_colonies_removed(self):
        """If any plate has anything removed

        Returns:
            bool    The removal status
        """
        return any(self.plate_has_any_colonies_removed(i) for i in
                   range(self._removed_filter.shape[0]))

    def get_position_list_filtered(self, position_list, value_type=Phenotypes.GenerationTime):

        values = []
        for pos in position_list:

            if isinstance(pos, StringTypes):
                plate, x, y = self._position_2_string_tuple(pos)
            else:
                plate, x, y = pos

            values.append(self.phenotypes[plate][x, y][value_type.value])

        return values

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
            meta_data = self._meta_data
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

                if plate is None:
                    continue

                for idX, X in enumerate(plate):

                    for idY, Y in enumerate(X):

                        # TODO: This is a hack to not break csv structure with vector phenotypes
                        Y = [v if not(isinstance(v, np.ndarray) and v.size > 1
                                      or isinstance(v, list)
                                      or isinstance(v, tuple)) else None for v in Y]

                        if meta_data is None:
                            fh.write("{0}{1}".format(delim.join(map(
                                str, [plate_index, idX, idY] + Y)), newline))
                        else:
                            fh.write("{0}{1}".format(delim.join(map(
                                str, [plate_index, idX, idY] + meta_data(plate_index, idX, idY) +
                                     Y)), newline))

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
            np.save(p, self._removed_filter)

        p = os.path.join(dir_path, self._paths.phenotype_times)
        if (not ask_if_overwrite or not os.path.isfile(p) or
                self._do_ask_overwrite(p)):
            np.save(p, self._times_data)

        p = os.path.join(dir_path, self._paths.phenotypes_extraction_params)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(
                p,
                [self._median_kernel_size,
                 self._gaussian_filter_sigma,
                 self._linear_regression_size])

        self._logger.info("State saved to '{0}'".format(dir_path))

    def save_input_data(self, path=None):

        if path is None:

            assert self._base_name is not None, "Must give path some way"

            path = self._base_name

        if path.endswith(".npy"):
            path = path[:-4]

        source = self._raw_growth_data
        if isinstance(source, xml_reader_module.XML_Reader):
            source = self._xml_reader_2_array(source)

        np.save(path + ".data.npy", source)
        np.save(path + ".times.npy", self._times_data)
