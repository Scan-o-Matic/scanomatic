import numpy as np
import os
import csv
from scipy.ndimage import median_filter, gaussian_filter1d
from collections import deque
import pickle
from enum import Enum
from types import StringTypes

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
from scanomatic.dataProcessing.curve_phase_phenotypes import phase_phenotypes
from scanomatic.generics.phenotype_filter import FilterArray, Filter
from scanomatic.io.meta_data import MetaData
from scanomatic.dataProcessing.strain_selector import StrainSelector
from scanomatic.dataProcessing.norm import Offsets

class SaveData(Enum):

    ScalarPhenotypesRaw = 0
    ScalarPhenotypesNormalized = 1
    VectorPhenotypesRaw = 10
    VectorPhenotypesNormalized = 11


# TODO: Phenotypes should possibly not be indexed based on enum value either and use dict like the undo/filter


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

    UNDO_HISTORY_LENGTH = 50

    def __init__(self, raw_growth_data, times_data=None,
                 median_kernel_size=5, gaussian_filter_sigma=1.5, linear_regression_size=5,
                 phenotypes=None, base_name=None, run_extraction=False,
                 phenotypes_inclusion=PhenotypeDataType.Trusted):

        self._paths = paths.Paths()

        self._raw_growth_data = raw_growth_data
        self._smooth_growth_data = None
        self._phenotypes = None
        self._vector_phenotypes = None
        self._times_data = None
        self._limited_phenotypes = phenotypes
        self._phenotypes_inclusion = phenotypes_inclusion
        self._base_name = base_name

        if isinstance(raw_growth_data, xml_reader_module.XML_Reader):

            if times_data is None:
                times_data = raw_growth_data.get_scan_times()

        self.times = times_data

        assert self._times_data is not None, "A data series needs its times"

        super(Phenotyper, self).__init__(None)

        self._phenotype_filter = np.array([{} for _ in self._raw_growth_data], dtype=np.object)
        self._phenotype_filter_undo = None

        self._logger = logger.Logger("Phenotyper")

        assert median_kernel_size % 2 == 1, "Median kernel size must be odd"
        self._median_kernel_size = median_kernel_size
        self._gaussian_filter_sigma = gaussian_filter_sigma
        self._linear_regression_size = linear_regression_size
        self._meta_data = None

        self._normalizable_phenotypes = set(
            Phenotypes.GenerationTime, Phenotypes.ExperimentGrowthYield, Phenotypes.ExperimentPopulationDoublings,
            Phenotypes.GenerationTimePopulationSize, Phenotypes.GrowthLag, Phenotypes.ColonySize48h)

        self._reference_surface_positions = [Offsets.LowerRight() for _ in self.enumerate_plates]

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
        try:
            vector_phenotypes = np.load(os.path.join(directory_path, _p.vector_phenotypes_raw))
        except IOError:
            vector_phenotypes = None

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
        phenotyper.set('vector_phenotypes', vector_phenotypes)

        filter_path = os.path.join(directory_path, _p.phenotypes_filter)
        if os.path.isfile(filter_path):
            phenotyper.set("phenotype_filter", np.load(filter_path))

        filter_undo_path = os.path.join(directory_path, _p.phenotypes_filter_undo)
        if os.path.isfile(filter_undo_path):
            with open(filter_undo_path, 'r') as fh:
                try:
                    phenotyper.set("phenotype_filter_undo", pickle.load(fh))
                except EOFError:
                    phenotyper._logger.warning("Could not load saved undo, file corrupt!")

        meta_data_path = os.path.join(directory_path, _p.phenotypes_meta_data)
        if os.path.isfile(meta_data_path):
            with open(meta_data_path, 'r') as fh:
                try:
                    phenotyper.set("meta_data", pickle.load(fh))
                except EOFError:
                    phenotyper._logger.warning("Could not load saved meta-data, file corrupt!")

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

        return cls(np.load(data_directory), np.load(times_data_path), base_name=path, run_extraction=True, **kwargs)

    @property
    def meta_data(self):

        return self._meta_data

    @property
    def raw_growth_data(self):

        return self._raw_growth_data

    @property
    def smooth_growth_data(self):

        return self._smooth_growth_data

    @property
    def enumerate_plates(self):

        for i, _ in enumerate(self._raw_growth_data):
            yield i

    @property
    def plate_shapes(self):

        for plate in self._raw_growth_data:

            if plate is None:
                yield None
            else:
                yield plate.shape[:2]

    @staticmethod
    def _xml_reader_2_array(data_object):

        return np.array([k in data_object.get_data().keys() and
                         data_object.get_data()[k] or None for k in
                         range(max((data_object.get_data().keys())) + 1)])

    def load_meta_data(self, *meta_data_paths):

        """Loads meta-data about the experiment based on paths to compatible files.

        See the wiki on how such files should be formatted

        :param paths: Any number of paths to files OpenOffice or Excel compatible that contains the meta data

        :return: None
        """

        self._meta_data = MetaData(tuple(self.plate_shapes), *meta_data_paths)

    def find_in_meta_data(self, value, column=None):

        selection = self.meta_data.find(value, column=column)
        return StrainSelector(self, tuple(zip(*s) for s in selection))

    def iterate_extraction(self):

        self._logger.info(
            "Iteration started, will extract {0} phenotypes".format(
                self.number_of_phenotypes))

        if not self.has_smooth_growth_data:
            self._smoothen()
            self._logger.info("Smoothed")
            yield 0
        else:
            self._logger.info("No smoothing, data already smooth!")

        for x in self._calculate_phenotypes():
            self._logger.debug("Phenotype extraction iteration")
            yield x

        self._init_remove_filter_and_undo_actions()

    def wipe_extracted_phenotypes(self):

        if self._phenotypes is not None:
            self._logger.info("Removing previous phenotypes")
        self._phenotypes = None

        if self._vector_phenotypes is not None:
            self._logger.info("Removing previous vector phenotypes")
        self._vector_phenotypes = None

        if self._phenotype_filter is not None:
            self._logger.info("Removing previous remove filter")
        self._phenotype_filter = None

        if self._phenotype_filter_undo is not None:
            self._logger.info("Removing filter undo history")
        self._phenotype_filter_undo = None

    def extract_phenotypes(self):

        self.wipe_extracted_phenotypes()

        self._logger.info("Extracting phenotypes. This will take a while...")

        if not self.has_smooth_growth_data:
            self._smoothen()

        for _ in self._calculate_phenotypes():
            pass

        self._init_remove_filter_and_undo_actions()

        self._logger.info("Phenotypes extracted")

    @property
    def has_smooth_growth_data(self):

        if self._smooth_growth_data is None or len(self._smooth_growth_data) != len(self._raw_growth_data):
            return False

        return all(((a is None) is (b is None)) or a.shape == b.shape for a, b in
                   zip(self._raw_growth_data, self._smooth_growth_data))

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
        all_vector_phenotypes = []

        regression_size = self._linear_regression_size
        position_offset = (regression_size - 1) / 2
        phenotypes_count = self.number_of_phenotypes
        vector_phenotypes_count = 2
        total_curves = float(self.number_of_curves)

        self._logger.info("Phenotypes (N={0}), extraction started for {1} curves".format(
            phenotypes_count, int(total_curves)))

        curves_in_completed_plates = 0
        phenotypes_inclusion = self._phenotypes_inclusion

        if phenotypes_inclusion is not PhenotypeDataType.Trusted:
            self._logger.warning("Will extract phenotypes beyond those that are trusted, this is not recommended!" +
                                 " It is your responsibility to verify the validity of those phenotypes!")

        for id_plate, plate in enumerate(self._smooth_growth_data):

            if plate is None:
                all_phenotypes.append(None)
                all_vector_phenotypes.append(None)
                continue

            plate_flat_regression_strided = self._get_plate_linear_regression_strided(plate)

            phenotypes = np.zeros((plate.shape[:2]) + (phenotypes_count,), dtype=np.float)
            vector_phenotypes = np.zeros((plate.shape[:2] + (vector_phenotypes_count, )), dtype=np.object) * np.nan

            all_phenotypes.append(phenotypes)
            all_vector_phenotypes.append(vector_phenotypes)

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

                phases_reported = False

                for phenotype in Phenotypes:

                    if not phenotypes_inclusion(phenotype):
                        continue

                    if PhenotypeDataType.Scalar(phenotype):
                        position_phenotypes[phenotype.value] = phenotype(**curve_data)

                    elif not phases_reported and PhenotypeDataType.Phases(phenotype):

                        phases, phases_phenotypes, _ = phase_phenotypes(
                            self, id_plate, (id0, id1), f=False,
                            experiment_doublings=position_phenotypes[Phenotypes.ExperimentPopulationDoublings.value])

                        if phenotypes_inclusion(Phenotypes.GrowthPhasesVector):
                            vector_phenotypes[id0, id1, 0] = phases
                        if phenotypes_inclusion(Phenotypes.GrowthPhasesPhenotypes):
                            vector_phenotypes[id0, id1, 1] = phases_phenotypes

                        phases_reported = True

                phenotypes[id0, id1, ...] = position_phenotypes

                if id0 == 0:
                    self._logger.debug("Done plate {0} pos {1} {2} {3}".format(
                        id_plate, id0, id1, list(position_phenotypes)))
                    yield (curves_in_completed_plates + pos_index + 1.0) / total_curves

            self._logger.info("Plate {0} Done".format(id_plate + 1))
            curves_in_completed_plates += 0 if plate is None else plate_flat_regression_strided.shape[0]

        self._phenotypes = np.array(all_phenotypes)
        self._vector_phenotypes = np.array(all_vector_phenotypes)

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

    def add_phenotype_to_normalization(self, phenotype):

        self._normalizable_phenotypes.add(phenotype)

    def remove_phenotype_from_normalization(self, phenotype):

        self._normalizable_phenotypes.remove(phenotype)

    def set_control_surface_offsets(self, offset, plate=None):

        if plate is None:
            self._reference_surface_positions = [offset() for _ in self.enumerate_plates]
        else:
            self._reference_surface_positions[plate] = offset()

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

        return self.get_phenotype(Phenotypes.GenerationTime)

    def get_phenotype(self, phenotype, filtered=True):

        def _plate_type_converter_vector(plate):

            out = plate.copy()
            if out.dtype == np.floating and out.shape[-1] == 1:
                return out.reshape(out.shape[:2])

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

        if filtered:

            return [None if p is None else
                    FilterArray(_plate_type_converter(p[..., phenotype.value]),
                                self._phenotype_filter[id_plate][phenotype])
                    for id_plate, p in enumerate(self._phenotypes)]
        else:
            return [None if p is None else _plate_type_converter(p[..., phenotype.value]) for p in self._phenotypes]

    @property
    def analysed_phenotypes(self):

        for p in Phenotypes:

            if self._phenotypes_inclusion(p) and self._phenotypes is not None\
                    and self._phenotypes[0].shape[-1] > p.value:
                yield p

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
                self.smooth_growth_data[plate][position].reshape(1, 1, self.times.size))[0], self.times_strided)[0]

    def set(self, data_type, data):

        if data_type == 'phenotypes':

            if isinstance(data, np.ndarray) and (data.size == 0 or not data.any()):
                self._phenotypes = None
            else:
                self._phenotypes = data

            self._init_remove_filter_and_undo_actions()

        elif data_type == 'vector_phenotypes':

            self._vector_phenotypes = data

            self._init_remove_filter_and_undo_actions()

        elif data_type == 'smooth_growth_data':

            self._smooth_growth_data = data

        elif data_type == "phenotype_filter_undo":

            if isinstance(data, tuple) and all(isinstance(q, deque) for q in data):
                self._phenotype_filter_undo = data
            else:
                self._logger.warning("Not a proper undo history")

            self._init_remove_filter_and_undo_actions()

        elif data_type == "phenotype_filter":

            if not all(True if plate is None else isinstance(plate, dict) for plate in data):
                self._phenotype_filter = self._convert_to_current_phenotype_filter(data)
            else:
                self._phenotype_filter = data
            self._init_remove_filter_and_undo_actions()

        elif data_type == "meta_data":

            if isinstance(data, MetaData) or data is None:
                self._meta_data = data
            else:
                self._logger.warning("Not a valid meta data type")
        else:

            self._logger.warning('Unknown type of data {0}'.format(data_type))

    def _convert_to_current_phenotype_filter(self, data):

        new_data = []
        for id_plate, plate in enumerate(data):

            if plate is None or plate.size == 0:
                new_data.append(None)
                continue

            if plate.dtype == np.int8 and plate.ndim == 3:
                new_plate = {}

                for id_phenotype in range(plate.shape[-1]):
                    try:
                        new_plate[Phenotypes(id_phenotype)] = plate[..., id_phenotype]
                    except ValueError:
                        self._logger.warning(
                            "Saved data had a Phenotype of index {0}, this is not a valid Phenotype".format(
                                id_phenotype))

                new_data.append(new_plate)

            else:

                self._logger.error(
                    "Skipping previous phenotype filter of plate {0} because not understood".format(id_plate + 1))
                new_data.append(None)

        return np.array(new_data)

    @staticmethod
    def _correct_shapes(guide, obj):

        if guide is None:
            return True
        elif obj is None:
            return False
        elif len(guide) != len(obj):
            return False

        if isinstance(obj, np.ndarray):
            for g, o in zip(guide, obj):
                if (g is None) is not (o is None):
                    return False
                if isinstance(o, dict):
                    for v in o.itervalues():
                        if g.shape[:2] != v.shape[:2]:
                            return False
                elif g.shape[:2] != o.shape[:2]:
                    return False
        return True

    def _init_remove_filter_and_undo_actions(self):

        if not self._correct_shapes(self._phenotypes, self._phenotype_filter):
            self._phenotype_filter = np.array([{} for _ in range(self._phenotypes.shape[0])], dtype=np.object)

            for plate_index in range(self._phenotypes.shape[0]):

                if self._phenotypes[plate_index] is None:
                    continue

                for phenotype in Phenotypes:

                    if self._phenotypes_inclusion(phenotype):

                        self._phenotype_filter[plate_index][phenotype] = np.zeros(
                            self._raw_growth_data[plate_index].shape[:2], dtype=np.int8)

                        self._phenotype_filter[plate_index][phenotype][
                            np.where(np.isfinite(self._phenotypes[plate_index][..., phenotype.value]) == False)] = \
                            Filter.UndecidedProblem.value

        if not self._correct_shapes(self._phenotypes, self._phenotype_filter_undo):
            self._phenotype_filter_undo = tuple(deque() for _ in self._phenotypes)

    def add_position_mark(self, plate, position_list, phenotype=None, position_mark=Filter.BadData,
                          undoable=True):

        if phenotype is None:

            for phen in Phenotypes:
                if self._phenotypes_inclusion(phen):
                    self.add_position_mark(plate, position_list, phen, position_mark, undoable=False)

            if undoable:
                self._logger.warning("Undoing this mark will assume all phenotypes were previously marked as OK")
                self._add_undo(plate, position_list, None, 0)

            return

        else:

            previous_state = self._phenotype_filter[plate][phenotype][position_list]

            if isinstance(previous_state, np.ndarray):
                if np.unique(previous_state).size == 1:
                    previous_state = previous_state[0]

            self._phenotype_filter[plate][phenotype][position_list] = position_mark.value

            if undoable:
                self._add_undo(plate, position_list, phenotype, previous_state)

    def _add_undo(self, plate, position_list, phenotype, previous_state):

        self._phenotype_filter_undo[plate].append((position_list, phenotype, previous_state))
        while len(self._phenotype_filter_undo[plate]) > self.UNDO_HISTORY_LENGTH:
            self._phenotype_filter_undo[plate].popleft()

    def undo(self, plate):

        if len(self._phenotype_filter_undo[plate]) == 0:
            self._logger.info("No more actions to undo")
            return

        position_list, phenotype, previous_state = self._phenotype_filter_undo[plate].pop()
        self._logger.info("Setting {0} for positions {1} to state {2}".format(
            phenotype,
            position_list,
            Filter(previous_state)))

        if phenotype is None:
            for phenotype in Phenotypes:
                if self._phenotypes_inclusion(phenotype) and phenotype in self._phenotype_filter[plate]:
                    self._phenotype_filter[plate][phenotype][position_list] = previous_state
        elif phenotype in self._phenotype_filter[plate]:
            self._phenotype_filter[plate][phenotype][position_list] = previous_state
        else:
            self._logger.warning("Could not undo for {0} because no filter present for phenotype".format(phenotype))

    def plate_has_any_colonies_removed(self, plate):
        """Get if plate has anything removed.

        Args:

            plate (int)   Index of plate

        Returns:

            bool    The status of the plate removals
        """

        return self._phenotype_filter[plate].any()

    def has_any_colonies_removed(self):
        """If any plate has anything removed

        Returns:
            bool    The removal status
        """
        return any(self.plate_has_any_colonies_removed(i) for i in
                   range(self._phenotype_filter.shape[0]))

    @staticmethod
    def _make_csv_row(*args):

        for a in args:

            if isinstance(a, StringTypes):
                yield a
            else:
                try:
                    for v in Phenotyper._make_csv_row(*a):
                        yield v
                except TypeError:
                    yield a

    def meta_data_headers(self, plate_index):

        if self._meta_data is not None:
            self._logger.info("Adding meta-data")
            return self._meta_data.get_header_row(plate_index)

        return tuple()

    def save_phenotypes(self, path=None, save_data=SaveData.ScalarPhenotypesRaw,
                        dialect=csv.excel, ask_if_overwrite=True):

        if path is None and self._base_name is not None:
            path = self._base_name
            if path == ".":
                path = "phenotypes"

        if save_data == SaveData.ScalarPhenotypesRaw:
            data_source = self._phenotypes
        else:
            self._logger.error("Not implemented saving '{0}'".format(save_data))
            return False

        default_meta_data = ('Plate', 'Row', 'Column')

        meta_data = self._meta_data
        no_metadata = tuple()

        phenotype_filter = np.where([self._phenotypes_inclusion(p) for p in Phenotypes])[0]
        phenotype_filter = phenotype_filter[phenotype_filter < self.number_of_phenotypes]

        for plate_index in self.enumerate_plates:
            plate_path = "{0}.{1}.{2}.csv".format(path, save_data.name.lower(), plate_index)

            if os.path.isfile(plate_path) and ask_if_overwrite:
                if 'y' not in raw_input("Overwrite existing file '{0}'? (y/N)".format(plate_path)).lower():
                    continue

            with open(plate_path, 'w') as fh:

                cw = csv.writer(fh, dialect=dialect)

                # HEADER ROW
                meta_data_headers = self.meta_data_headers(plate_index)

                cw.writerow(
                    tuple(self._make_csv_row(
                        default_meta_data,
                        meta_data_headers,
                        (p for p in Phenotypes if self._phenotypes_inclusion(p)))))

                # DATA
                plate = data_source[plate_index]
                filt = self._phenotype_filter[plate_index]

                if plate is None:
                    continue

                for idX, X in enumerate(plate):

                    for idY, Y in enumerate(X):

                        cw.writerow(
                            tuple(self._make_csv_row(
                                (plate_index, idX, idY),
                                no_metadata if meta_data is None else meta_data(plate_index, idX, idY),
                                (y if filt[Phenotypes(idP)][idX, idY] == 0 else
                                 Filter(filt[Phenotypes(idP)][idX, idY]).name
                                 for idP, y in zip(phenotype_filter, Y[phenotype_filter])))))

                self._logger.info("Saved {0}, plate {1} to {2}".format(save_data, plate_index + 1, plate_path))

        return True

    @staticmethod
    def _do_ask_overwrite(path):
        return raw_input("Overwrite '{0}' (y/N)".format(
            path)).strip().upper().startswith("Y")

    def save_state(self, dir_path, ask_if_overwrite=True):

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        p = os.path.join(dir_path, self._paths.phenotypes_raw_npy)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(p, self._phenotypes)

        p = os.path.join(dir_path, self._paths.vector_phenotypes_raw)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(p, self._vector_phenotypes)

        p = os.path.join(dir_path, self._paths.phenotypes_input_data)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(p, self._raw_growth_data)

        p = os.path.join(dir_path, self._paths.phenotypes_input_smooth)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(p, self._smooth_growth_data)

        p = os.path.join(dir_path, self._paths.phenotypes_filter)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(p, self._phenotype_filter)

        p = os.path.join(dir_path, self._paths.phenotypes_filter_undo)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):

            with open(p, 'w') as fh:
                pickle.dump(self._phenotype_filter_undo, fh)

        p = os.path.join(dir_path, self._paths.phenotype_times)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(p, self._times_data)

        p = os.path.join(dir_path, self._paths.phenotypes_meta_data)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            with open(p, 'w') as fh:
                pickle.dump(self._meta_data, fh)

        p = os.path.join(dir_path, self._paths.phenotypes_extraction_params)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(
                p,
                [self._median_kernel_size,
                 self._gaussian_filter_sigma,
                 self._linear_regression_size])

        self._logger.info("State saved to '{0}'".format(dir_path))

