import csv
import glob
import os

from collections import deque
from itertools import izip, product, chain
from types import StringTypes
import cPickle as pickle
import numpy as np
from enum import Enum
from scipy.ndimage import median_filter
from scipy.stats import norm
from scanomatic.io.pickler import unpickle, unpickle_with_unpickler

#
#   INTERNAL DEPENDENCIES
#

# TODO: Something is wrong with phase features again

import mock_numpy_interface
import scanomatic.io.xml.reader as xml_reader_module
import scanomatic.io.logger as logger
import scanomatic.io.paths as paths
import scanomatic.io.image_data as image_data
from scanomatic.data_processing.growth_phenotypes import Phenotypes, get_preprocessed_data_for_phenotypes, \
    get_derivative
from scanomatic.data_processing.phases.features import extract_phenotypes, \
    CurvePhaseMetaPhenotypes, VectorPhenotypes
from scanomatic.data_processing.phases.analysis import get_phase_analysis
from scanomatic.data_processing.phenotypes import PhenotypeDataType, infer_phenotype_from_name
from scanomatic.generics.phenotype_filter import FilterArray, Filter
from scanomatic.io.meta_data import MetaData2 as MetaData
from scanomatic.data_processing.strain_selector import StrainSelector
from scanomatic.data_processing.norm import Offsets, get_normalized_data, get_reference_positions


def time_based_gaussian_weighted_mean(data, time, sigma=1):
    center = (time.size - time.size % 2) / 2
    delta_time = np.abs(time - time[center])
    kernel = norm.pdf(delta_time, loc=0, scale=sigma)
    finite = np.isfinite(data)
    if not finite.any() or not finite[center]:
        return np.nan
    kernel /= kernel[finite].sum()
    return (data[finite] * kernel[finite]).sum()


class EdgeCondition(Enum):
    Reflect = 0
    """:type : EdgeCondition"""
    Symmetric = 1
    """:type : EdgeCondition"""
    Nearest = 2
    """:type : EdgeCondition"""
    Valid = 3
    """:type : EdgeCondition"""


def edge_condition(arr, mode=EdgeCondition.Reflect, kernel_size=3):

    if not kernel_size % 2 == 1:
        raise ValueError("Only odd-size kernels supported")

    origin = (kernel_size - 1) / 2
    idx = 0

    # First edge:
    if mode is EdgeCondition.Symmetric:
        while idx < origin:
            yield np.hstack((arr[:origin - idx][::-1], arr[:idx + 1 + origin]))
            idx += 1

    elif mode is EdgeCondition.Nearest:
        while idx < origin:
            yield np.hstack((tuple(arr[0] for _ in xrange(origin - idx)), arr[:idx + 1 + origin]))
            idx += 1

    elif mode is EdgeCondition.Reflect:
        while idx < origin:
            yield np.hstack((arr[1: origin - idx + 1][::-1], arr[:idx + 1 + origin]))
            idx += 1
    elif mode is EdgeCondition.Valid:
        pass

    # Valid range
    while arr.size - idx > origin:
        yield arr[idx - origin: idx + origin + 1]
        idx += 1

    # Second edge
    if mode is EdgeCondition.Symmetric:
        while idx < arr.size:
            yield np.hstack((arr[idx - origin:], arr[arr.size - idx - origin - 1:][::-1]))
            idx += 1

    elif mode is EdgeCondition.Nearest:
        while idx < arr.size:
            yield np.hstack((arr[idx - origin:], tuple(arr[-1] for _ in xrange(-1*(arr.size - idx - origin - 1)))))
            idx += 1

    elif mode is EdgeCondition.Reflect:
        while idx < arr.size:
            yield np.hstack((arr[idx - origin:], arr[arr.size - idx - origin - 2: -1][::-1]))
            idx += 1
    elif mode is EdgeCondition.Valid:
        pass


def merge_convolve(arr1, arr2, edge_condition_mode=EdgeCondition.Reflect, kernel_size=5,
                   func=time_based_gaussian_weighted_mean, func_kwargs=None):

    if not func_kwargs:
        func_kwargs = {}

    return tuple(func(v1, v2, **func_kwargs) for v1, v2 in izip(
        edge_condition(arr1, mode=edge_condition_mode, kernel_size=kernel_size),
        edge_condition(arr2, mode=edge_condition_mode, kernel_size=kernel_size)))


def get_phenotype(name):

    try:
        return Phenotypes[name]
    except KeyError:
        pass

    try:
        return CurvePhaseMetaPhenotypes[name]
    except KeyError:
        pass

    try:
        return VectorPhenotypes[name]
    except KeyError:
        pass

    raise KeyError("Unknown phenotype {0}".format(name))


def path_has_saved_project_state(directory_path, require_phenotypes=True):

    if not directory_path:
        return False

    _p = paths.Paths()

    if require_phenotypes:
        try:
            unpickle_with_unpickler(np.load, os.path.join(directory_path, _p.phenotypes_raw_npy))
        except IOError:
            return False

    try:
        unpickle_with_unpickler(np.load, os.path.join(directory_path,  _p.phenotypes_input_data))
        unpickle_with_unpickler(np.load, os.path.join(directory_path, _p.phenotype_times))
        unpickle_with_unpickler(np.load, os.path.join(directory_path, _p.phenotypes_input_smooth))
        unpickle_with_unpickler(np.load, os.path.join(directory_path, _p.phenotypes_extraction_params))
    except IOError:
        return False

    return True


def get_project_dates(directory_path):

    def most_recent(stat_result):

        return max(stat_result.st_mtime, stat_result.st_atime, stat_result.st_ctime)

    analysis_date = None
    _p = paths.Paths()
    image_data_files = glob.glob(os.path.join(directory_path, _p.image_analysis_img_data.format("*")))
    if image_data_files:
        analysis_date = max(most_recent(os.stat(p)) for p in image_data_files)
    try:
        phenotype_date = most_recent(os.stat(os.path.join(directory_path, _p.phenotypes_raw_npy)))
    except OSError:
        phenotype_date = None

    state_date = phenotype_date

    for path in (_p.phenotypes_input_data, _p.phenotype_times, _p.phenotypes_input_smooth,
                 _p.phenotypes_extraction_params, _p.phenotypes_filter, _p.phenotypes_filter_undo,
                 _p.phenotypes_meta_data, _p.normalized_phenotypes, _p.vector_phenotypes_raw,
                 _p.vector_meta_phenotypes_raw, _p.phenotypes_reference_offsets):

        try:
            state_date = max(state_date, most_recent(os.stat(os.path.join(directory_path, path))))
        except OSError:
            pass

    return analysis_date, phenotype_date, state_date


class Smoothing(Enum):
    Keep = 0
    """:type : Smoothing"""
    MedianGauss = 1
    """:type : Smoothing"""
    Polynomial = 2
    """:type : Smoothing"""
    PolynomialWeightedMulti = 3
    """:type : Smoothing"""


class SaveData(Enum):
    """Types of data that can be exported to csv.

    Attributes:
        SaveData.ScalarPhenotypesRaw: The non-normalized scalar-value phenotypes.
        SaveData.ScalarPhenotypesNormalized: The normalized scalar-value phenotypes.
        SaveData.VectorPhenotypesRaw: The non-normalized phenotype vectors.
        SaveData.VectorPhenotypesNormalized: The normalized phenotype vectors.

    See Also:
        scanomatic.data_processing.phenotypes.PhenotypeDataType: Classification of phenotypes.
        Phenotyper.save_phenotypes: Exporting phenotypes to csv.
    """
    ScalarPhenotypesRaw = 0
    """:type : SaveData"""
    ScalarPhenotypesNormalized = 1
    """:type : SaveData"""
    VectorPhenotypesRaw = 10
    """:type : SaveData"""
    VectorPhenotypesNormalized = 11
    """:type : SaveData"""


class NormState(Enum):
    """Spatial bias normalisation state in data

    Attributes:
        NormState.Absolute:
            Data is as produced by phenotype extraction.
            This includes spatial 2D bias
        NormState.NormalizedRelative:
            Data is log_2 normalized relative values or
            strain coefficients. This relates to the LSC values
            in Warringer (2003) but there are slight differences.
        NormState.NormalizedAbsoluteBatched:
            This is a plate-wise recalculation of absolute values
            based on the `NormState.NormalizedRelative` values and
            the median `NormState.Absolute` value of the reference
            positions.
            *Note that this reintroduces plate-wise batch effects,
            so it is only recommended if all experiments for a given
            environment were done on a single plate*
        NormState.NormalizedAbsoluteNonBatched:
            This is a global recalculation of absolute values
            based on the `NormState.NormalizedRelative` values and
            a supplied mean of all comparable plate-median reference position
            `NormState.Absolute` values.
    """
    Absolute = 0
    """:type : NormState"""
    NormalizedRelative = 1
    """:type : NormState"""
    NormalizedAbsoluteBatched = 2
    """:type : NormState"""
    NormalizedAbsoluteNonBatched = 3
    """:type : NormState"""


# TODO: Phenotypes should possibly not be indexed based on enum value either and use dict like the undo/filter


class Phenotyper(mock_numpy_interface.NumpyArrayInterface):
    """The Phenotyper class is a class for producing phenotypes
    based on growth curves as well as storing and easy displaying them.

    There are foure modes of instantiating a phenotyper instance:

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
                 median_kernel_size=5,
                 gaussian_filter_sigma=1.5,
                 linear_regression_size=5,
                 no_growth_monotonocity_threshold=0.6,
                 no_growth_pop_doublings_threshold=1.0,
                 base_name=None, run_extraction=False, phenotypes=None,
                 phenotypes_inclusion=PhenotypeDataType.Trusted):

        self._paths = paths.Paths()

        self._raw_growth_data = raw_growth_data
        self._smooth_growth_data = None

        self._phenotypes = phenotypes
        self._vector_phenotypes = None
        self._vector_meta_phenotypes = None
        self._normalized_phenotypes = None

        self._times_data = None

        self._phenotypes_inclusion = phenotypes_inclusion
        self._base_name = base_name

        if isinstance(raw_growth_data, xml_reader_module.XML_Reader):

            if times_data is None:
                times_data = raw_growth_data.get_scan_times()

        self.times = times_data

        assert self._times_data is not None, "A data series needs its times"

        super(Phenotyper, self).__init__(self._raw_growth_data)

        self._phenotype_filter = None
        self._phenotype_filter_undo = None

        self._logger = logger.Logger("Phenotyper")

        assert median_kernel_size % 2 == 1, "Median kernel size must be odd"
        self._median_kernel_size = median_kernel_size
        self._gaussian_filter_sigma = gaussian_filter_sigma
        self._linear_regression_size = linear_regression_size
        self._no_growth_monotonicity_threshold = no_growth_monotonocity_threshold
        self._no_growth_pop_doublings_threshold = no_growth_pop_doublings_threshold

        self._meta_data = None

        self._normalizable_phenotypes = {
            Phenotypes.GenerationTime,
            Phenotypes.GenerationTimePopulationSize,
            Phenotypes.ExperimentGrowthYield,
            Phenotypes.ExperimentPopulationDoublings,
            Phenotypes.GenerationTimePopulationSize,
            Phenotypes.GrowthLag,
            Phenotypes.ColonySize48h,
            Phenotypes.ResidualGrowth,
            Phenotypes.ResidualGrowthAsPopulationDoublings,
            Phenotypes.ExperimentLowPoint,
            Phenotypes.ExperimentBaseLine,

            CurvePhaseMetaPhenotypes.InitialLag,
            CurvePhaseMetaPhenotypes.InitialLagAlternativeModel,
            CurvePhaseMetaPhenotypes.MajorImpulseAveragePopulationDoublingTime,
            CurvePhaseMetaPhenotypes.MajorImpulseYieldContribution,
            CurvePhaseMetaPhenotypes.MajorImpulseFlankAsymmetry,
            CurvePhaseMetaPhenotypes.InitialAccelerationAsymptoteAngle,
            CurvePhaseMetaPhenotypes.InitialAccelerationAsymptoteIntersect,
            CurvePhaseMetaPhenotypes.FinalRetardationAsymptoteAngle,
            CurvePhaseMetaPhenotypes.FinalRetardationAsymptoteIntersect,
            CurvePhaseMetaPhenotypes.TimeBeforeMajorGrowth,
        }

        self._reference_surface_positions = [Offsets.LowerRight() for _ in self.enumerate_plates]

        if run_extraction:
            self.extract_phenotypes()

    def __contains__(self, phenotype):
        """

        :param phenotype: The phenotype
         :type phenotype: [enum.Enum OR str]
        :return: bool
        """
        if isinstance(phenotype, StringTypes):
            try:
                phenotype = infer_phenotype_from_name(phenotype)
            except ValueError:
                return False

        if isinstance(phenotype, Phenotypes) and self._phenotypes is not None:
            return self._phenotypes is not None and phenotype.value < self._phenotypes.shape[-1]
        elif isinstance(phenotype, CurvePhaseMetaPhenotypes) and self._vector_meta_phenotypes is not None:
            return any(phenotype in plate for plate in self._vector_meta_phenotypes if plate is not None)
        return False

    def set_phenotype_inclusion_level(self, level):
        """Change which phenotypes to be included in feature extraction.

        Default is `scanomatic.data_processing.phenotypes.PhenotypeDataType.Trusted` which indicates that
        the phenotypes are unlikely to be modified in the future and that the algorithms have been thoroughly
        vetted and are expected to not change.

        If the `Phenotyper` instance has been saved (`Phenotyper.save_state()`) after a change to the
        inclusion level has been made, next time the same feature extraction is loaded  by
        `Phenotyper.LoadFromState` that inclusion level is loaded instead of `PhenotypeDataType.Trusted`.

        Args:
            level: The PhenotypeDataType-level to toggle to.
                Options are (`PhenotypeDataType.Trusted`, `PhenotypeDataType.UnderDevelopment` and
                `PhenotypeDataType.Other`.
                :type level: scanomatic.data_processing.phenotypes.PhenotypeDataType

        Notes:
            Using `PhenotypeDataType.UnderDevelopment` or `PhenotypeDataType.Other` is not recommended
            outside the scope of testing stuff. They are both prone to change and haven't been vetted for
            bugs and errors. Especially `Other` can include discarded ideas and sketches. If however you
            decide on using one of them, you should discuss this with Martin before-hand and you yourself
            need to ensure that you trust both the data and the algorithms that produces the data.

        See Also:
            Phenotyper.extract_phenotypes: Run new feature extraction
            Phenotyper.add_phenotype_to_normalization: Including phenotype to what is normalized
            Phenotyper.remove_phenotype_from_normalization: Remove phenotype so that it is not normalized

        """
        if isinstance(level, PhenotypeDataType):
            self._phenotypes_inclusion = level
        else:
            self._logger.error("Value not a PhenotypeDataType!")

    def phenotype_names(self, normed=False):

        if normed:
            if self._normalized_phenotypes is None:
                return []
            else:
                return set(pheno.name for pheno in
                           chain(*(plate.keys() for plate in self._normalized_phenotypes
                                   if plate is not None)))
        else:
            return tuple(p.name for p in self.phenotypes if p in self)

    @classmethod
    def LoadFromXML(cls, path, **kwargs):
        """Class Method used to create a Phenotype Strider directly
        from a path do an xml

        Parameters:

            path:
                The path to the xml-file
            kwargs:
                Optional parameters passed on to the constructor

        Optional Parameters can be passed as keywords and will be
        used in instantiating the class.
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

            directory_path (str):
                Path to the directory holding the relevant files

        Returns:

            Phenotyper instance
        """
        _p = paths.Paths()

        raw_growth_data = unpickle_with_unpickler(np.load, os.path.join(directory_path,  _p.phenotypes_input_data))

        times = unpickle_with_unpickler(np.load, os.path.join(directory_path, _p.phenotype_times))

        phenotyper = cls(raw_growth_data, times, run_extraction=False, base_name=directory_path)

        try:
            phenotypes = unpickle_with_unpickler(np.load, os.path.join(directory_path, _p.phenotypes_raw_npy))
        except (IOError, ValueError):
            phenotyper._logger.warning(
                "Could not load Phenotypes, probably too old extraction, please rerun!")
            phenotypes = None

        try:
            vector_phenotypes = unpickle_with_unpickler(
                np.load, os.path.join(directory_path, _p.vector_phenotypes_raw))
        except (IOError, ValueError):
            phenotyper._logger.warning(
                "Could not load Vector Phenotypes, probably too old extraction, please rerun!")
            vector_phenotypes = None

        try:
            vector_meta_phenotypes = unpickle_with_unpickler(
                np.load, os.path.join(directory_path, _p.vector_meta_phenotypes_raw))
        except (IOError, ValueError):
            phenotyper._logger.warning(
                "Could not load Vector Meta Phenotypes, probably too old extraction, please rerun!")
            vector_meta_phenotypes = None

        smooth_growth_data = unpickle_with_unpickler(
            np.load, os.path.join(directory_path, _p.phenotypes_input_smooth))

        try:
            extraction_params = unpickle_with_unpickler(
                np.load, os.path.join(directory_path, _p.phenotypes_extraction_params))
        except IOError:
            phenotyper._logger.warning(
                "Could not find stored extraction parameters, assuming defaults were used")
        else:
            if extraction_params.size > 0:
                if extraction_params.size == 3:
                    median_filt_size, gauss_sigma, linear_reg_size = extraction_params
                elif extraction_params.size == 4:
                    median_filt_size, gauss_sigma, linear_reg_size, inclusion_name = extraction_params
                    if inclusion_name is None:
                        inclusion_name = 'Trusted'
                    phenotyper.set_phenotype_inclusion_level(PhenotypeDataType[inclusion_name])
                elif extraction_params.size == 6:

                    median_filt_size,\
                        gauss_sigma, \
                        linear_reg_size, \
                        inclusion_name, \
                        no_growth_monotonicity_threshold, \
                        no_growth_pop_doublings_threshold = extraction_params

                    if inclusion_name is None:
                        inclusion_name = 'Trusted'

                    phenotyper._no_growth_monotonicity_threshold = float(no_growth_monotonicity_threshold)
                    phenotyper._no_growth_pop_doublings_threshold = float(no_growth_pop_doublings_threshold)

                    phenotyper.set_phenotype_inclusion_level(PhenotypeDataType[inclusion_name])

                else:
                    raise ValueError("Stored parameters in {0} can't be understood".format(
                        os.path.join(directory_path, _p.phenotypes_extraction_params)))

                phenotyper._median_kernel_size = int(median_filt_size)
                phenotyper._gaussian_filter_sigma = float(gauss_sigma)
                phenotyper._linear_regression_size = int(linear_reg_size)

        phenotyper.set('smooth_growth_data', smooth_growth_data)
        phenotyper.set('phenotypes', phenotypes)
        phenotyper.set('vector_phenotypes', vector_phenotypes)
        phenotyper.set('vector_meta_phenotypes', vector_meta_phenotypes)

        filter_path = os.path.join(directory_path, _p.phenotypes_filter)
        if os.path.isfile(filter_path):
            phenotyper._logger.info("Loading previous filter {0}".format(filter_path))
            try:
                phenotyper.set("phenotype_filter", unpickle_with_unpickler(np.load, filter_path))
            except (ValueError, IOError):
                phenotyper._logger.warning(
                    "Could not load QC Filter, probably too old extraction, please rerun!")

        offsets_path = os.path.join(directory_path, _p.phenotypes_reference_offsets)
        if os.path.isfile(offsets_path):
            phenotyper.set("reference_offsets", unpickle_with_unpickler(np.load, offsets_path))

        normalized_phenotypes = os.path.join(directory_path, _p.normalized_phenotypes)
        if os.path.isfile(normalized_phenotypes):
            try:
                phenotyper.set("normalized_phenotypes", unpickle_with_unpickler(np.load, normalized_phenotypes))
            except (ValueError, IOError):
                phenotyper._logger.warning(
                    "Could not load Normalized Phenotypes, probably too old extraction, please rerun!")

        filter_undo_path = os.path.join(directory_path, _p.phenotypes_filter_undo)
        if os.path.isfile(filter_undo_path):
            try:
                phenotyper.set("phenotype_filter_undo", unpickle(filter_undo_path))
            except EOFError:
                phenotyper._logger.warning("Could not load saved undo, file corrupt!")

        meta_data_path = os.path.join(directory_path, _p.phenotypes_meta_data)
        if os.path.isfile(meta_data_path):
            try:
                phenotyper.set("meta_data", unpickle(meta_data_path))
            except EOFError:
                phenotyper._logger.warning("Could not load saved meta-data, file corrupt!")

        return phenotyper

    @classmethod
    def LoadFromImageData(cls, path='.', phenotype_inclusion=None):
        """Loads image data files and performs an extraction

        This is what you use if you have only run an analysis or only
        want your `Phenotyper`-object to be free of previous feature
        extraction.

        Args:
            path: optional, default is current directory
            phenotype_inclusion: optional setting for inclusion level
                during phenotype extraction.

        Returns: Phenotyper

        """
        times, data = image_data.ImageData.read_image_data_and_time(path)
        instance = cls(data, times)
        if phenotype_inclusion is not None:
            instance.set_phenotype_inclusion_level(phenotype_inclusion)
        instance.extract_phenotypes()
        return instance

    @classmethod
    def LoadFromNumPy(cls, path, times_data_path=None, **kwargs):
        """Class Method used to create a Phenotype Strider from
        a saved numpy data array and a saved numpy times array.

        Parameters:

            path:
                The path to the data numpy file

            times_data_path:
                The path to the times numpy file
                If not supplied both paths are assumed
                to be named as:

                    some/path.data.npy
                    some/path.times.npy

                And path parameter is expexted to be
                'some/path' in this examply.

        Optional Parameters can be passed as keywords and will be
        used in instantiating the class.
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

        return cls(unpickle_with_unpickler(np.load, data_directory),
                   unpickle_with_unpickler(np.load, times_data_path),
                   base_name=path, run_extraction=True, **kwargs)

    @staticmethod
    def is_segmentation_based_phenotype(phenotype):

        return isinstance(phenotype, CurvePhaseMetaPhenotypes)

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
    def curve_segments(self):

        try:
            return [self._vector_phenotypes[plate][VectorPhenotypes.PhasesClassifications] for
                    plate in self.enumerate_plates]

        except (ValueError, IndexError, TypeError, KeyError):
            return None

    @property
    def enumerate_plates(self):

        for i, _ in enumerate(self._raw_growth_data):
            yield i

    def enumerate_plate_positions(self, plate):

        shape = self._raw_growth_data[plate].shape[:-1]
        for pos_tup in izip(*np.unravel_index(np.arange(np.prod(shape)), shape)):
            yield pos_tup

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

        Args:
            meta_data_paths:
                Any number of paths to files OpenOffice or Excel
                compatible that contains the meta data

        """
        m = MetaData(tuple(self.plate_shapes), *(os.path.expanduser(p) for p in meta_data_paths))
        if m.loaded:
            self._meta_data = m
            return True
        else:
            self._logger.warning(
                "New meta-data incompatible, keeping previous meta-data (if any existed).")
            return False

    def find_in_meta_data(self, query, column=None, plates=None):
        """Look for results for specific strains.

        Args:
            query: What to look for
            column: Optional, exact name of column to look in. If omitted, will search in all columns
            plates: Optional, what plates to include results from, should be a list of plate numbers,
                starting with index 0 for the first plate

        Returns: A StrainSelector object with the search results.

        """
        selection = self.meta_data.find(query, column=column)
        return StrainSelector(self, tuple((zip(*s) if plates is None or i in plates else tuple())
                                          for i, s in enumerate(selection)))

    def iterate_extraction(self):

        self._logger.info(
            "Iteration started, will extract {0} phenotypes".format(
                self.get_number_of_phenotypes()))

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

    def wipe_extracted_phenotypes(self, keep_filter=False):
        """ This clears all extracted phenotypes but keeps the log2_curve data

        Args:
            keep_filter: Optional, if the markings of curves should be kept, default is to
            not keep them [`True`, `False`]

        """
        if self._phenotypes is not None:
            self._logger.info("Removing previous phenotypes")
        self._phenotypes = None

        if self._vector_phenotypes is not None:
            self._logger.info("Removing previous vector phenotypes")
        self._vector_phenotypes = None

        if self._vector_meta_phenotypes is not None:
            self._logger.info("Removing previous vector meta phenotypes")
        self._vector_meta_phenotypes = None

        if keep_filter:
            self._logger.warning("Keeping the filter may cause inconsistencies with what curves are marked as bad."
                                 " Use with care, and consider running the `infer_filter` method.")
        if not keep_filter:
            if self._phenotype_filter is not None:
                self._logger.info("Removing previous remove filter")
            self._phenotype_filter = None

            if self._phenotype_filter_undo is not None:
                self._logger.info("Removing filter undo history")
            self._phenotype_filter_undo = None

    def extract_phenotypes(self, keep_filter=False, smoothing=Smoothing.Keep, smoothing_coeffs={}):
        """Extract phenotypes given the current inclusion level

        Args:
            keep_filter:
                Optional, if previous log2_curve marks on phenotypes should
                be kept or not. Default is to clear previous log2_curve
                marks
            smoothing:
                Optional, if smoothing should be redone.
                Default is not to redo it/keep smoothing.
                Alternatives are `Smoothing.MedianGauss` (as published in
                original Scan-o-matic manuscript.
                Or `Smoothing.Polynomial` for polynomial smoothing, or
                `Smoothing.PolynomialWeightedMulti` if fancy weighting of
                all relevant polynomials should be used.
            smoothing_coeffs:
                Optional dict of key-value parameters for the smoothing
                to override default values.

        See Also:
            Phenotyper.set_phenotype_inclusion_level:
                How to change what phenotypes are extracted
            Phenotyper.get_phenotype:
                Accessing extracted phenotypes.
            Phenotyper.normalize_phenotypes:
                Normalize phenotypes.
        """
        self.wipe_extracted_phenotypes(keep_filter)

        self._logger.info("Extracting phenotypes. This will take a while...")

        if self.has_smooth_growth_data is None and smoothing is Smoothing.Keep:
            self._smoothen()
        elif smoothing is Smoothing.MedianGauss:
            self._smoothen()
        elif smoothing is Smoothing.Polynomial:
            self._poly_smoothen_raw_growth(**smoothing_coeffs)
        elif smoothing is Smoothing.PolynomialWeightedMulti:
            self._poly_smoothen_raw_growth_weighted(**smoothing_coeffs)

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

    @property
    def has_normalized_data(self):

        return self._normalizable_phenotypes is not None and \
               isinstance(self._normalized_phenotypes, np.ndarray) and \
               not all(plate is None or plate.size == 0 for plate in self._normalized_phenotypes) and \
               self._normalized_phenotypes.size > 0

    def _poly_smoothen_raw_growth(self, power=3, time_delta=5):

        assert power > 1, "Power must be 2 or greater"

        self._logger.info("Starting Polynomial smoothing")

        smooth_data = []
        times = self.times
        time_diffs = np.subtract.outer(times, times)
        filt = (time_diffs < time_delta) & (time_diffs > -time_delta)

        for id_plate, plate in enumerate(self._raw_growth_data):
            if plate is None:
                smooth_data.append(None)
                self._logger.info("Plate {0} has no data".format(id_plate + 1))
                continue

            log2_data = np.log2(plate).reshape(np.prod(plate.shape[:2]), plate.shape[-1])
            smooth_plate = np.array(tuple(
                tuple(self._poly_smoothen_raw_growth_curve(times, log2_curve, power, filt))
                for log2_curve in log2_data))

            self._logger.info("Plate {0} data polynomial smoothed".format(id_plate + 1))

            smooth_data.append(smooth_plate.reshape(plate.shape))

        self._smooth_growth_data = np.array(smooth_data)

        self._logger.info("Completed Polynomial smoothing")

    def _poly_smoothen_raw_growth_weighted(self, power=3, time_delta=5.1, gauss_sigma=1.5, apply_median=False):

        assert power > 1, "Power must be 2 or greater"

        self._logger.info("Starting Weighted Multi-Polynomial smoothing")

        median_kernel = np.ones((1, self._median_kernel_size))
        smooth_data = []
        times = self.times
        time_diffs = np.subtract.outer(times, times)
        filt = (time_diffs < time_delta) & (time_diffs > -time_delta)

        for id_plate, plate in enumerate(self._raw_growth_data):
            if plate is None:
                smooth_data.append(None)
                self._logger.info("Plate {0} has no data".format(id_plate + 1))
                continue

            log2_data = np.log2(plate).reshape(np.prod(plate.shape[:2]), plate.shape[-1])

            if apply_median:
                log2_data[...] = median_filter(log2_data, footprint=median_kernel, mode='reflect')

            smooth_plate = [None] * log2_data.shape[0]
            for id_curve, log2_curve in enumerate(log2_data):

                p, r, r0 = zip(*self._poly_estimate_raw_growth_curve(times, log2_curve, power, filt))

                smooth_plate[id_curve] = tuple(self._multi_poly_smooth(
                    times, p, np.array(r), np.array(r0), filt, gauss_sigma))

            self._logger.info("Plate {0} data polynomial smoothed".format(id_plate + 1))

            smooth_data.append(np.array(smooth_plate).reshape(plate.shape))

        self._smooth_growth_data = np.array(smooth_data)

        self._logger.info("Completed Weighted Multi-Polynomial smoothing")

    def _multi_poly_smooth(self, times, polys, r, r0, filt, gauss_sigma):

        included = [v is not None for v in r]
        for t, f in izip(times, filt):

            f2 = f & included

            w1 = norm.pdf(times[f2], loc=t, scale=gauss_sigma)
            w2 = 1 - r[f2] / r0[f2]
            w = w1 * w2
            yield (w * tuple(np.power(2, p(t)) for p, i in izip(polys, f2) if i)).sum() / w.sum()

    def _poly_estimate_raw_growth_curve(self, times, log2_data, power, filt):

        finites = np.isfinite(log2_data)

        for t, f in izip(times, filt):

            f2 = f & finites
            x = times[f2]
            y = log2_data[f2]
            for pwr in range(power, -1, -1):
                try:
                    p, r, _, _, _ = np.polyfit(x, y, power, full=True)
                except TypeError:
                    yield None, None, None
                if r.size > 1:
                    break

            try:
                yield np.poly1d(p), r[0] / f2.sum(), np.var(y)
            except IndexError:
                # This is invoked if only two measurements have finite data for the
                # entire interval and the values of these are identical.
                # Heuristically the residuals are set so we have absolute confidence
                # in this result
                yield np.poly1d(p),  0, 1

    def _poly_smoothen_raw_growth_curve(self, times, log2_data, power, filt):

        finites = np.isfinite(log2_data)

        for t, f in izip(times, filt):

            x = times[f]
            y = log2_data[f]
            fin = finites[f]

            try:
                p = np.polyfit(x[fin], y[fin], power)
            except TypeError:
                yield np.nan
            else:
                yield np.power(2, np.poly1d(p)(t))

    def _smoothen(self):

        self.set("smooth_growth_data", self._raw_growth_data.copy())
        self._logger.info("Smoothing Started")
        median_kernel = np.ones((1, self._median_kernel_size))
        times = self.times

        # This conversion is done to reflect that previous filter worked on
        # indices and expected ratio to hours is 1:3.
        gauss_kwargs = {
            'sigma':
                self._gaussian_filter_sigma / 3.0 if self._gaussian_filter_sigma == 5 else self._gaussian_filter_sigma}

        for plate_id, plate in enumerate(self._smooth_growth_data):

            if plate is None:
                self._logger.info("Plate {0} has no data, skipping".format(plate_id + 1))
                continue

            plate_as_flat = np.lib.stride_tricks.as_strided(
                plate,
                shape=(plate.shape[0] * plate.shape[1], plate.shape[2]),
                strides=(plate.strides[1], plate.strides[2]))

            plate_as_flat[...] = median_filter(
                plate_as_flat, footprint=median_kernel, mode='reflect')

            plate_as_flat[...] = tuple(
                merge_convolve(v, times, func_kwargs=gauss_kwargs) for v in plate_as_flat
            )

            self._logger.info("Smoothing of plate {0} done".format(plate_id + 1))

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
        all_vector_meta_phenotypes = []

        regression_size = self._linear_regression_size
        position_offset = (regression_size - 1) / 2
        phenotypes_count = self.get_number_of_phenotypes()

        total_curves = float(self.number_of_curves)

        self._logger.info("Phenotypes (N={0}), extraction started for {1} curves".format(
            phenotypes_count, int(total_curves)))

        phenotypes_max_value = max(p.value for p in self._phenotypes_inclusion()
                                   if PhenotypeDataType.Scalar(p) and not PhenotypeDataType.Phases(p)) + 1

        curves_in_completed_plates = 0
        phenotypes_inclusion = self._phenotypes_inclusion

        if phenotypes_inclusion is not PhenotypeDataType.Trusted:
            self._logger.warning("Will extract phenotypes beyond those that are trusted, this is not recommended!" +
                                 " It is your responsibility to verify the validity of those phenotypes!")

        for id_plate, plate in enumerate(self._smooth_growth_data):

            if plate is None:
                all_phenotypes.append(None)
                all_vector_phenotypes.append(None)
                all_vector_meta_phenotypes.append(None)
                continue

            plate_flat_regression_strided = self._get_plate_linear_regression_strided(plate)

            phenotypes = np.zeros((plate.shape[:2]) + (phenotypes_max_value,), dtype=np.float)
            plate_size = np.prod(plate.shape[:2])
            self._logger.info("Plate {0} has {1} curves".format(id_plate + 1, plate_size))

            vector_phenotypes = {
                p: np.zeros(plate.shape[:2], dtype=np.object) * np.nan
                for p in VectorPhenotypes if phenotypes_inclusion(p)}

            vector_meta_phenotypes = {}

            all_phenotypes.append(phenotypes)
            all_vector_phenotypes.append(vector_phenotypes)
            all_vector_meta_phenotypes.append(vector_meta_phenotypes)

            for pos_index, pos_data in enumerate(plate_flat_regression_strided):

                position_phenotypes = [None] * phenotypes_max_value

                id1 = pos_index % plate.shape[1]
                id0 = pos_index / plate.shape[1]

                curve_data = get_preprocessed_data_for_phenotypes(
                    curve=plate[id0, id1],
                    curve_strided=pos_data,
                    flat_times=flat_times,
                    times_strided=times_strided,
                    index_for_48h=index_for_48h,
                    position_offset=position_offset)

                curve_has_data = True

                if curve_data['curve_smooth_growth_data'].mask.all():

                    self._logger.warning("Position ({0}, {1}) on plate {2} seems void of data".format(
                        id0, id1, id_plate + 1
                    ))
                    curve_has_data = False

                else:

                    for phenotype in Phenotypes:

                        if not phenotypes_inclusion(phenotype):
                            continue

                        if PhenotypeDataType.Scalar(phenotype):

                            try:
                                position_phenotypes[phenotype.value] = phenotype(**curve_data)
                            except IndexError:
                                self._logger.critical(
                                    "Could not store {0} (index {1}) expected max {2}.".format(
                                        phenotype, phenotype.value, phenotypes_max_value))
                                return

                phenotypes[id0, id1, ...] = position_phenotypes

                if curve_has_data and (
                            phenotypes_inclusion(VectorPhenotypes.PhasesClassifications) or
                            phenotypes_inclusion(VectorPhenotypes.PhasesPhenotypes)):

                    phases, phases_phenotypes = get_phase_analysis(
                        self, id_plate, (id0, id1),
                        experiment_doublings=position_phenotypes[Phenotypes.ExperimentPopulationDoublings.value])

                    if phenotypes_inclusion(VectorPhenotypes.PhasesClassifications):
                        vector_phenotypes[VectorPhenotypes.PhasesClassifications][id0, id1] = phases
                    if phenotypes_inclusion(VectorPhenotypes.PhasesPhenotypes):
                        vector_phenotypes[VectorPhenotypes.PhasesPhenotypes][id0, id1] = phases_phenotypes

                if id1 == 0:

                    self._logger.debug("Done plate {0} pos {1} {2} {3}".format(
                        id_plate, id0, id1, list(position_phenotypes)))

                    self._logger.info("Plate {1} growth phenotypes {0:.1f}% done".format(
                        100.0 * (pos_index + 1.0) / plate_size,
                        id_plate + 1,
                    ))

                    yield (curves_in_completed_plates + pos_index + 1.0) / total_curves

            for phenotype in CurvePhaseMetaPhenotypes:

                self._logger.info("Extracting {0} for plate {1}".format(phenotype.name, id_plate + 1))

                if not phenotypes_inclusion(phenotype):
                    continue

                if not phenotypes_inclusion(VectorPhenotypes.PhasesPhenotypes):
                    self._logger.warning("Can't extract {0} because {1} has not been included.".format(
                        phenotype, VectorPhenotypes.PhasesPhenotypes))
                    continue

                phenotype_data = extract_phenotypes(vector_phenotypes[VectorPhenotypes.PhasesPhenotypes],
                                                    phenotype, phenotypes)

                vector_meta_phenotypes[phenotype] = phenotype_data.astype(np.float)

            self._logger.info("Plate {0} Done".format(id_plate + 1))
            curves_in_completed_plates += 0 if plate is None else plate_flat_regression_strided.shape[0]

        self._phenotypes = np.array(all_phenotypes)
        self._vector_phenotypes = np.array(all_vector_phenotypes)
        self._vector_meta_phenotypes = np.array(all_vector_meta_phenotypes)
        self._normalized_phenotypes = None
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
        """ Add a phenotype to the set of phenotypes that are normalized.

        Only those for which there is previously extracted non-normalized data
        are included in the normalization, so you may need to change inclusion level
        and rerun feature extraction before any change have affect on the data
        produced by normalization.

        Args:
            phenotype: The phenotype to include.
                Typically this is one of
                    * `scanomatic.data_processing.growth_phenotypes.Phenotypes`
                      (Based on direct analysis of curves)
                    * `scanomatic.data_processing.curve_phase.phenotypes.CurvePhasePhenotypes`
                      (Based on segmenting curves into phases)

        Notes:
            If the current module `phenotyper` was imported, it will have imported the
            `phenotyper.Phenotypes` and `phenotyper.CurvePhasePhenotypes`. For more
            information on these refer to their respective help.

        See Also:
            Phenotyper.remove_phenotype_from_normalization: Removing phenotype from normalization
            Phenotyper.set_phenotype_inclusion_level: Setting which phenotypes are extracted.
            Phenotyper.normalize_phenotypes: Normalize phenotypes
        """
        self._normalizable_phenotypes.add(phenotype)

    def remove_phenotype_from_normalization(self, phenotype):
        """Removes a phenotype so that it is not normalized.

        Args:
            phenotype: The phenotype to include.
                Typically this is one of
                    * `scanomatic.data_processing.growth_phenotypes.Phenotypes`
                      (Based on direct analysis of curves)
                    * `scanomatic.data_processing.curve_phase.phenotypes.CurvePhasePhenotypes`
                      (Based on segmenting curves into phases)

        See Also:
            Phenotyper.add_phenotype_to_normalization: Adding phenotype to normalization
            Phenotyper.set_phenotype_inclusion_level: Setting which phenotypes are extracted.
            Phenotyper.normalize_phenotypes: Normalize phenotypes
        """
        self._normalizable_phenotypes.remove(phenotype)

    def get_normalizable_phenotypes(self):

        return self._normalizable_phenotypes

    def get_curve_phases(self, plate, outer, inner):

        try:
            val = self._vector_phenotypes[plate][VectorPhenotypes.PhasesClassifications][outer, inner]
            if isinstance(val, np.ma.masked_array):
                return val
            return None
        except (ValueError, IndexError, TypeError, KeyError):
            return None

    def get_curve_phases_at_time(self, plate, time_index):

        try:
            p = self._vector_phenotypes[plate][VectorPhenotypes.PhasesClassifications]

            # The init value is illegal, no phase has that value. On purpose
            arr = np.ones(p.shape, dtype=int) * -2
            for id1, id2 in product(*(range(d) for d in p.shape)):
                v = self.get_curve_phases(plate, id1, id2)
                if v is not None:
                    arr[id1, id2] = v[time_index]
            return np.ma.masked_array(arr, arr == -2)
        except (ValueError, IndexError, TypeError, KeyError):
            return None

    def get_curve_phase_data(self, plate, outer, inner):

        try:
            return self._vector_phenotypes[plate][VectorPhenotypes.PhasesPhenotypes][outer, inner]
        except (ValueError, IndexError, TypeError, KeyError):
            return None

    @property
    def phenotypes(self):
        return tuple(p for p in self._phenotypes_inclusion())

    @property
    def phenotypes_that_normalize(self):

        return tuple(v for v in set(self._normalizable_phenotypes.intersection(self._phenotypes_inclusion())))

    @property
    def phenotypes_that_dont_normalize(self):

        return tuple(set(self.phenotypes).difference(self.phenotypes_that_normalize))

    def set_control_surface_offsets(self, offset, plate=None):
        """Set which of four offsets is the control surface positions.

        When a new `Phenotyper` instance is created, the offset is assumed to be
        `Offsets.LowerRight`.

        Args:
            offset: The offset used, one of the `scanomatic.data_processing.norm.Offsets`.
            plate: Optional, plate index (start at 0 for first plate), default is to
             set offset to all plates.
        """
        if plate is None:
            self._reference_surface_positions = [offset() for _ in self.enumerate_plates]
        else:
            self._reference_surface_positions[plate] = offset()

    def get_control_surface_offset(self, plate):

        try:
            return self._reference_surface_positions[plate]
        except IndexError:
            return 0

    def normalize_phenotypes(self):
        """Normalize phenotypes.

        See Also:
            Phenotyper.get_phenotype: Getting phenotypes, including normalized versions.
            Phenotyper.set_control_surface_offsets: setting which offset is the control surface
            Phenotyper.add_phenotype_to_normalization: Adding phenotype to normalization
            Phenotyper.remove_phenotype_from_normalization: Removing phenotype from normalization
            Phenotyper.set_phenotype_inclusion_level: Setting which phenotypes are extracted.
        """
        if self._normalized_phenotypes is None:
            self._normalized_phenotypes = np.array([{} for _ in self.enumerate_plates], dtype=np.object)

        for phenotype in self._normalizable_phenotypes:

            if self._phenotypes_inclusion(phenotype) is False:
                self._logger.info("Because {0} has not been extracted it is skipped".format(phenotype))
                continue

            try:
                data = self.get_phenotype(phenotype)
            except (ValueError, KeyError):
                self._logger.info("{0} had not been extracted, so skipping it".format(phenotype))
                continue

            if all(v is None for v in data):
                self._logger.info("{0} had not been extracted, so skipping it".format(phenotype))
                continue

            for id_plate, plate in enumerate(get_normalized_data(data, self._reference_surface_positions)):
                self._normalized_phenotypes[id_plate][phenotype] = plate

    @property
    def number_of_curves(self):

        return sum(p.shape[0] * p.shape[1] if (p is not None and p.ndim > 1) else 0 for p in self._raw_growth_data)

    def get_number_of_phenotypes(self, normed=False):

        return len(self.phenotypes_that_normalize if normed else self.phenotypes)

    @property
    def generation_times(self):

        return self.get_phenotype(Phenotypes.GenerationTime)

    def get_reference_median(self, phenotype):
        """ Getting reference position medians per plate.

        Args:
            phenotype: The phenotype of interest.

        Returns: tuple of floats

        """
        plates = self.get_phenotype(phenotype, filtered=False)
        return tuple(
            np.ma.median(np.ma.masked_invalid(get_reference_positions([plate], [offset]))) for
            plate, offset in zip(plates, self._reference_surface_positions))

    def get_phenotype(self, phenotype, filtered=True, norm_state=NormState.Absolute,
                      reference_values=None, **kwargs):
        """Getting phenotype data

        Args:
            phenotype:
                The phenotype, either a `scanomatic.data_processing.growth_phenotypes.Phenotypes`
                or a `scanomatic.data_processing.curve_phase_phenotypes.CurvePhasePhenotypes`
            filtered:
                Optional, if the log2_curve-markings should be present or not on the returned object.
                Defaults to including log2_curve markings.
            norm_state:
                Optional, the type of data-state to return.
                If `NormState.NormalizedAbsoluteNonBatched`, then `reference_values` must be supplied.
            reference_values:
                Optional, tuple of the means of all comparable plates-medians
                of their reference positions.
                One value per plate in the current project.

        Returns:
            List of plate-wise phenotype data. Depending on the `filtered` argument this is either `FilteredArrays`
            that behave similar to `numpy.ma.masked_array` or pure `numpy.ndarray`s for non-filtered data.

        See Also:
            Phenotyper.get_reference_median:
                Produces reference values for plates.
        """

        if phenotype not in self:

            raise ValueError(
                "'{0}' has not been extracted, please re-run 'extract_phenotypes()' to include it.".format(
                    phenotype.name))

        if 'normalized' in kwargs:
            self._logger.warning("Deprecation warning: Use phenotyper.NormState enums instead")
            norm_state = NormState.NormalizedRelative if kwargs['normalized'] else NormState.Absolute

        if not PhenotypeDataType.Trusted(phenotype):
            self._logger.warning("The phenotype '{0}' has not been fully tested and verified!".format(phenotype.name))

        self._init_remove_filter_and_undo_actions()

        if norm_state is NormState.NormalizedRelative:

            return self._get_norm_phenotype(phenotype, filtered)

        elif norm_state is NormState.Absolute:

            return self._get_abs_phenotype(phenotype, filtered)

        else:

            normed_plates = self._get_norm_phenotype(phenotype, filtered)
            if norm_state is NormState.NormalizedAbsoluteBatched:
                reference_values = self.get_reference_median(phenotype)
            return tuple((None if ref_val is None or plate is None else ref_val * np.power(2, plate))
                         for ref_val, plate in izip(reference_values, normed_plates))

    def _get_norm_phenotype(self, phenotype, filtered):

        if self._normalized_phenotypes is None or not \
                all(True if p is None else phenotype in p for p in self._normalized_phenotypes):

            if self._normalized_phenotypes is None:
                self._logger.warning("No phenotypes have been normalized")
            else:
                self._logger.warning("Phenotypes {0} not included in normalized phenotypes".format(phenotype))
            return [None for _ in self._phenotype_filter]

        if filtered:

            return [None if (p is None or phenotype not in self._phenotype_filter[id_plate]) else
                    FilterArray(p[phenotype], self._phenotype_filter[id_plate][phenotype])
                    for id_plate, p in enumerate(self._normalized_phenotypes)]

        else:

            return [None if p is None else p[phenotype] for _, p in enumerate(self._normalized_phenotypes)]

    def _get_abs_phenotype(self, phenotype, filtered):

        if isinstance(phenotype, Phenotypes):
            data = self._restructure_growth_phenotype(phenotype)
        else:
            data = self._get_phenotype_data(phenotype)

        if filtered:

            return [None if (p is None or phenotype not in self._phenotype_filter[id_plate]) else
                    FilterArray(p, self._phenotype_filter[id_plate][phenotype])
                    for id_plate, p in enumerate(data)]
        else:
            return data

    def _get_phenotype_data(self, phenotype):

        if isinstance(phenotype, CurvePhaseMetaPhenotypes) and self._vector_meta_phenotypes is not None:
            return [None if p is None else p[phenotype] for p in self._vector_meta_phenotypes]
        return [None for _ in self.enumerate_plates]

    def _restructure_growth_phenotype(self, phenotype):

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

        return [None if (p is None or phenotype not in self) else _plate_type_converter(p[..., phenotype.value])
                for p in self._phenotypes]

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

    def get_quality_index(self, plate):

        shape = tuple(self.plate_shapes)[plate]

        try:
            gt = self.get_phenotype(Phenotypes.GenerationTime)[plate].data
        except (AttributeError, IndexError):
            gt = np.ones(shape)

        try:
            gt_err = self.get_phenotype(Phenotypes.GenerationTimeStErrOfEstimate)[plate].data
        except (AttributeError, IndexError):
            gt_err = np.ones(shape)

        try:
            cr_fit = self.get_phenotype(Phenotypes.ChapmanRichardsFit)[plate].data
        except (AttributeError, IndexError):
            cr_fit = np.ones(shape)

        try:
            lag = self.get_phenotype(Phenotypes.GrowthLag)[plate].data
        except (AttributeError, IndexError):
            lag = np.ones(shape)

        try:
            growth = self.get_phenotype(Phenotypes.ExperimentGrowthYield)[plate].data
        except (AttributeError, IndexError):
            growth = np.ones(shape)

        gt_mean = gt[np.isfinite(gt)].mean()
        lag_mean = lag[np.isfinite(lag)].mean()
        growth_mean = growth[np.isfinite(growth)].mean()

        badness = (np.abs(gt - gt_mean) / gt_mean +
                   gt_err * 100 +
                   (1 - cr_fit.clip(0, 1)) * 25 +
                   np.abs(lag - lag_mean) / lag_mean +
                   np.abs(growth - growth_mean) / growth)

        return np.unravel_index(badness.ravel().argsort(), badness.shape)

    def set(self, data_type, data):

        if data_type == 'phenotypes':

            if isinstance(data, np.ndarray) and (data.size == 0 or not data.any()):
                self._phenotypes = None

            else:
                self._phenotypes = data

            self._init_remove_filter_and_undo_actions()
            self._init_default_offsets()

        elif data_type == 'reference_offsets':

            self._reference_surface_positions = data

        elif data_type == 'normalized_phenotypes':

            if isinstance(data, np.ndarray) and (data.size == 0 or not data.any()):
                self._normalized_phenotypes = None
            else:
                self._normalized_phenotypes = data

            self._init_default_offsets()

        elif data_type == 'vector_phenotypes':

            self._vector_phenotypes = data

            self._init_remove_filter_and_undo_actions()
            self._init_default_offsets()

        elif data_type == 'vector_meta_phenotypes':

            self._vector_meta_phenotypes = data

            self._init_remove_filter_and_undo_actions()
            self._init_default_offsets()

        elif data_type == 'smooth_growth_data':

            self._smooth_growth_data = data

        elif data_type == "phenotype_filter_undo":

            if isinstance(data, tuple) and all(isinstance(q, deque) for q in data):
                self._phenotype_filter_undo = data
            else:
                self._logger.warning("Not a proper undo history")

            self._init_remove_filter_and_undo_actions()

        elif data_type == "phenotype_filter":

            if all(True if plate is None else isinstance(plate, dict) for plate in data):
                self._phenotype_filter = data
            else:
                self._phenotype_filter = self._convert_to_current_phenotype_filter(data)

            self._init_remove_filter_and_undo_actions()

        elif data_type == "meta_data":

            if isinstance(data, MetaData) or data is None:
                self._meta_data = data
            else:
                self._logger.warning("Not a valid meta data type")
        else:

            self._logger.warning('Unknown type of data {0}'.format(data_type))

    def _convert_to_current_phenotype_filter(self, data):

        self._logger.info("Converting old filter format to new.")
        self._logger.warning("If you save the state the qc-filter will not be readable to old scan-o-matic qc.")
        new_data = []
        for id_plate, plate in enumerate(data):

            if plate is None or plate.size == 0:
                new_data.append(None)
                continue

            if plate.ndim == 3:
                new_plate = {}

                for id_phenotype in range(plate.shape[-1]):
                    phenotype = plate[..., id_phenotype]
                    if phenotype.dtype == bool or np.unique(phenotype).max() == 1:
                        phenotype = phenotype.astype(np.uint8) * Filter.UndecidedProblem.value
                    else:
                        phenotype = phenotype.astype(np.uint8)
                    phenotype.clip(min(f.value for f in Filter), max(f.value for f in Filter))

                    try:
                        new_plate[Phenotypes(id_phenotype)] = phenotype
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

    def _init_default_offsets(self):

        if len(self._reference_surface_positions) != len(self._phenotypes):
            self.set_control_surface_offsets(Offsets.LowerRight)

    def _init_plate_filter(self, plate_index, phenotype, phenotype_data, growth_filter):

        self._phenotype_filter[plate_index][phenotype] = np.zeros(
            self._raw_growth_data[plate_index].shape[:2], dtype=np.int8)

        if phenotype_data is not None:
            self._phenotype_filter[plate_index][phenotype][np.where(np.isfinite(phenotype_data) == False)] = \
                Filter.UndecidedProblem.value

            self._phenotype_filter[plate_index][phenotype][growth_filter] = Filter.NoGrowth.value

        if self._phenotype_filter_undo[plate_index]:
            self._logger.warning(
                "Undo cleared for plate {0} because of rewriting, better solution not yet implemented.".format(
                    plate_index + 1
                ))

    def _init_remove_filter_and_undo_actions(self):

        if self._phenotypes is None:
            self._phenotype_filter = None
            self._phenotype_filter_undo = None
            return

        if self._phenotype_filter is None or len(self._phenotypes) != len(self._phenotype_filter):

            self._logger.warning("Filter doesn't match number of plates. Rewriting...")
            self._phenotype_filter = np.array([{} for _ in range(self._phenotypes.shape[0])], dtype=np.object)
            self._phenotype_filter_undo = tuple(deque() for _ in self._phenotypes)

        elif self._phenotype_filter_undo is None or len(self._phenotypes) != len(self._phenotype_filter_undo):

            self._logger.warning("Undo doesn't match number of plates. Rewriting...")
            self._phenotype_filter_undo = tuple(deque() for _ in self._phenotypes)

        if Phenotypes.Monotonicity in self and Phenotypes.ExperimentPopulationDoublings in self:
            growth_filter = [
                ((plate[..., Phenotypes.Monotonicity.value] < self._no_growth_monotonicity_threshold) |
                 (np.isfinite(plate[..., Phenotypes.Monotonicity.value]) == np.False_)) &
                ((plate[..., Phenotypes.ExperimentPopulationDoublings.value] <
                  self._no_growth_pop_doublings_threshold) |
                 (np.isfinite(plate[..., Phenotypes.ExperimentPopulationDoublings.value]) == np.False_))
                for plate in self._phenotypes]
        elif Phenotypes.Monotonicity in self:
            growth_filter = [
                ((plate[..., Phenotypes.Monotonicity.value] < self._no_growth_monotonicity_threshold) |
                 (np.isfinite(plate[..., Phenotypes.Monotonicity.value]) == np.False_))
                for plate in self._phenotypes]
        elif Phenotypes.ExperimentPopulationDoublings in self:
            growth_filter = [
                ((plate[..., Phenotypes.ExperimentPopulationDoublings.value] <
                  self._no_growth_pop_doublings_threshold) |
                 (np.isfinite(plate[..., Phenotypes.ExperimentPopulationDoublings.value]) == np.False_))
                for plate in self._phenotypes]
        else:
            growth_filter = [[] for _ in range(self._phenotypes.shape[0])]

        for phenotype in self.phenotypes:

            if phenotype not in self:
                continue

            phenotype_data = self._get_abs_phenotype(phenotype, False)

            for plate_index in range(self._phenotypes.shape[0]):

                if phenotype_data[plate_index] is None:

                    continue

                if phenotype not in self._phenotype_filter[plate_index]:

                    self._init_plate_filter(plate_index, phenotype,
                                            phenotype_data[plate_index],
                                            growth_filter[plate_index])

                elif self._phenotype_filter[plate_index][phenotype].shape != phenotype_data[0].shape:

                    self._logger.warning("The phenotype filter doesn't match plate {0} shape!".format(plate_index + 1))
                    self._init_plate_filter(plate_index, phenotype,
                                            phenotype_data[plate_index],
                                            growth_filter[plate_index])

    def infer_filter(self, template, *phenotypes):
        """Transfer all marks on one phenotype to other phenotypes.

        Note that the Filter.OK is not transferred, thus not replacing any existing
        non Filter.OK marking for those positions.

        :param template: The phenotype used as source
        :param phenotypes: The phenotypes that should be updated

        """
        self._logger.warning("Inferring is done without ability to undo.")

        template_filt = {plate: self._phenotype_filter[plate][template] for plate in self.enumerate_plates}

        for mark in Filter:

            if mark == Filter.OK:
                continue

            for phenotype in phenotypes:

                for plate in self.enumerate_plates:

                    if phenotype in self._phenotype_filter[plate]:

                        self._phenotype_filter[plate][phenotype][template_filt[plate] == mark.value] = mark.value

    def get_curve_qc_filter(self, plate, threshold=0.8):

        filt = self._phenotype_filter[plate]

        if filt is None:
            return filt

        filt = np.array(filt.values())

        return ((np.sum(filt, axis=0) / float(filt.shape[0]) > threshold) |
                np.any((filt == Filter.NoGrowth.value) | (filt == Filter.Empty.value), axis=0))

    def add_position_mark(self, plate, positions, phenotype=None, position_mark=Filter.BadData,
                          undoable=True):
        """ Adds log2_curve mark for position or set of positions.

        Args:
            plate: The plate index (0 for firs)
            positions: Tuple of positions to be marked. _NOTE_: It must be a tuple and first index is 0,
                e.g. `(1, 2)` will mark the the second row, third column.
                If several positions should be marked at once the coordinates should be structured as a
                tuple of two tuples, first with the rows and second with the columns.
                e.g. `((0, 0, 1), (0, 1, 0))` will mark the corner triplicate excluding the lower right
                control position.
            phenotype: Optional, which phenotype to mark. If submitted only that pheontype will recieve the mark,
                defaults to placing mark for all phenotypes.
            position_mark: Optional, the mark to apply, one of the `scanomatic.generics.phenotype_filter.Filter`.
                Defaults to `Filter.BadData`.
            undoable: Optional, if mark should be undoable, default is yes [`True`, `False`].

        Returns: If query was accepted.

        Notes:
            * If the present module `scanomtic.data_processing.phenotyper` was imported, then you can
              reach the `phenotype_mark` filters at `phenotyper.Filter`.
            * If all pheontypes are to be marked at once, the undo-action will assume all phenotypes
              previously were `Filter.OK`. This may of may not have been true.

        """
        if position_mark in (Filter.Empty, Filter.NoGrowth) and phenotype is not None:
            self._logger.error("{0} can only be set for all phenotypes, not specifically for {1}".format(
                position_mark, phenotype))
            return False

        if phenotype is None:

            for phen in Phenotypes:
                if self._phenotypes_inclusion(phen):
                    return self.add_position_mark(plate, positions, phen, position_mark, undoable=False)

            if undoable:
                self._logger.warning("Undoing this mark will assume all phenotypes were previously marked as OK")
                self._add_undo(plate, positions, None, 0)

        else:

            previous_state = self._phenotype_filter[plate][phenotype][positions]

            if isinstance(previous_state, np.ndarray):
                if np.unique(previous_state).size == 1:
                    previous_state = previous_state[0]

            self._phenotype_filter[plate][phenotype][positions] = position_mark.value

            if undoable:
                self._add_undo(plate, positions, phenotype, previous_state)

        return True

    def _add_undo(self, plate, position_list, phenotype, previous_state):

        self._phenotype_filter_undo[plate].append((position_list, phenotype, previous_state))
        while len(self._phenotype_filter_undo[plate]) > self.UNDO_HISTORY_LENGTH:
            self._phenotype_filter_undo[plate].popleft()

    def undo(self, plate):
        """Undo most recent position mark that was undoable on plate

        Args:
            plate: The plate index (0 for first plate)

        Returns:
            bool, If anything was undone
        """
        if len(self._phenotype_filter_undo[plate]) == 0:
            self._logger.info("No more actions to undo")
            return False

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
            return False
        return True

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

    def get_csv_file_name(self, dir_path, save_data, plate_index):

        path = os.path.join(dir_path, self._paths.phenotypes_csv_pattern)
        return path.format(save_data.name, plate_index + 1)

    def save_phenotypes(self, dir_path=None, save_data=NormState.Absolute,
                        dialect=csv.excel, ask_if_overwrite=True, reference_values=None):
        """Exporting phenotypes to csv format.

        Args:
            dir_path: The directory where to put the data, file names will be
                automatically generated
            save_data: Optional, what data to save.
                One of the `NormState` values.
                Default is raw absolute phenotypes.
            dialect: Optional. The csv-dialect to use, defaults to excel-compatible.
            ask_if_overwrite: Optional, if warning before overwriting any files, defaults to `True`.
            reference_values:
                Optional, tuple of the means of all comparable plates-medians,
                needed for using non-batched saving
                of their reference positions.
                One value per plate in the current project.
        """
        if dir_path is None and self._base_name is not None:
            dir_path = self._base_name
        elif dir_path is None:
            self._logger.error("Needs somewhere to save the phenotype")
            return False

        dir_path = os.path.abspath(dir_path)

        phenotypes = self.phenotypes_that_normalize if save_data in (
            NormState.NormalizedAbsoluteBatched, NormState.NormalizedAbsoluteNonBatched,
            NormState.NormalizedRelative) else self.phenotypes

        phenotypes = tuple(p for p in phenotypes if PhenotypeDataType.Scalar(p) and p in self)

        data_source = {p: self.get_phenotype(p, norm_state=save_data, reference_values=reference_values)
                       for p in phenotypes}

        default_meta_data = ('Plate', 'Row', 'Column')

        meta_data = self._meta_data
        no_metadata = tuple()

        for plate_index in self.enumerate_plates:

            if any(data_source[p][plate_index] is None for p in data_source):
                continue

            plate_path = self.get_csv_file_name(dir_path, save_data, plate_index)

            if os.path.isfile(plate_path) and ask_if_overwrite:
                if 'y' not in raw_input("Overwrite existing file '{0}'? (y/N)".format(plate_path)).lower():
                    continue

            with open(plate_path, 'wb') as fh:

                cw = csv.writer(fh, dialect=dialect)

                # DATA
                plate = data_source[data_source.keys()[0]][plate_index]

                # HEADER ROW
                meta_data_headers = self.meta_data_headers(plate_index)

                cw.writerow(
                    tuple(self._make_csv_row(
                        default_meta_data,
                        meta_data_headers,
                        data_source.keys())))

                filt = self._phenotype_filter[plate_index]

                for idX, X in enumerate(plate):

                    for idY, Y in enumerate(X):

                        cw.writerow(
                            tuple(self._make_csv_row(
                                (plate_index, idX, idY),
                                no_metadata if meta_data is None else meta_data(plate_index, idX, idY),
                                tuple(data_source[p][plate_index][idX, idY] if filt[p][idX, idY] == 0 else
                                 Filter(filt[p][idX, idY]).name
                                 for p in data_source))))

                self._logger.info("Saved {0}, plate {1} to {2}".format(save_data, plate_index + 1, plate_path))

        return True

    @staticmethod
    def _do_ask_overwrite(path):
        return raw_input("Overwrite '{0}' (y/N)".format(
            path)).strip().upper().startswith("Y")

    def save_state(self, dir_path, ask_if_overwrite=True):
        """Save the `Phenotyper` instance's state for future work.

        Args:
            dir_path: Directory where state should be saved
            ask_if_overwrite: Optional, default is `True`
        """
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        p = os.path.join(dir_path, self._paths.phenotypes_raw_npy)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(p, self._phenotypes)

        p = os.path.join(dir_path, self._paths.vector_phenotypes_raw)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(p, self._vector_phenotypes)

        p = os.path.join(dir_path, self._paths.vector_meta_phenotypes_raw)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(p, self._vector_meta_phenotypes)

        p = os.path.join(dir_path, self._paths.normalized_phenotypes)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(p, self._normalized_phenotypes)

        p = os.path.join(dir_path, self._paths.phenotypes_input_data)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(p, self._raw_growth_data)

        p = os.path.join(dir_path, self._paths.phenotypes_input_smooth)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(p, self._smooth_growth_data)

        p = os.path.join(dir_path, self._paths.phenotypes_filter)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(p, self._phenotype_filter)

        p = os.path.join(dir_path, self._paths.phenotypes_reference_offsets)
        if not ask_if_overwrite or not os.path.isfile(p) or self._do_ask_overwrite(p):
            np.save(p, self._reference_surface_positions)

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
                 self._linear_regression_size,
                 None if self._phenotypes_inclusion is None else self._phenotypes_inclusion.name,
                 self._no_growth_monotonicity_threshold,
                 self._no_growth_pop_doublings_threshold])

        self._logger.info("State saved to '{0}'".format(dir_path))

    def for_each_call(self, extra_keyword_args=tuple(), start_plate=None, start_pos=None, funcs=tuple()):
        """For each log2_curve, the supplied functions are called.

        Each function must have the following initial argument order:

            phenotyper_object, plate, position_tuple

        After those, the rest of the arguments should have default values
        and/or have their needed information supplied in the
        ```extra_keyword_args``` parameter.

        Args:
            extra_keyword_args (tuple or dict):
                Tuple of dicts or a single dict used as extra keyword
                arguments sent to the functions. Either specific for
                each function or the same for all.
            start_plate (int): Starting plate to iterate from
            start_pos ((int, int)): Starting position on plate to iterate from
            funcs (tuple[function]):
                A list of functions to be called for each position.

        Returns:
            A generator to loop through the calling on the
            positions.

        Examples:

            Interactive plotting of the phase-classification of all
            curves:

            ```
            from matplotlib as plt
            from scanomatic.qc import phenotype_results

            def outer(i):
                for _ in i:
                    f.show()
                    yield
                    for ax in f.axes:
                        if ax is not ax1 and ax is not ax2:
                            f.delaxes(ax)
                        else:
                            ax.cla()

            f = plt.figure()
            ax1 = f.add_subplot(2, 1, 1,)
            ax2 = f.add_subplot(2, 1, 2)
            plt.ion()

            i = P.for_each_call(
                ({'ax': ax1}, {'ax': ax2}),
                phenotype_results.plot_phases,
                phenotype_results.plot_curve_and_derivatives)

            o = outer(i)
            ```

            Then ```next(o)``` can be used to step through the
            generator.
        """
        if not isinstance(extra_keyword_args, tuple):
            extra_keyword_args = tuple(extra_keyword_args for _ in range(len(self)))

        skip = False if start_pos is None else True

        for plate, shape in enumerate(self.plate_shapes):

            if not shape or start_plate is not None and plate < start_plate:
                continue

            for pos in product(*(range(s) for s in shape)):

                if skip:
                    if pos == start_pos:
                        skip = False
                    else:
                        continue

                for idf, f in enumerate(funcs):
                    f(self, plate, pos, **extra_keyword_args[idf])

                yield plate, pos
