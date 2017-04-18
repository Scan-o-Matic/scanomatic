import pytest
from scanomatic.data_processing.phases.analysis import _locate_segment, get_data_needed_for_segmentation,\
    DEFAULT_THRESHOLDS, segment, _phenotype_phases, CurvePhasePhenotypes

from scanomatic.data_processing.phenotyper import Phenotyper
import numpy as np

"""
@pytest.fixture(scope='session')
def setup_something(tmpdir_factory):
    pass

"""


def build_model(x_data, y_data):

    phenotyper_object = Phenotyper(y_data, x_data)
    phenotyper_object._smooth_growth_data = y_data

    assert phenotyper_object.smooth_growth_data[0][0, 0].size == phenotyper_object.times.size
    assert phenotyper_object.raw_growth_data[0].shape == phenotyper_object.smooth_growth_data[0].shape
    return get_data_needed_for_segmentation(phenotyper_object, 0, (0, 0), DEFAULT_THRESHOLDS)


def test_locate_segment():

    filt = np.ones((20,), dtype=bool)
    left, right = _locate_segment(filt)

    assert right - left == filt.sum()

    filt[:4] = False
    filt[-3:] = False

    left, right = _locate_segment(filt)

    assert left == 4
    assert right == 20 - 3
    assert right - left == filt.sum()


def test_assign_common_phase_phenotypes():

    assert False, "Not implemented"


def test_assign_linear_phase_phenotypes():

    assert False, "Not implemented"


def test_assign_non_linear_phase_phenotypes():

    assert False, "Not implemented"


def test_using_filter_for_phase_phenotypes_correctly():

    assert False, "Not implemented"


def test_phases_are_chornological_and_not_overlapping():

    model = build_model(np.arange(100) * 1/3.,
                        np.array(np.arange(100), ndmin=4) * 50000 + 5000)

    for _ in segment(model, DEFAULT_THRESHOLDS):

        pass

    phenotypes = _phenotype_phases(model, 5)

    assert len(model.times) == len(model.phases), "Inconsistency in number of phase positions"

    starts = np.array([phase[CurvePhasePhenotypes.Start] for _, phase in phenotypes])

    assert all(np.diff(starts) > 0), "Non chronological"

    ends = np.array([phase[CurvePhasePhenotypes.Start] + phase[CurvePhasePhenotypes.Duration] for _, phase in phenotypes])

    assert not any(ends[:-1] - starts[1:] > 0), "Overlapping"

    assert False, "Should test reasonable curves"


def test_segments():

    model = build_model(np.arange(100) * 1/3.,
                        np.array(np.arange(100), ndmin=4) * 50000 + 5000)
    for _ in segment(model, DEFAULT_THRESHOLDS):

        pass

    assert model.phases is not None
    assert len(model.phases) > 0
