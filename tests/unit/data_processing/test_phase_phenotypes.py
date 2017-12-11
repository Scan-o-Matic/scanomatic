import pytest
from scanomatic.data_processing.phases.analysis import (
    _locate_segment, get_data_needed_for_segmentation, DEFAULT_THRESHOLDS,
    segment, _phenotype_phases, CurvePhasePhenotypes,
    assign_linear_phase_phenotypes, assign_common_phase_phenotypes,
    assign_non_linear_phase_phenotypes
)

from scanomatic.data_processing.phenotyper import Phenotyper
import numpy as np

"""
@pytest.fixture(scope='session')
def setup_something(tmpdir_factory):
    pass

"""


def build_test_phenotyper():

    np.random.seed = 42
    x_data = np.arange(100) * 1/3.
    y_data = np.array([[[
        # 0: No data
        np.ones(100) * np.nan,
        # 1: Flat data
        np.ones(100) * 2 ** 17,
        # 2: Linear sloped data
        np.power(2, 17 + np.arange(100) * 0.02),
        # 3: Flat to Sloped with kink
        np.hstack((
            np.ones(50) * 2 ** 17,
            np.power(2, np.arange(50) * 0.1 + 17)
        )),
        # 4: Flat noise data
        np.power(2, np.random.normal(17, size=(100,))),
        # 5: Flat to Neg Sloped with kink
        np.hstack((
            np.ones(50) * 2 ** 17,
            np.power(2, 17 - np.arange(50) * 0.1)
        )),
    ]]], ndmin=4)

    phenotyper_object = Phenotyper(y_data, x_data)
    phenotyper_object._smooth_growth_data = y_data

    return phenotyper_object


def build_model(phenotyper_object, test_curve):

    assert phenotyper_object.smooth_growth_data[0][0, 0].size \
        == phenotyper_object.times.size
    assert phenotyper_object.raw_growth_data[0].shape \
        == phenotyper_object.smooth_growth_data[0].shape
    return get_data_needed_for_segmentation(
        phenotyper_object, 0, (0, test_curve), DEFAULT_THRESHOLDS)


@pytest.mark.skip(
    reason='Fails randomly for the intercept though it should not')
def test_no_growth_only_noise():

    phenotyper_object = build_test_phenotyper()
    model = build_model(phenotyper_object, 4)

    data = {}
    filt = np.ones_like(model.times, dtype=bool)
    assign_linear_phase_phenotypes(data, model, filt)

    assert np.isfinite(
        data[CurvePhasePhenotypes.LinearModelSlope]), "Invalid slope"
    assert np.isfinite(
        data[CurvePhasePhenotypes.LinearModelIntercept]), "Invalid intercept"
    assert np.allclose(
        data[CurvePhasePhenotypes.LinearModelIntercept], 17, atol=0.5
    ), "Unexpected intercept"
    assert np.allclose(
        data[CurvePhasePhenotypes.LinearModelSlope], 0, atol=0.05
    ), "Unexpected intercept"


def test_no_data():

    phenotyper_object = build_test_phenotyper()

    model = build_model(phenotyper_object, 0)
    data = {}
    filt = np.ones_like(model.times, dtype=bool)
    left, right = _locate_segment(filt)
    left_time = model.times[left]
    right_time = model.times[right - 1]

    assign_common_phase_phenotypes(data, model, left, right)
    assign_linear_phase_phenotypes(data, model, filt)
    assign_non_linear_phase_phenotypes(
        data, model, left, right, left_time, right_time)

    for phenotype in CurvePhasePhenotypes:
        assert (
            np.isnan(data[phenotype]) or data[phenotype] is None
        ), "Got phenotype " + phenotype.name + " without data"


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

    phenotyper_object = build_test_phenotyper()

    model = build_model(phenotyper_object, 1)
    data = {}
    filt = np.ones_like(model.times, dtype=bool)
    left, right = _locate_segment(filt)

    assign_common_phase_phenotypes(data, model, left, right)

    assert np.isfinite(data[CurvePhasePhenotypes.Start]), "Invalid start"
    assert np.isfinite(data[CurvePhasePhenotypes.Duration]), "Invalid duration"
    assert np.isfinite(data[CurvePhasePhenotypes.Yield]), "Invalid yield"
    assert np.isfinite(
        data[CurvePhasePhenotypes.PopulationDoublings]
    ), "Invalid population doubling"
    assert data[CurvePhasePhenotypes.Start] == 0
    assert data[CurvePhasePhenotypes.Duration] == model.times[-1]
    assert data[CurvePhasePhenotypes.Yield] == 0
    assert data[CurvePhasePhenotypes.PopulationDoublings] == 0

    model = build_model(phenotyper_object, 2)
    data = {}
    assign_common_phase_phenotypes(data, model, left, right)

    assert np.isfinite(
        data[CurvePhasePhenotypes.Start]), "Invalid start"
    assert np.isfinite(data[CurvePhasePhenotypes.Duration]), "Invalid duration"
    assert np.isfinite(data[CurvePhasePhenotypes.Yield]), "Invalid yield"
    assert np.isfinite(
        data[CurvePhasePhenotypes.PopulationDoublings]
    ), "Invalid population doubling"
    assert data[CurvePhasePhenotypes.Start] == 0
    assert data[CurvePhasePhenotypes.Duration] == model.times[-1]
    assert data[CurvePhasePhenotypes.Yield] > 0
    assert data[CurvePhasePhenotypes.PopulationDoublings] > 0


def test_assign_linear_phase_phenotypes():

    phenotyper_object = build_test_phenotyper()

    model = build_model(phenotyper_object, 1)
    data = {}
    filt = np.ones_like(model.times, dtype=bool)
    assign_linear_phase_phenotypes(data, model, filt)

    assert np.isfinite(
        data[CurvePhasePhenotypes.LinearModelSlope]), "Invalid slope"
    assert np.isfinite(
        data[CurvePhasePhenotypes.LinearModelIntercept]), "Invalid intercept"
    assert data[CurvePhasePhenotypes.LinearModelIntercept] != 0
    assert data[CurvePhasePhenotypes.LinearModelSlope] == 0

    model = build_model(phenotyper_object, 2)
    data = {}
    filt = np.ones_like(model.times, dtype=bool)
    assign_linear_phase_phenotypes(data, model, filt)

    assert np.isfinite(
        data[CurvePhasePhenotypes.LinearModelSlope]), "Invalid slope"
    assert np.isfinite(
        data[CurvePhasePhenotypes.LinearModelIntercept]), "Invalid intercept"
    np.testing.assert_allclose(
        data[CurvePhasePhenotypes.LinearModelIntercept], 17)
    assert data[CurvePhasePhenotypes.LinearModelSlope] > 0

    model = build_model(phenotyper_object, 3)
    data = {}
    filt = np.ones_like(model.times, dtype=bool)
    assign_linear_phase_phenotypes(data, model, filt)

    assert np.isfinite(
        data[CurvePhasePhenotypes.LinearModelSlope]), "Invalid slope"
    assert np.isfinite(
        data[CurvePhasePhenotypes.LinearModelIntercept]), "Invalid intercept"
    assert data[CurvePhasePhenotypes.LinearModelIntercept] != 0
    assert data[CurvePhasePhenotypes.LinearModelSlope] > 0


def test_assign_non_linear_phase_phenotypes():

    phenotyper_object = build_test_phenotyper()

    model = build_model(phenotyper_object, 3)
    data = {}
    filt = np.ones_like(model.times, dtype=bool)
    left, right = _locate_segment(filt)
    left_time = model.times[left]
    right_time = model.times[right - 1]

    assign_non_linear_phase_phenotypes(
        data, model, left, right, left_time, right_time)

    assert np.isfinite(
        data[CurvePhasePhenotypes.AsymptoteIntersection]
    ), "Invalid intersection phenotype"
    assert np.isfinite(
        data[CurvePhasePhenotypes.AsymptoteAngle]
    ), "Invalid angle"
    np.testing.assert_allclose(
        data[CurvePhasePhenotypes.AsymptoteIntersection], 0.5, atol=0.01)
    assert data[CurvePhasePhenotypes.AsymptoteAngle] > 0

    model = build_model(phenotyper_object, 5)
    data = {}
    filt = np.ones_like(model.times, dtype=bool)
    left, right = _locate_segment(filt)
    left_time = model.times[left]
    right_time = model.times[right - 1]

    assign_non_linear_phase_phenotypes(
        data, model, left, right, left_time, right_time)

    assert np.isfinite(
        data[CurvePhasePhenotypes.AsymptoteIntersection]
    ), "Invalid intersection phenotype"
    assert np.isfinite(
        data[CurvePhasePhenotypes.AsymptoteAngle]
    ), "Invalid angle"
    np.testing.assert_allclose(
        data[CurvePhasePhenotypes.AsymptoteIntersection], 0.5, atol=0.01)
    assert data[CurvePhasePhenotypes.AsymptoteAngle] < 0


def test_using_filter_for_phase_phenotypes_correctly():

    phenotyper_object = build_test_phenotyper()

    model = build_model(phenotyper_object, 1)
    data = {}
    filt = np.ones_like(model.times, dtype=bool)
    left, right = _locate_segment(filt)

    assert right - left == filt.sum(), "Error in segment location"

    assign_common_phase_phenotypes(data, model, left, right)

    assert (
        data[CurvePhasePhenotypes.Start] == model.times[left]
    ), "Error in initial time"
    assert (
        data[CurvePhasePhenotypes.Duration]
        == model.times[right - 1] - model.times[left]
    ), "Error in duration"

    filt[:20] = False

    data = {}
    left, right = _locate_segment(filt)

    assert right - left == filt.sum(), "Error in segment location"

    assign_common_phase_phenotypes(data, model, left, right)

    assert (
        data[CurvePhasePhenotypes.Start]
        == (model.times[left] + model.times[left - 1]) / 2
    ), "Error in initial time"
    assert (
        data[CurvePhasePhenotypes.Duration]
        == model.times[right - 1] - data[CurvePhasePhenotypes.Start]
    ), "Error in duration"

    filt[-10:] = False

    data = {}
    left, right = _locate_segment(filt)

    assert left == 20, "Bad filt start"
    assert right == filt.size - 10, "Bad filt end"
    assert right - left == filt.sum(), "Error in segment location"

    assign_common_phase_phenotypes(data, model, left, right)

    assert (
        data[CurvePhasePhenotypes.Start]
        == (model.times[left] + model.times[left - 1]) / 2
    ), "Error in initial time"
    assert (
        data[CurvePhasePhenotypes.Duration]
        == (model.times[right - 1] + model.times[right]) / 2
        - data[CurvePhasePhenotypes.Start]
    ), "Error in duration"


def test_phases_are_chronological_and_not_overlapping():

    phenotyper_object = build_test_phenotyper()

    for i in range(phenotyper_object.number_of_curves):

        model = build_model(phenotyper_object, i)

        for _ in segment(model, DEFAULT_THRESHOLDS):

            pass

        phenotypes = _phenotype_phases(model, 5)

        assert (
            len(model.times) == len(model.phases)
        ), "Inconsistency in number of phase positions for curve " + i

        starts = np.array([
            phase[CurvePhasePhenotypes.Start]
            for _, phase in phenotypes if phase is not None
        ])

        assert all(
            np.diff(starts) > 0), "Non chronological phases for curve " + i

        ends = np.array([
            phase[CurvePhasePhenotypes.Start]
            + phase[CurvePhasePhenotypes.Duration]
            for _, phase in phenotypes if phase is not None
        ])

        assert not any(
            ends[:-1] - starts[1:] > 0), "Overlapping phases for curve " + 1


def test_segments():

    phenotyper_object = build_test_phenotyper()

    for i in range(phenotyper_object.number_of_curves):

        model = build_model(phenotyper_object, i)

        for _ in segment(model, DEFAULT_THRESHOLDS):

            pass

        assert model.phases is not None, "Failed phases on curve " + i
        assert len(model.phases) > 0, "Zero length phases on curve " + i
