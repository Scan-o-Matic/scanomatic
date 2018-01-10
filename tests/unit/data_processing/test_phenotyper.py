from __future__ import absolute_import
import numpy as np
import pytest
from scanomatic.data_processing import phenotyper
import itertools


@pytest.fixture(scope='function')
def empty_phenotyper():
    return phenotyper.Phenotyper(np.array(
        [[[
            [100, 2000, 3000], [50, 70, 80]], [
            [1000, 2000, 3000], [80, 2000, 4000]]]]),
        np.array([0, 20, 40]))


class TestSettingData:

    def test_set_unknown_data(self, empty_phenotyper):

        assert not empty_phenotyper.set('unknown', np.array([1]))

    @pytest.mark.parametrize("key,data", itertools.product((
        'phenotypes', 'normalized_phenotypes', 'vector_phenotypes',
        'vector_meta_phenotypes', 'smooth_growth_data'
    ), (
        None, np.array(None), np.array([]), np.array([None, None]),
        np.array([None, np.array([None])]),
        np.array([[None, None], [None, None]]),
        np.array([{0: np.array([[None]])}]), np.zeros((10, 10), dtype=int),
    )))
    def test_rejects_empty_data(self, empty_phenotyper, key, data):
        assert not empty_phenotyper.set(key, data)

    def test_rejects_invalid_meta_data(self, empty_phenotyper):
        my_meta = "I'm so meta"
        assert not empty_phenotyper.set('meta_data', my_meta)
        assert empty_phenotyper.meta_data != my_meta

    @pytest.mark.parametrize('data', (
        np.array([]),
    ))
    def test_rejects_invalid_phenotype_filter(self, empty_phenotyper, data):
        assert not empty_phenotyper.set('phenotype_filter', data)

    @pytest.mark.parametrize('data', (
        None,
        [],
        (None, 5),
    ))
    def test_rejects_invalid_phenotype_filter_undo(
        self, empty_phenotyper, data
    ):
        assert not empty_phenotyper.set('pheontype_filter_undo', data)

    def test_accepts_any_reference_offset(self, empty_phenotyper):
        assert empty_phenotyper.set('reference_offsets', 42)

    @pytest.mark.parametrize('phenotype,data_name', (
        (phenotyper.Phenotypes.GenerationTime, 'phenotypes'),
        (
            phenotyper.CurvePhaseMetaPhenotypes.Modalities,
            'vector_meta_phenotypes',
        ),
    ))
    def test_accepts_phenotypes(self, empty_phenotyper, phenotype, data_name):
        values = np.array([[1, 1], [2, 1]])
        data = np.array([{phenotype: values}])
        assert empty_phenotyper.set(data_name, data)
        assert phenotype in empty_phenotyper
        assert (
            empty_phenotyper.get_phenotype(phenotype, False) == values).all()

    def test_accepts_vector_phenotypes(self, empty_phenotyper):
        phenotype = phenotyper.VectorPhenotypes.PhasesPhenotypes
        values = np.array([[1, 1], [2, 1]])
        data = np.array([{phenotype: values}])
        assert empty_phenotyper.set('vector_phenotypes', data)
        assert empty_phenotyper.get_curve_phase_data(0, 1, 0) == 2


class TestMarkingCurve:

    @pytest.fixture(scope='function')
    def phenotyper_with_phenotype(self, empty_phenotyper):
        values = np.array([[1, 3], [2, 4]])
        phenotype = phenotyper.Phenotypes.GenerationTime
        data = np.array([{phenotype: values}])
        assert empty_phenotyper.set('phenotypes', data)
        return empty_phenotyper

    def test_add_position_mark_default(self, phenotyper_with_phenotype):
        phenotyper_with_phenotype.add_position_mark(0, (1, 1))
        phenotype = phenotyper.Phenotypes.GenerationTime
        assert phenotype in phenotyper_with_phenotype
        data = phenotyper_with_phenotype.get_phenotype(phenotype)[0]
        assert data[0, 0] == 1
        assert data.filter[0, 0] == 0
        assert data[0, 1] == 3
        assert data.filter[0, 1] == 0
        assert data[1, 0] == 2
        assert data.filter[1, 0] == 0
        assert np.ma.is_masked(data[1, 1])
        assert data.filter[1, 1] == phenotyper.Filter.BadData.value
