import numpy as np
import pytest
from scanomatic.data_processing import phenotyper
import itertools

class TestSettingData:

    @pytest.fixture(scope='function')
    def empty_phenotyper(self):
        return phenotyper.Phenotyper([], [])

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

    def test_rejects_invalid_phenotype_filter(self, empty_phenotyper):
        assert False

    def test_rejects_invalid_phenotype_filter_undo(self, empty_phenotyper):
        assert False

    def test_accepts_phenotypes(self, empty_phenotyper):
        assert False
