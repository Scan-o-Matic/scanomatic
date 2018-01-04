from __future__ import absolute_import
import pytest
import mock
import numpy as np
import os

from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from scanomatic.models.analysis_model import AnalysisModel
from scanomatic.data_processing.calibration import (
    get_polynomial_coefficients_from_ccc)


@pytest.fixture(scope='function')
def analysis_model():

    return AnalysisModelFactory.create()


@pytest.fixture(scope='function')
def analysis_serialized_object(analysis_model):

    return AnalysisModelFactory.serializer.serialize(analysis_model)


@pytest.fixture(scope='session')
def data_path():
    return os.path.join(os.path.dirname(__file__), 'data')


class TestAnalysisModels:

    def test_model_has_ccc(self, analysis_model):

        assert hasattr(analysis_model, 'cell_count_calibration')

    def test_model_has_ccc_id(self, analysis_model):

        assert hasattr(analysis_model, 'cell_count_calibration_id')

    def test_model_can_serialize(self, analysis_model):

        serial = AnalysisModelFactory.serializer.serialize(analysis_model)
        assert len(serial) == 2

    def test_model_can_deserialize(self, analysis_serialized_object):

        result = AnalysisModelFactory.serializer.load_serialized_object(
            analysis_serialized_object)

        assert len(result) == 1
        assert isinstance(result[0], AnalysisModel)

    def test_can_create_using_default_ccc(self, analysis_model):

        default = get_polynomial_coefficients_from_ccc('default')

        np.testing.assert_allclose(
            analysis_model.cell_count_calibration, default)

        assert analysis_model.cell_count_calibration_id == 'default'

    def test_doesnt_overwrite_poly_coeffs_if_no_ccc_specified(self):

        coeffs = [1, 1, 2, 3, 5, 8]
        model = AnalysisModelFactory.create(cell_count_calibration=coeffs)
        np.testing.assert_allclose(model.cell_count_calibration, coeffs)
        assert model.cell_count_calibration_id is None

    @mock.patch(
        'scanomatic.models.factories.analysis_factories.get_polynomial_coefficients_from_ccc',
        return_value=[1, 3, 9, 27])
    def test_can_create_using_ccc_id(self, my_mock):

        model = AnalysisModelFactory.create(cell_count_calibration_id='mock')
        assert model.cell_count_calibration_id == 'mock'
        np.testing.assert_allclose(model.cell_count_calibration, [1, 3, 9, 27])

    def test_create_with_unknown_ccc_raises_error(self):

        with pytest.raises(KeyError):
            AnalysisModelFactory.create(cell_count_calibration_id='BadCCC')

    @pytest.mark.parametrize('basename', (
        'analysis.model',
        'analysis.model.2017.11',
        'analysis.model.2017.12',
    ))
    def test_can_load_serialized_files_from_disk(self, basename, data_path):
        model = AnalysisModelFactory.serializer.load_first(
            os.path.join(data_path, basename)
        )
        assert isinstance(model, AnalysisModel)

    @pytest.mark.parametrize('basename', (
        'test.project.compilation',
    ))
    def test_cant_load_other_serialized_files_from_disk(
        self, basename, data_path
    ):
        model = AnalysisModelFactory.serializer.load_first(
            os.path.join(data_path, basename)
        )
        assert model is None

    @pytest.mark.parametrize('keys', (
        [1, 2, 3, 4],
        [
            'email',
            'use_local_fixture',
            'fake',
        ],
    ))
    def test_bad_keys_dont_match(self, keys):

        assert AnalysisModelFactory.all_keys_valid(keys) is False

    def test_right_keys_match(self):

        assert AnalysisModelFactory.all_keys_valid(
            tuple(AnalysisModelFactory.default_model.keys()))
