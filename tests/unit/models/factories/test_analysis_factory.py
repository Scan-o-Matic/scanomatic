from __future__ import absolute_import

import os

import mock
import numpy as np
import pytest

from scanomatic.data_processing.calibration import (
    get_polynomial_coefficients_from_ccc
)
from scanomatic.models.analysis_model import AnalysisModel
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from tests.factories import mkcalibration


@pytest.fixture
def calibrationstore():
    return mock.MagicMock()


@pytest.fixture(autouse=True)
def store_from_env(calibrationstore):
    calibrationstore.get_all_calibrations.return_value = [
        mkcalibration(identifier='default', active=True),
    ]
    calibrationstore.get_calibration_by_id.return_value = (
        mkcalibration(active=True)
    )
    with mock.patch(
        'scanomatic.models.factories.analysis_factories.store_from_env',
    ) as store_from_env:
        store_from_env.return_value.__enter__.return_value = calibrationstore
        yield


@pytest.fixture(scope='function')
def analysis_model(store_from_env):
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

    def test_can_create_using_default_ccc(self, calibrationstore):
        default = [1, 2, 3, 4]
        calibrationstore.get_calibration_by_id.return_value = (
            mkcalibration(polynomial=default, active=True)
        )
        model = AnalysisModelFactory.create()
        np.testing.assert_allclose(
            model.cell_count_calibration, default)
        assert model.cell_count_calibration_id == 'default'

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

    def test_create_with_unknown_ccc_raises_error(self, calibrationstore):
        calibrationstore.get_calibration_by_id.side_effect = LookupError
        with pytest.raises(LookupError):
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

    def test_can_have_no_pinning_instructions(self):
        assert AnalysisModelFactory.create().pinning_matrices is None

    def test_using_default_pinning_is_valid(self):
        assert (
            'pinning_matrices' not in
            AnalysisModelFactory.get_invalid_names(
                AnalysisModelFactory.create()
            )
        )
