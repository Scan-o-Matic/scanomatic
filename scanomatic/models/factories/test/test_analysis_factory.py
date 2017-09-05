import pytest
import numpy as np

from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from scanomatic.models.analysis_model import (
    AnalysisModel, get_original_calibration)


@pytest.fixture(scope='function')
def analysis_model():

    return AnalysisModelFactory.create()


@pytest.fixture(scope='function')
def analysis_serialized_object(analysis_model):

    return AnalysisModelFactory.serializer.serialize(analysis_model)


class TestAnalysisModels:

    def test_model_has_ccc(self, analysis_model):

        assert hasattr(analysis_model, 'cell_count_calibration')

    def test_model_can_serialize(self, analysis_model):

        serial = AnalysisModelFactory.serializer.serialize(analysis_model)
        assert len(serial) == 3

    def test_model_can_deserialize(self, analysis_serialized_object):

        result = AnalysisModelFactory.serializer.load_serialized_object(
            analysis_serialized_object)

        assert len(result) == 1
        assert isinstance(result[0], AnalysisModel)

    def test_can_load_default_ccc(self, analysis_serialized_object):

        result = AnalysisModelFactory.serializer.load_serialized_object(
            analysis_serialized_object)

        np.testing.assert_allclose(
            result[0].cell_count_calibration,
            get_original_calibration())

    def test_load_unknown_ccc_throws_error(self):

        with pytest.raises(KeyError):
            AnalysisModelFactory.create(cell_count_calibration='BadCCC')
