import os
import mock
import pytest

from scanomatic.data_processing.analysis_loader import (
    AnalysisLoader, CorruptAnalysisError, PlateNotFoundError
)
ANALYSIS_DIRECTORY = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    'fixtures',
    'analysis'
))


@pytest.fixture
def mock_convert_url_to_path():
    m = mock.patch(
        'scanomatic.data_processing.analysis_loader.convert_url_to_path',
        return_value=ANALYSIS_DIRECTORY,
    )
    yield m.start()
    m.stop()


class TestAnalysisLoader:
    def test_converts_project_to_path(self, mock_convert_url_to_path):
        loader = AnalysisLoader('test')
        mock_convert_url_to_path.assert_called_once_with('test')
        assert loader.path == ANALYSIS_DIRECTORY

    def test_loads_times(self, mock_convert_url_to_path):
        loader = AnalysisLoader('test')
        assert loader.times.shape == (218, )

    def test_raises_exception_if_times_are_missing(self):
        loader = AnalysisLoader('test')
        with pytest.raises(CorruptAnalysisError):
            loader.times

    def test_loads_raw(self, mock_convert_url_to_path):
        loader = AnalysisLoader('test')
        assert loader.raw_growth_data.shape[0] == 4
        assert loader.raw_growth_data[0].shape == (32, 48, 218)

    def test_raises_exception_if_raw_are_missing(self):
        loader = AnalysisLoader('test')
        with pytest.raises(CorruptAnalysisError):
            loader.raw_growth_data

    def test_loads_smooth(self, mock_convert_url_to_path):
        loader = AnalysisLoader('test')
        assert loader.smooth_growth_data.shape[0] == 4
        assert loader.smooth_growth_data[0].shape == (32, 48, 218)

    def test_raises_exception_if_smoth_are_missing(self):
        loader = AnalysisLoader('test')
        with pytest.raises(CorruptAnalysisError):
            loader.smooth_growth_data

    def test_returns_reqested_plate(self, mock_convert_url_to_path):
        loader = AnalysisLoader('test')
        plate = loader.get_plate_data(1)
        expect = [
            28670.644118, 30694.83756341, 31778.23297606, 31825.43530291,
            33754.40538746,
        ]
        assert plate.raw[3, 2, :5] == pytest.approx(expect)
        expect = [
            31106.964553620215,
            31123.325816294455,
            31735.80178143327,
            32786.3651218409,
            34479.145420360335,
        ]
        assert plate.smooth[3, 2, :5] == pytest.approx(expect)

    def test_raises_exception_if_plate_not_known(
        self, mock_convert_url_to_path,
    ):
        loader = AnalysisLoader('test')
        with pytest.raises(PlateNotFoundError):
            loader.get_plate_data(43)
