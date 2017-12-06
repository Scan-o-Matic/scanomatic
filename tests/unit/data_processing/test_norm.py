import numpy as np
import pytest

from scanomatic.data_processing import norm


class TestDownSampling:

    def test_get_default_downsampling(self):
        """Test using Two plates with shape 4 x 6 each."""
        data = np.arange(48).reshape(2, 4, 6)
        sampled = norm.get_downsampled_plates(data)
        assert len(sampled) == 2
        a, b = sampled
        assert a.shape == (2, 3)
        assert a[0, 0] == 7
        assert b.shape == (2, 3)
        assert b[0, 0] == 31

    @pytest.mark.parametrize("setting,expected", [
        ('TL', 0),
        ('TR', 1),
        ('BL', 2),
        ('BR', 3),
    ])
    def test_get_downsampling_specified_downsampling_as_string(
            self, setting, expected):

        data = np.arange(4).reshape(1, 2, 2)
        sample = norm.get_downsampled_plates(data, setting)[0]
        assert sample.shape == (1, 1)
        assert sample[0, 0] == expected

    def test_get_specified_downsampling(self):

        data = np.arange(4).reshape(1, 2, 2)
        setting = np.array([
            [
                [1, 0],
                [0, 0],
            ]
        ])
        expected = 0
        sample = norm.get_downsampled_plates(data, setting)[0]
        assert sample.shape == (1, 1)
        assert sample[0, 0] == expected

    @pytest.mark.parametrize('settings', [
        (
            np.array([
                [
                    [0, 0],
                    [0, 0],
                ]
            ]),
        ),
        (
            np.array([
                [
                    [0, 1],
                    [0, 1],
                ]
            ]),
        ),
        (
            np.array([
                [
                    [0.25, 0.25],
                    [0.25, 0.25],
                ]
            ]),
        ),
        (
            np.array([
                [
                    [0, 0, 1],
                    [0, 0, 0],
                ]
            ]),
        ),

    ])
    def test_faulty_downsampling_raises(self, settings):

        data = np.arange(4).reshape(1, 2, 2)
        expected = 0

        with pytest.raises(ValueError):
            norm.get_downsampled_plates(data, settings)
