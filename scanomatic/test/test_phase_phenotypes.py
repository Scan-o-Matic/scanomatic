import pytest
from scanomatic.data_processing.phases.analysis import _locate_segment
import numpy as np

"""
@pytest.fixture(scope='session')
def setup_something(tmpdir_factory):
    pass

"""


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