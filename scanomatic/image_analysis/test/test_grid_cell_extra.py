import numpy as np

from scanomatic.image_analysis import grid_cell_extra


def test_filter_array_is_bool():
    """filter_array should have dtype bool regardless of dtype of data"""
    blob = grid_cell_extra.Blob(
        identifier=(0, 0, 0), grid_array=np.ones((5, 5), dtype=np.float))
    assert blob.filter_array.dtype == np.bool
