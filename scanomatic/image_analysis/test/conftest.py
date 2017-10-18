import pytest
from scipy import ndimage


@pytest.fixture(scope='session')
def easy_plate():
    return ndimage.io.imread(
        './scanomatic/image_analysis/test/testdata/test_fixture_easy.tiff')


@pytest.fixture(scope='session')
def hard_plate():
    return ndimage.io.imread(
        './scanomatic/image_analysis/test/testdata/test_fixture_difficult.tiff'
    )
