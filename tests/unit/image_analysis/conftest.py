from __future__ import absolute_import

import os

import pytest
from scipy import ndimage


TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata')


@pytest.fixture(scope='session')
def easy_plate():
    return ndimage.io.imread(
        os.path.join(TESTDATA, 'test_fixture_easy.tiff')
    )


@pytest.fixture(scope='session')
def hard_plate():
    return ndimage.io.imread(
        os.path.join(TESTDATA, 'test_fixture_difficult.tiff')
    )
