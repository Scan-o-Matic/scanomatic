from __future__ import absolute_import

from itertools import product
import json
import os

from flask import Flask
import mock
import numpy as np
import pytest
from scipy.ndimage import center_of_mass
from scipy.stats import norm

from scanomatic.models.analysis_model import COMPARTMENTS
from scanomatic.ui_server import calibration_api
from scanomatic.ui_server.calibration_api import (
    get_bounding_box_for_colony, get_colony_detection
)


@pytest.mark.parametrize('data,expected', (
    (None, None),
    ([], None),
    ((1, 2), (1, 2)),
    (('1', '2'), (1, 2)),
))
def test_get_int_tuple(data, expected):

    assert calibration_api.get_int_tuple(data) == expected


def test_get_bounding_box_for_colony():

    # 3x3 colony grid
    grid = np.array(
        [
            # Colony positions' y according to their positions in the grid
            [
                [51, 102, 151],
                [51, 101, 151],
                [50, 102, 152],
            ],

            # X according to their positions on the grid
            [
                [75, 125, 175],
                [75, 123, 175],
                [75, 125, 175],
            ]
        ]
    )
    width = 50
    height = 30

    for x, y in product(range(3), range(3)):

        box = calibration_api.get_bounding_box_for_colony(
            grid, x, y, width, height)

        assert (box['center'] == grid[:, y, x]).all()
        assert box['yhigh'] - box['ylow'] == height + 1
        assert box['xhigh'] - box['xlow'] == width + 1
        assert box['xlow'] >= 0
        assert box['ylow'] >= 0


def test_get_boundin_box_for_colony_if_grid_partially_outside():
    """only important that never gets negative numbers for box"""

    grid = np.array(
        [
            [
                [-5, 10],
                [51, 101],
            ],

            [
                [10, 125],
                [-5, 123],
            ]
        ]
    )
    width = 50
    height = 30

    for x, y in product(range(2), range(2)):

        box = calibration_api.get_bounding_box_for_colony(
            grid, x, y, width, height)

        assert box['center'][0] >= 0
        assert box['center'][1] >= 0
        assert box['xlow'] >= 0
        assert box['ylow'] >= 0


@pytest.mark.parametrize('grid,x,y,w,h,expected', (
    (np.array([[[10]], [[10]]]), 0, 0, 5, 6,
     {'ylow': 7, 'yhigh': 14, 'xlow': 8, 'xhigh': 13, 'center': (10, 10)}),
    (np.array([[[10, 20]], [[10, 10]]]), 1, 0, 5, 6,
     {'ylow': 17, 'yhigh': 24, 'xlow': 8, 'xhigh': 13, 'center': (20, 10)}),
    (np.array([[[5]], [[10]]]), 0, 0, 10, 20,
     {'ylow': 0, 'yhigh': 16, 'xlow': 5, 'xhigh': 16, 'center': (5, 10)}),
))
def test_bounding_box_for_colony(grid, x, y, w, h, expected):
    result = get_bounding_box_for_colony(grid, x, y, w, h)
    assert result == expected


@pytest.fixture(scope='function')
def colony_image():
    im = np.ones((25, 25)) * 80
    cell_vector = norm.pdf(np.arange(-5, 6)/2.)
    colony = np.multiply.outer(cell_vector, cell_vector) * 20
    im[6:17, 5:16] -= colony
    return im


class TestGetColonyDetection:

    def test_colony_is_darker(self, colony_image):
        grid_cell = get_colony_detection(colony_image)
        blob = grid_cell.get_item(COMPARTMENTS.Blob).filter_array
        background = grid_cell.get_item(COMPARTMENTS.Background).filter_array
        assert (
            grid_cell.source[blob].mean() <
            grid_cell.source[background].mean()
        )

    def test_blob_and_background_dont_overlap(self, colony_image):
        grid_cell = get_colony_detection(colony_image)
        blob = grid_cell.get_item(COMPARTMENTS.Blob).filter_array
        background = grid_cell.get_item(COMPARTMENTS.Background).filter_array
        assert (blob & background).sum() == 0

    def test_blob_is_of_expected_size(self, colony_image):
        grid_cell = get_colony_detection(colony_image)
        blob = grid_cell.get_item(COMPARTMENTS.Blob).filter_array
        assert blob.sum() == pytest.approx(100, abs=10)

    def test_blob_has_expected_center(self, colony_image):
        grid_cell = get_colony_detection(colony_image)
        blob = grid_cell.get_item(COMPARTMENTS.Blob).filter_array
        assert center_of_mass(blob) == pytest.approx((11, 10), abs=1)
