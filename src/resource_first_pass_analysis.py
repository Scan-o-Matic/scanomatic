#!/usr/bin/env python
"""Resource module for first pass analysis."""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os
import logging

_logger = logging.getLogger("1st Pass Analysis")

#
# INTERNAL DEPENDENCIES
#

import src.resource_fixture_image as resource_fixture_image

#
# EXCEPTIONS
#


class Marker_Detection_Failed(Exception):
    pass

#
# FUNCTION
#


def analyse(file_name, im_acq_time=None, experiment_directory=None,
            paths=None, fixture_name=None,
            fixture_directory=None):

    if im_acq_time is None:
        im_acq_time = os.stat(file_name).st_mtime

    im_data = {'Time': im_acq_time, 'File': file_name}

    _logger.info("Fixture init for {0}".format(file_name))

    if fixture_name is None:
        fixture_name = paths.experiment_local_fixturename

    if fixture_directory is None:
        fixture_directory = experiment_directory

    if experiment_directory is not None:
        file_name = os.path.join(experiment_directory,
                                 os.path.basename(file_name))

        _logger.info(
            "File path changed to match experiment directory: {0}".format(
                file_name))

    fixture = resource_fixture_image.Fixture_Image(
        fixture_name,
        fixture_directory=fixture_directory,
        image_path=file_name)

    #logger.info("Fixture set for {0}".format(file_name))

    #fixture.set_image(image_path=file_name)

    _logger.info("Image loaded for fixture {0}".format(file_name))

    im_data['im_shape'] = fixture['image'].shape

    _logger.info("Image has shape {0}".format(im_data['im_shape']))

    fixture.run_marker_analysis()

    _logger.info("Marker analysis run".format(file_name))

    im_data['mark_X'], im_data['mark_Y'] = fixture['markers']

    if im_data['mark_X'] is None:
        raise Marker_Detection_Failed()
        return None

    fixture.set_current_areas()

    _logger.info("Setting current image areas for {0}".format(file_name))

    im_data['scale'] = fixture['scale']
    fixture.analyse_grayscale()

    _logger.info("Grayscale analysed for {0}".format(file_name))

    gsTarget = fixture['grayscaleTarget']
    gsSource = fixture['grayscaleSource']

    if gsTarget is None:
        _logger.error("Grayscale not properly set up (used {0})".format(
            fixture['grayscale_type']))
    if gsSource is None:
        _logger.error("Grayscale analysis failed (used {0})".format(
            fixture['grayscale_type']))

    im_data['grayscale_values'] = gsSource
    im_data['grayscale_indices'] = gsTarget

    sections_areas = fixture['plates']

    im_data['plates'] = len(sections_areas)

    plate_str = "plate_{0}_area"
    for i, a in enumerate(sections_areas):
        im_data[plate_str.format(i)] = list(a)

    _logger.info("First pass analysis done for {0}".format(file_name))

    return im_data
