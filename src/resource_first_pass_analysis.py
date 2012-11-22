#!/usr/bin/env python
"""Resource module for first pass analysis."""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.997"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os

#
# INTERNAL DEPENDENCIES
#

import src.resource_logger as resource_logger
import src.resource_fixture_image as resource_fixture_image
import src.resource_path as resource_path

#
# EXCEPTIONS
#

class Marker_Detection_Failed(Exception): pass

#
# FUNCTION
#

def analyse(file_name, im_acq_time, experiment_directory,
        paths, logger=None):

    analysis = dict()
    if logger is None:
        logger = resource_logger.Log_Garbage_Collector()

    im_data = {'Time':im_acq_time, 'File': file_name}

    fixture = resource_fixture_image.Fixture_Image(
            paths.experiment_local_fixturename,
            fixture_directory=experiment_directory)

    fixture.set_image(image_path=file_name)

    fixture.run_marker_analysis()

    im_data['mark_X'], im_data['mark_Y'] = fixture['markers']            

    if im_data['mark_X'] is None:
        raise Marker_Detection_Failed()
        return None

    fixture.set_current_areas()
    fixture.analyse_grayscale()

    gs_indices, gs_values = fixture['grayscale']
    
    im_data['grayscale_values'] = gs_values
    im_data['grayscale_indices'] = gs_indices

    sections_areas = fixture['plates']

    im_data['plates'] = len(sections_areas)

    plate_str = "plate_{0}_area"
    for i, a in enumerate(sections_areas):
        im_data[plate_str.format(i)] = list(a)

    return im_data

