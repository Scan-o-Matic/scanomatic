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

def analyse(file_name, im_acq_time=None, experiment_directory=None,
        paths=None, logger=None, fixture_name=None, fixture_directory=None):

    analysis = dict()
    if logger is None:
        logger = resource_logger.Log_Garbage_Collector()
    else:
        logger = resource_logger.Logging_Log(logger)

    if im_acq_time is None:
        im_acq_time = os.stat(file_name).st_mtime

    im_data = {'Time':im_acq_time, 'File': file_name}

    logger.info("Fixture init for {0}".format(file_name))

    if fixture_name is None:
        fixture_name = paths.experiment_local_fixturename

    if fixture_directory is None:
        fixture_directory = experiment_directory

    fixture = resource_fixture_image.Fixture_Image(
            fixture_name,
            fixture_directory=fixture_directory,
            logger=logger, image_path=file_name)

    #logger.info("Fixture set for {0}".format(file_name))

    #fixture.set_image(image_path=file_name)

    logger.info("Image loaded for fixture {0}".format(file_name))

    im_data['im_shape'] = fixture['image'].shape

    logger.info("Image has shape {0}".format(im_data['im_shape']))

    fixture.run_marker_analysis(output_function=logger)

    logger.info("Marker analysis run".format(file_name))


    im_data['mark_X'], im_data['mark_Y'] = fixture['markers']            

    if im_data['mark_X'] is None:
        raise Marker_Detection_Failed()
        return None

    fixture.set_current_areas()

    logger.info("Setting current image areas for {0}".format(file_name))

    fixture.analyse_grayscale()

    logger.info("Grayscale analysed for {0}".format(file_name))

    gs_indices, gs_values = fixture['grayscale']
    
    im_data['grayscale_values'] = gs_values
    im_data['grayscale_indices'] = gs_indices

    sections_areas = fixture['plates']

    im_data['plates'] = len(sections_areas)

    plate_str = "plate_{0}_area"
    for i, a in enumerate(sections_areas):
        im_data[plate_str.format(i)] = list(a)

    logger.info("First pass analysis done for {0}".format(file_name))

    return im_data

