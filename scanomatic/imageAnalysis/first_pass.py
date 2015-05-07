#!/usr/bin/env python
"""Resource module for first pass analysis."""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
from scanomatic.models.factories.analysis_factories import FixturePlateFactory
from scanomatic.models.factories.fixture_factories import FixtureFactory, FixturePlateFactory
from scanomatic.imageAnalysis.first_pass_image import FixtureImage


#
# GLOBALS
#

_logger = logger.Logger("1st Pass Analysis")

#
# EXCEPTIONS
#


class MarkerDetectionFailed(Exception):
    pass

#
# FUNCTION
#


def analyse(compile_image_model, fixture):
    """


    :type fixture: scanomatic.io.fixtures.FixtureSettings
    :type compile_image_model: scanomatic.models.compile_project_model.CompileImageModel
    :rtype : scanomatic.models.fixture_models.FixtureModel
    """

    image_model = FixtureFactory.create(index=compile_image_model.index,
                                              time=compile_image_model.time_stamp,
                                              path=compile_image_model.path)

    image = FixtureImage(fixture=fixture, image_path=image_model.path)

    _do_markers(image_model, image)

    _logger.info("Setting current image areas for {0}".format(compile_image_model.path))

    image.set_current_areas()

    _do_grayscale(image_model, image)

    _do_plates(image_model, image)

    image_model.coordinates_scale = image['scale']
    image_model.shape = image['shape']

    _logger.info("First pass analysis done for {0}".format(image_model.path))

    return image_model


def _do_markers(image_model, image):

    """
    :type image: scanomatic.imageAnalysis.first_pass_image.FixtureImage
    :type image_model: scanomatic.models.fixture_models.FixtureModel
    """
    _logger.info("Running marker analysis on {0}".format(image_model.path))

    image.run_marker_analysis()

    _logger.info("Marker analysis run".format(image_model.path))

    image_model.orientation_marks_x, image_model.orientation_marks_y = image['markers']

    if image_model.orientation_marks_x is None:
        raise MarkerDetectionFailed()


def _do_grayscale(image_model, image):

    """
    :type image: scanomatic.imageAnalysis.first_pass_image.FixtureImage
    :type image_model: scanomatic.models.fixture_models.FixtureModel
    """
    image.analyse_grayscale()

    _logger.info("Grayscale analysed for {0}".format(image_model.path))

    image_model.grayscale_targets = image['grayscaleTarget']
    image_model.grayscale_values = image['grayscaleSource']

    if image_model.grayscale_targets is None:
        _logger.error("Grayscale not properly set up (used {0})".format(
            image['grayscale_type']))
    if image_model.grayscale_values is None:
        _logger.error("Grayscale analysis failed (used {0})".format(
            image['grayscale_type']))


def _do_plates(image_model, image):

    """

    :type image: scanomatic.imageAnalysis.first_pass_image.FixtureImage
    :type image_model: scanomatic.models.fixture_models.FixtureModel
    """
    sections_areas = image['plates']

    for i, a in enumerate(sections_areas):
        image_model.plates.append(FixturePlateFactory.create(index=i, **a))