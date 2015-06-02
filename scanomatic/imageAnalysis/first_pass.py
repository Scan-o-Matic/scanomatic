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
from scanomatic.models.factories.fixture_factories import FixtureFactory, FixturePlateFactory
from scanomatic.models.factories.compile_project_factory import CompileImageAnalysisFactory
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


def analyse(compile_image_model, fixture_settings):
    """


    :type fixture_settings: scanomatic.io.fixtures.FixtureSettings
    :type compile_image_model: scanomatic.models.compile_project_model.CompileImageModel
    :rtype : scanomatic.models.compile_project_model.CompileImageAnalysisModel
    """

    compile_analysis_model = CompileImageAnalysisFactory.create(
        image=compile_image_model, fixture=FixtureFactory.copy(fixture_settings.model))

    fixture_image = FixtureImage(fixture=fixture_settings)

    _do_image_preparation(compile_analysis_model, fixture_image)

    _do_markers(compile_analysis_model, fixture_image)

    _logger.info("Setting current fixture_image areas for {0}".format(compile_image_model))

    fixture_image.set_current_areas()

    _do_grayscale(compile_analysis_model, fixture_image)

    _do_plates(compile_analysis_model, fixture_image)

    compile_analysis_model.coordinates_scale = fixture_image['scale']
    compile_analysis_model.shape = fixture_image['shape']

    _logger.info("First pass analysis done for {0}".format(compile_analysis_model))

    return compile_analysis_model


def _do_image_preparation(compile_analysis_model, image):
    """

    :type compile_analysis_model: scanomatic.models.compile_project_model.CompileImageAnalysisModel
    :type image: scanomatic.imageAnalysis.first_pass_image.FixtureImage
    """

    image['current'].model = compile_analysis_model.fixture
    image.set_image(image_path=compile_analysis_model.image.path)


def _do_markers(compile_analysis_model, image):

    """
    :type image: scanomatic.imageAnalysis.first_pass_image.FixtureImage
    :type compile_analysis_model: scanomatic.models.compile_project_model.CompileImageAnalysisModel
    """
    _logger.info("Running marker analysis on {0}".format(compile_analysis_model))

    image.run_marker_analysis()

    _logger.info("Marker analysis run".format(compile_analysis_model))

    compile_analysis_model.fixture.orientation_marks_x, \
        compile_analysis_model.fixture.orientation_marks_y = image['markers']

    if compile_analysis_model.fixture.orientation_marks_x is None:
        raise MarkerDetectionFailed()


def _do_grayscale(compile_analysis_model, image):

    """
    :type image: scanomatic.imageAnalysis.first_pass_image.FixtureImage
    :type compile_analysis_model: scanomatic.models.compile_project_model.CompileImageAnalysisModel
    """
    image.analyse_grayscale()

    _logger.info("Grayscale analysed for {0}".format(compile_analysis_model))

    compile_analysis_model.fixture.grayscale_targets = image['grayscaleTarget']
    compile_analysis_model.fixture.grayscale_values = image['grayscaleSource']

    if compile_analysis_model.fixture.grayscale_targets is None:
        _logger.error("Grayscale not properly set up (used {0})".format(
            image['grayscale_type']))
    if compile_analysis_model.fixture.grayscale_values is None:
        _logger.error("Grayscale analysis failed (used {0})".format(
            image['grayscale_type']))


def _do_plates(compile_analysis_model, image):

    """

    :type image: scanomatic.imageAnalysis.first_pass_image.FixtureImage
    :type compile_analysis_model: scanomatic.models.compile_project_model.CompileImageAnalysisModel
    """
    sections_areas = image['plates']

    for i, a in enumerate(sections_areas):
        compile_analysis_model.fixture.plates.append(FixturePlateFactory.create(index=i, **a))