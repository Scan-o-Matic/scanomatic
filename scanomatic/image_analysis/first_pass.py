import scanomatic.io.logger as logger
from scanomatic.models.factories.fixture_factories import FixtureFactory
from scanomatic.models.factories.compile_project_factory import CompileImageAnalysisFactory
from scanomatic.image_analysis.first_pass_image import FixtureImage


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


def analyse(compile_image_model, fixture_settings, issues):
    """
    :type fixture_settings: scanomatic.io.fixtures.FixtureSettings
    :type compile_image_model: scanomatic.models.compile_project_model.CompileImageModel
    :type issues: dict
    :rtype : scanomatic.models.compile_project_model.CompileImageAnalysisModel
    """

    compile_analysis_model = CompileImageAnalysisFactory.create(
        image=compile_image_model, fixture=FixtureFactory.copy(fixture_settings.model))

    fixture_image = FixtureImage(fixture=fixture_settings)

    _do_image_preparation(compile_analysis_model, fixture_image)

    _do_markers(compile_analysis_model, fixture_image)

    _logger.info("Setting current fixture_image areas for {0}".format(compile_image_model))

    fixture_image.set_current_areas(issues)

    _do_grayscale(compile_analysis_model, fixture_image)

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

    if compile_analysis_model.fixture.orientation_marks_x is None:
        raise MarkerDetectionFailed()


def _do_grayscale(compile_analysis_model, image):

    """
    :type image: scanomatic.imageAnalysis.first_pass_image.FixtureImage
    :type compile_analysis_model: scanomatic.models.compile_project_model.CompileImageAnalysisModel
    """
    image.analyse_grayscale()

    _logger.info("Grayscale analysed for {0}".format(compile_analysis_model))

    if compile_analysis_model.fixture.grayscale.values is None:
        _logger.error("Grayscale not properly set up (used {0})".format(compile_analysis_model.fixture.grayscale.name))