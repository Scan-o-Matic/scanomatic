from flask import request, Flask, jsonify, send_from_directory, abort
from types import ListType, DictType, StringTypes
import numpy as np
import os
import glob
import shutil
from enum import Enum
from ConfigParser import Error as ConfigError

from scanomatic.data_processing import phenotyper
from scanomatic.data_processing.calibration import (
    get_polynomial_coefficients_from_ccc)

from scanomatic.io.paths import Paths
from scanomatic.io.logger import Logger
from scanomatic.io.fixtures import Fixtures

from scanomatic.image_analysis.support import save_image_as_png
from scanomatic.image_analysis.image_grayscale import get_grayscale
from scanomatic.image_analysis.image_basics import Image_Transpose
from scanomatic.image_analysis.grid_cell import GridCell
from scanomatic.image_analysis.grayscale import getGrayscales, getGrayscale
from scanomatic.image_analysis.first_pass_image import FixtureImage

from scanomatic.models.factories.fixture_factories import FixtureFactory
from scanomatic.models.fixture_models import GrayScaleAreaModel
from scanomatic.models.analysis_model import COMPARTMENTS, VALUES
from scanomatic.models.factories.analysis_factories import (
    AnalysisFeaturesFactory)

from .general import (
    get_fixture_image_by_name,
    usable_markers,
    split_areas_into_grayscale_and_plates,
    get_area_too_large_for_grayscale,
    get_grayscale_is_valid,
    usable_plates,
    image_is_allowed,
    get_fixture_image,
    convert_url_to_path,
    get_fixture_image_from_data,
    get_2d_list,
    string_parse_2d_list,
    get_image_data_as_array,
    json_abort,
    convert_path_to_url,
    get_search_results,
)


_logger = Logger("Data API")


def _depth(arr, lvl=1):

    if isinstance(arr, ListType) and len(arr) and isinstance(arr[0], ListType):
        _depth(arr[0], lvl + 1)
    else:
        return lvl


def _validate_depth(data):

    depth = _depth(data)

    while depth < 4:
        data = [data]
        depth += 1

    return data


def json_data(data):

    if data is None:
        return None
    elif hasattr(data, "tolist"):
        return json_data(data.tolist())
    elif isinstance(data, ListType):
        return [json_data(d) for d in data]
    elif isinstance(data, DictType):
        return {json_data(k): json_data(data[k]) for k in data}
    elif isinstance(data, Enum):
        return data.name
    else:
        return data


def add_routes(app, rpc_client, is_debug_mode):

    """

    Args:
        app (Flask): The flask app to decorate
        rpc_client (scanomatic.io.rpc_client._ClientProxy):
            A dynamic rpc-client bridge
        is_debug_mode (bool): If running in debug-mode
    """

    @app.route("/api/data/phenotype", methods=['POST'])
    def data_phenotype():
        """Takes growth data extracts phenotypes and normalizes.

        _NOTE_: This API only accepts json-formatted POST calls.

        Minimal example of request json:
        ```
            {"raw_growth_data": [1.23, 1.52 ... 4.24],
             "times_data": [0, 0.33, 0.67 ... 72]}
        ```

        Mandatory keys:

            "raw_growth_data", a 4-dimensional array where the outer dimension
                represents plates, the two middle represents a plate layout and
                the fourth represents the data vector/time dimension.
                It is possible to send a single log2_curve in a 1-d array or arrays with
                2 or 3 dimensions. These will be reshaped with new dimensions added
                outside of the existing dimensions until i has 4 dimensions.
                _NOTE_: The values should be on linear scale, not log2 transformed
                    or similar. This will be done internally by the feature
                    extraction.

            "times_data", a time vector with the same length as the inner-most
                dimension of "raw_growth_data". Time should be given in hours.

        Optional keys:

            "smooth_growth_data" must match in shape with "raw_growth_data" and
                will trigger skipping the default log2_curve smoothing.
            "inclusion_level" (Trusted, UnderDevelopment, Other) default is
                Trusted.
            "normalize", (True, False), default is False
            "settings" a key: value type array/dictionary which can contain
                the following keys:

                    "median_kernel_size", odd value, default 5
                    "gaussian_filter_sigma", float, default 1.5
                    "linear_regression_size", odd value, default 5

            "reference_offset" str, is optional and only active if "normalize" is
                set to true. Any of the following (UpperLeft, UpperRight,
                LowerLeft, LowerRight).

        :return: json-object with analyzed data
        """

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        if len(data_object) == 0:
            return jsonify(
                success=False, reason="No valid json or post is empty")
        else:
            raw_growth_data = data_object.get("raw_growth_data", [])
            times_data = data_object.get("times_data", [])
            settings = data_object.get("settings", {})
            smooth_growth_data = data_object.get("smooth_growth_data", [])
            inclusion_level = data_object.get("inclusion_level", "Trusted")
            normalize = data_object.get("normalize", False)
            reference_offset = data_object.get(
                "reference_offset", "LowerRight")

        raw_growth_data = _validate_depth(raw_growth_data)

        state = phenotyper.Phenotyper(
            np.array(raw_growth_data),
            np.array(times_data),
            run_extraction=False,
            **settings)

        if smooth_growth_data:
            smooth_growth_data = _validate_depth(smooth_growth_data)
            state.set("smooth_growth_data", np.array(smooth_growth_data))

        state.set_phenotype_inclusion_level(
            phenotyper.PhenotypeDataType[inclusion_level])
        state.extract_phenotypes(resmoothen=len(smooth_growth_data) == 0)

        curve_segments = state.curve_segments

        if normalize:
            state.set_control_surface_offsets(
                phenotyper.Offsets[reference_offset])

            return jsonify(
                success=True,
                smooth_growth_data=json_data(state.smooth_growth_data),
                phenotypes={
                    pheno.name: [
                        None if p is None else p.tojson() for p in
                        state.get_phenotype(pheno)]
                    for pheno in state.phenotypes},
                phenotypes_normed={
                    pheno.name: [
                        p.tojson() for p in state.get_phenotype(
                            pheno,
                            norm_state=phenotyper.NormState.NormalizedRelative)
                    ] for pheno in state.phenotypes_that_normalize},
                curve_phases=json_data(curve_segments))

        return jsonify(
            smooth_growth_data=json_data(state.smooth_growth_data),
            phenotypes={
                pheno.name: [
                    None if p is None else p.tojson() for p in
                    state.get_phenotype(pheno)] for pheno in state.phenotypes},
            curve_phases=json_data(curve_segments))

    @app.route("/api/data/grayscales", methods=['post', 'get'])
    def _grayscales():
        """The known grayscale names

        Returns: json-object with key 'garyscales' having an array of strings.

        """
        grayscales = getGrayscales()

        # TODO: This should be part of app_config really
        if 'SilverFast' in grayscales:
            default = 'SilverFast'
        else:
            default = None

        return jsonify(success=True, grayscales=grayscales, default=default)

    @app.route("/api/data/grayscale/image/<grayscale_name>", methods=['POST'])
    def _gs_get_from_image(grayscale_name):
        """Analyse image slice as grayscale-strip

        The image should be supplied via POST, preferably in a
        json-object, under the key 'image'

        Args:
            grayscale_name: The type of strip, known by default
                ['Kodak', 'SilverFast']

        Returns: json-object with keys
            'source_values' as an array of each strip segment's
                average value in the image
            'target_values' as an array of the reference values
                supplied by the manufacturer
            'grayscale' as the name of the grayscale
            'success' if analysis completed and if not 'reason'
                is added as to why not.
            'exits' lists the suggested further requests
            'transform_grayscale' list with the uri to transform image
                in grayscale calibrated space using the results
                of the current request

        See Also:
            _grayscales @ route /api/data/grayscales:
                Getting the names of the known grayscales
        """
        raise NotImplemented("This endpoint is not complete")

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        image = get_image_data_as_array(
            data_object.get("image", default=[[]]),
            reshape=data_object.get("shape", default=None))

        grayscale_area_model = GrayScaleAreaModel(
            name=grayscale_name,
            x1=0, x2=image.shape[1],
            y1=0, y2=image.shape[0])

        if get_area_too_large_for_grayscale(grayscale_area_model):

            return jsonify(
                success=False, source_values=None, target_values=None,
                grayscale=None, reason="Area too large")

        try:
            _, values = get_grayscale(
                fixture, grayscale_area_model, debug=is_debug_mode)
        except TypeError:
            return jsonify(
                success=False, is_endpoint=True,
                reason="Grayscale detection failed")

        grayscale_object = getGrayscale(grayscale_area_model.name)
        valid = get_grayscale_is_valid(values, grayscale_object)

        return jsonify(
            success=valid,
            source_values=values,
            target_values=grayscale_object['targets'],
            grayscale=grayscale_area_model.name,
            reason=(
                None if valid else
                "No valid grayscale detected for {0}".format(
                    dict(**grayscale_area_model))),
            exits=["transform_grayscale"],
            transform_grayscale=["/api/data/image/transform/grayscale"])

    @app.route(
        "/api/data/grayscale/fixture/<fixture_name>", methods=['POST', 'GET'])
    def _gs_get_from_fixture(fixture_name):
        """Get grayscale analysis based on fixture image

        _NOTE_: This URI does not support json-objects posted

        Args:
            fixture_name: Name of the fixture

        Returns: json-object with keys
            'source_values' as an array of each strip segment's average value in the image
            'target_values' as an array of the reference values supplied by the manufacturer
            'grayscale' as the name of the grayscale
            'success' if analysis completed and if not 'reason' is added as to why not.
        """

        grayscale_area_model = GrayScaleAreaModel(
            name=request.args.get("grayscale_name", "", type=str),
            x1=request.values.get("x1", type=float),
            x2=request.values.get("x2", type=float),
            y1=request.values.get("y1", type=float),
            y2=request.values.get("y2", type=float))

        if get_area_too_large_for_grayscale(grayscale_area_model):

            return jsonify(
                success=True, source_values=None, target_values=None,
                grayscale=False, reason="Area too large")

        _logger.info("Grayscale area to be tested {0}".format(
            dict(**grayscale_area_model)))

        fixture = get_fixture_image_by_name(fixture_name)

        try:
            _, values = get_grayscale(
                fixture, grayscale_area_model, debug=is_debug_mode)
        except TypeError:
            return jsonify(
                success=False, is_endpoint=True,
                reason="Grayscale detection failed")

        grayscale_object = getGrayscale(grayscale_area_model.name)
        valid = get_grayscale_is_valid(values, grayscale_object)

        return jsonify(
            success=True,
            source_values=values,
            target_values=grayscale_object['targets'],
            grayscale=valid,
            reason=None if valid else "No Grayscale")

    @app.route("/api/data/fixture/names")
    def _fixure_names():
        """Names of fixtures

        Returns: json-object with keys
            "fixtures" as the an array of strings, the names
            "success" if could be obtained and if not "reason" to explain why.

        """

        if rpc_client.online:
            return jsonify(fixtures=rpc_client.get_fixtures(), success=True)
        else:
            return jsonify(
                fixtures=[],
                success=False,
                reason="Scan-o-Matic server offline")

    @app.route("/api/data/fixture/local/<path:project>")
    @app.route("/api/data/fixture/local")
    def _fixture_local_data(project=""):

        path = os.path.join(
            convert_url_to_path(project),
            Paths().experiment_local_fixturename)

        try:
            fixture = FixtureFactory.serializer.load_first(path)
            if fixture is None:
                return jsonify(
                    success=False,
                    reason="File is missing")
            return jsonify(
                success=True, grayscale=dict(**fixture.grayscale),
                plates=[dict(**plate) for plate in fixture.plates],
                markers=zip(
                    fixture.orientation_marks_x,
                    fixture.orientation_marks_y))
        except IndexError:
            return jsonify(success=False, reason="Fixture without data")
        except ConfigError:
            return jsonify(success=False, reason="Fixture data corrupted")

    @app.route("/api/data/fixture/get/<name>")
    def _fixture_data(name=None):
        """Get the specifications of a fixture

        Args:
            name: The name of the fixture

        Returns: json-object where keys:
            "plates" is an array of key-value arrays of the
                included plates specs
            "grayscale" is a key-value array of its specs
            "markers" is a 2D array of the marker centra
            "success" if the fixture was found and valid else
            "reason" to explain why not.

        """
        if not rpc_client.online:
            return jsonify(success=False, reason="Scan-o-Matic server offline")
        elif name in rpc_client.get_fixtures():
            path = Paths().get_fixture_path(name)
            try:
                fixture = FixtureFactory.serializer.load_first(path)
                if fixture is None:
                    return jsonify(
                        success=False,
                        reason="File is missing"
                    )
                return jsonify(
                    success=True, grayscale=dict(**fixture.grayscale),
                    plates=[dict(**plate) for plate in fixture.plates],
                    markers=zip(
                        fixture.orientation_marks_x,
                        fixture.orientation_marks_y))
            except IndexError:
                return jsonify(success=False, reason="Fixture without data")
            except ConfigError:
                return jsonify(success=False, reason="Fixture data corrupted")
        else:
            return jsonify(success=False, reason="Unknown fixture")

    @app.route("/api/data/fixture/remove/<name>")
    def _fixture_remove(name):
        """Remove a fixture by name

        Args:
            name: The name of the fixture to remove

        Returns: json-object with keys
            "success" if removed else
            "reason" to explain why not.

        """
        name = Paths().get_fixture_name(name)
        known_fixtures = tuple(
            Paths().get_fixture_name(f) for f in rpc_client.get_fixtures())
        if name not in known_fixtures:
            return jsonify(success=False, reason="Unknown fixture")
        source = Paths().get_fixture_path(name)
        path, ext = os.path.splitext(source)
        i = 0
        pattern = "{0}.deleted{1}"
        while os.path.isfile(pattern.format(path, i)):
            i += 1
        try:
            shutil.move(source, pattern.format(path, i))
        except IOError:
            return jsonify(success=False, reason="Error while removing")
        return jsonify(success=True, reason="Happy")

    @app.route("/api/data/fixture/image/get/<name>")
    def _fixture_get_image(name):
        """Get downscaled png image for the fixture.

        Args:
            name: Name of the fixture

        Returns: image

        """
        image = os.path.extsep.join((name, "png"))
        _logger.info("Sending fixture image {0}".format(image))
        return send_from_directory(Paths().fixtures, image)

    @app.route("/api/data/fixture/set/<name>", methods=["POST"])
    def _fixture_set(name):

        if not rpc_client.online:
            return jsonify(success=False, reason="Scan-o-Matic server offline")

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        if len(data_object) == 0:
            return jsonify(
                success=False, reason="No valid json or post is empty")

        areas = data_object.get("areas")
        markers = data_object.get("markers")
        grayscale_name = data_object.get("grayscale_name")

        name = Paths().get_fixture_name(name)
        if not name:
            return jsonify(success=False, reason="Fixtures need a name")

        _logger.info(
            "Attempting to save {0} with areas {1} and markers {2}".format(
                name, areas, markers))

        try:
            fixture = get_fixture_image_by_name(name)
        except IOError:
            return jsonify(success=False, reason="Fixture image not on server")

        if not usable_markers(markers, fixture.im):
            return jsonify(success=False, reason="Bad markers")

        grayscale_area_model, plates = split_areas_into_grayscale_and_plates(
            areas)
        _logger.info("Grayscale {0}".format(grayscale_area_model))
        _logger.info("Plates".format(plates))

        if grayscale_area_model:

            if grayscale_name not in getGrayscales():
                return jsonify(success=False, reason="Unknown grayscale type")
            if get_area_too_large_for_grayscale(grayscale_area_model):
                return jsonify(
                    success=False, reason="Area too large for grayscale")

            grayscale_area_model.name = grayscale_name

            try:
                _, values = get_grayscale(fixture, grayscale_area_model)
            except TypeError:

                return jsonify(
                    success=False, reason="Could not detect grayscale")

            grayscale_object = getGrayscale(grayscale_area_model.name)
            valid = get_grayscale_is_valid(values, grayscale_object)

            if not valid:
                return jsonify(
                    success=False, reason="Could not detect valid grayscale")

            grayscale_area_model.values = values

        if not usable_plates(plates):
            return jsonify(success=False, reason="Bad plate selections")

        fixture_model = FixtureFactory.create(
            path=Paths().get_fixture_path(name),
            grayscale=grayscale_area_model,
            orientation_marks_x=tuple(mark[0] for mark in markers),
            orientation_marks_y=tuple(mark[1] for mark in markers),
            shape=fixture.im.shape,
            coordinates_scale=1.0,
            plates=plates,
            name=name,
            scale=1.0)

        if not FixtureFactory.validate(fixture_model):
            return jsonify(
                success=False, reason="Final compilation doesn't validate")

        FixtureFactory.serializer.dump(fixture_model, fixture_model.path)
        return jsonify(success=True)

    @app.route("/api/data/fixture/calculate/<fixture_name>", methods=['POST'])
    def _get_transposed_fixture_coordinates(fixture_name):

        image = get_image_data_as_array(
            request.files.get('image', default=np.array([])))

        markers = get_2d_list(request.values, 'markers')

        if not markers and isinstance(
                request.values.get('markers', default=None), StringTypes):

            _logger.warning(
                "Attempting fallback string parsing of markers as text")
            markers = string_parse_2d_list(request.values.get('markers', ""))

        if not markers:
            _logger.warning("Assuming markers have been sent as flat list")
            markers = request.values.get('markers', [])
            if len(markers) == 6:
                markers = [[float(m) for m in markers[:3]],
                           [float(m) for m in markers[3:]]]
                _logger.info("Markers successfully reshaped {0}".format(
                    markers))
            else:
                _logger.error("Unexpected number of markers {0} ({1})".format(
                    len(markers), markers))

        markers = np.array(markers)

        _logger.info("Using markers {0}".format(markers))

        if (markers.ndim != 2 and
                markers.shape[0] != 2 and
                markers.shape[1] < 3):
            return jsonify(
                success=False,
                reason="Markers should be a 2D array with shape (2, 3) or greater for last dimension",
                is_endpoint=True,
            )

        fixture_settings = Fixtures()[fixture_name]

        if fixture_settings is None:
            return jsonify(
                success=False,
                reason="Fixture '{0}' is not known".format(fixture_name),
                is_endpoint=True,
            )

        fixture = FixtureImage(fixture_settings)
        current_settings = fixture['current']
        current_settings.model.orientation_marks_x = markers[0]
        current_settings.model.orientation_marks_y = markers[1]
        issues = {}
        fixture.set_current_areas(issues)

        return jsonify(
            success=True,
            is_endpoint=True,
            plates=[
                dict(
                    index=plate.index,
                    x1=plate.x1,
                    x2=plate.x2,
                    y1=plate.y1,
                    y2=plate.y2,
                    data=(
                        None if image.size == 0 else
                        image[
                            plate.y1: plate.y2,
                            plate.x1: plate.y2].tolist()),
                    shape=[plate.y2 - plate.y1, plate.x2 - plate.x1]
                )
                for plate in current_settings.model.plates
            ],
            grayscale_area=dict(
                x1=current_settings.model.grayscale.x1,
                x2=current_settings.model.grayscale.x2,
                y1=current_settings.model.grayscale.y1,
                y2=current_settings.model.grayscale.y2,
                data=None if image.size == 0 else image[
                    current_settings.model.grayscale.y1:
                    current_settings.model.grayscale.y2,
                    current_settings.model.grayscale.x1:
                    current_settings.model.grayscale.y2].tolist(),

                shape=[
                    current_settings.model.grayscale.y2 -
                    current_settings.model.grayscale.y1,
                    current_settings.model.grayscale.x2 -
                    current_settings.model.grayscale.x1]
            ),
            grayscale_name=current_settings.model.grayscale.name,
            report=issues,
        )

    @app.route("/api/data/markers/detect/<fixture_name>", methods=['POST'])
    def _markers_detect(fixture_name):

        markers = request.values.get('markers', default=3, type=int)

        try:
            save_fixture = bool(request.values.get('save', default=1))
        except ValueError:
            save_fixture = True

        image = request.files.get('image')

        name = os.path.basename(fixture_name)
        image_name, ext = os.path.splitext(image.filename)
        _logger.info(
            "Working on detecting marker for fixture {0} using image {1} ({2})".format(
                name, image.filename, image_is_allowed(ext)))

        if name and image_is_allowed(ext):

            fixture_file = Paths().get_fixture_path(name)

            path = os.path.extsep.join(
                (fixture_file, ext.lstrip(os.path.extsep)))

            if save_fixture:
                image.save(path)

                fixture = get_fixture_image(name, path)
                fixture.run_marker_analysis(markings=markers)

                save_image_as_png(path)

                return jsonify(
                    success=True,
                    is_endpoint=True,
                    markers=json_data(
                        fixture['current'].get_marker_positions()),
                    image=os.path.basename(fixture_file))

            else:

                fixture = get_fixture_image_from_data(name, image)
                fixture.run_marker_analysis(markings=markers)

                return jsonify(
                    success=True,
                    is_endpoint=True,
                    markers=json_data(
                        fixture['current'].get_marker_positions())
                )

        _logger.warning(
            "Refused detection (keys files: {0} values: {1})".format(
                request.files.keys(), request.values.keys()))

        return jsonify(
            success=False,
            is_endpoint=True,
            reason="No fixture image name" if image_is_allowed(ext) else
            "Image type not allowed")

    @app.route("/api/data/image/transform/grayscale", methods=['POST'])
    def image_transform_grayscale():
        """Method to convert image to grayscale space.

        The request must be sent over POST and preferrably in json-format.

        Request keys:
            "image": the 2D array data of image to be converted,
                keep image no larger than needed as it takes some
                processing doing the conversion.
            "grayscale_values": the image values of grayscale segments
            "grayscale_targets": the manufacturer's target values.

        Returns: json-object with keys
            "success" if successfully converted
            "reason" why not
            "image" the converted image
            "exits" list keys with suggested further requests
            "detect_colony" list with the URI for detecting a colony in an image

        See Also:
            _gs_get_from_image @ route "/api/data/grayscale/image/<grayscale_name>":
                analysing grayscales in images.
        """
        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        image = get_image_data_as_array(
            data_object.get("image", default=[[]]),
            reshape=data_object.get("shape", default=None))

        grayscale_values = np.array(data_object.get("grayscale_values", []))
        grayscale_targets = np.array(data_object.get("grayscale_targets", []))

        if not grayscale_targets:
            grayscale_targets = getGrayscale(
                data_object.get("grayscale_name", ""))['targets']

        transpose_polynomial = Image_Transpose(
            sourceValues=grayscale_values,
            targetValues=grayscale_targets)

        return jsonify(
            success=True,
            image=transpose_polynomial(image).tolist(),
            exits=["detect_colony"],
            detect_colony=["/api/data/image/detect/colony"])

    @app.route("/api/data/image/detect/colony", methods=['POST'])
    def image_detect_colony():
        """Detect colony in image

        The request must be sent over POST and preferrably in
        json-format.

        Request keys:
            "image": the 2D array data of an image of grayscale
                values

        Returns: json-object
            "success" if successfully detected
            "reason" why not
            "blob" a 2D boolean array identifying pixels as blob
            "background" a 2D boolean array identifying pixels as
                background
            "exits" list keys with suggested further requests
            "transform_cells" list of the URI for transforming grayscale
                calibrated values to cells.
            "analyse_compartments": list of URI for analysing compartments
        """
        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        image = get_image_data_as_array(
            data_object.get("image", default=[[]]),
            reshape=data_object.get("shape", default=None))

        # first plate, upper left colony (just need something
        identifier = ["unknown_image", 0, [0, 0]]

        gc = GridCell(
            identifier,
            get_polynomial_coefficients_from_ccc('default'),
            save_extra_data=False)

        gc.source = image.astype(np.float64)
        gc.attach_analysis(
            blob=True, background=True, cell=True,
            run_detect=False)

        gc.detect(remember_filter=False)

        return jsonify(
            success=True,
            blob=gc.get_item(COMPARTMENTS.Blob).filter_array.tolist(),
            background=gc.get_item(
                COMPARTMENTS.Background).filter_array.tolist(),
            exits=['transform_cells', 'analyse_compartment'],
            transform_cells=['/api/data/image/transform/cells'],
            analyse_compartment=[
                '/api/data/image/analyse/compartment/{0}'.format(c) for c in
                COMPARTMENTS]
        )

    @app.route("/api/data/image/analyse/colony", methods=['POST'])
    def image_analyse_colony():
        """Automatically analyse image section

        The request must be sent over POST and preferrably in
        json-format.

        Request keys:
            "image": the 2D array data of an image of grayscale
                values

        Returns: json-object
            "success" if successfully detected
            "reason" why not
            "blob" a 2D boolean array identifying pixels as blob
            "background" a 2D boolean array identifying pixels as
                background
            "features" nested key-value array structure containing
                all analysis results.
        """
        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        image = get_image_data_as_array(
            data_object.get("image", default=[[]]),
            reshape=data_object.get("shape", default=None))

        # first plate, upper left colony (just need something
        identifier = ["unknown_image", 0, [0, 0]]

        gc = GridCell(
            identifier,
            get_polynomial_coefficients_from_ccc('default'),
            save_extra_data=False)

        gc.source = image.astype(np.float64)
        gc.attach_analysis(
            blob=True, background=True, cell=True,
            run_detect=False)

        gc.analyse(detect=True, remember_filter=False)

        return jsonify(
            success=True,
            blob=gc.get_item(COMPARTMENTS.Blob).filter_array.tolist(),
            background=gc.get_item(
                COMPARTMENTS.Background).filter_array.tolist(),
            features=json_data(
                AnalysisFeaturesFactory.deep_to_dict(gc.features))
        )

    @app.route("/api/data/image/transform/cells", methods=['POST'])
    def image_transform_cells():
        """Transform image values into cells per pixel

        The request must be sent over POST and preferrably in
        json-format.

        Request keys:
            "image": the 2D array data of an image of grayscale
                values
            "background_filter": the 2D array of matching size
                with boolean values indicating pixels being part of
                the background.

        Returns: json-object
            "success": if successfully detected
            "reason": why not
            "image": a 2D array of cells per pixel
            "exits": list of suggested further URI requests
            "analyse_compartment: list of request URIs to analyse
                detections.
        """

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        image = get_image_data_as_array(
            data_object.get("image", default=[[]]),
            reshape=data_object.get("shape", default=None))

        background_filter = np.array(data_object.get("background_filter"))

        # first plate, upper left colony (just need something
        identifier = ["unknown_image", 0, [0, 0]]

        gc = GridCell(identifier, None, save_extra_data=False)
        gc.source = image.astype(np.float64)
        gc.attach_analysis(
            blob=True, background=True, cell=True,
            run_detect=False)

        gc.set_new_data_source_space(
            space=VALUES.Cell_Estimates, bg_sub_source=background_filter,
            polynomial_coeffs=get_polynomial_coefficients_from_ccc('default'))

        return jsonify(
            success=True, image=gc.source.tolist(),
            exits=['analyse_compartment'],
            analyse_compartment=[
                '/api/data/image/analyse/compartment/{0}'.format(c) for c in
                COMPARTMENTS])

    @app.route("/api/data/image/analyse/compartment/<compartment>")
    def image_analyse_compartment(compartment):

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        image = get_image_data_as_array(
            data_object.get("image", default=[[]]),
            reshape=data_object.get("shape", default=None))

        filt = get_image_data_as_array(
            data_object.get("filter", default=[[]]),
            reshape=data_object.get("shape", default=None))

        # first plate, upper left colony (just need something
        identifier = ["unknown_image", 0, [0, 0]]

        gc = GridCell(identifier, None, save_extra_data=False)
        gc.attach_analysis(
            blob=True, background=True, cell=True,
            run_detect=False)

        if compartment == 'blob':
            gc_compartment = gc.get_item(COMPARTMENTS.Blob)
        elif compartment == 'background':
            gc_compartment = gc.get_item(COMPARTMENTS.Background)
        elif compartment == 'total':
            gc_compartment = gc.get_item(COMPARTMENTS.Total)
        else:
            return jsonify(
                success=False,
                reason="Unknown compartment {0}".format(compartment))

        gc_compartment.grid_array = image.astype(np.float64)
        gc_compartment.filter_array = filt.astype(np.int)

        gc_compartment.do_analysis()

        return jsonify(
            success=True,
            features=json_data(
                AnalysisFeaturesFactory.deep_to_dict(gc_compartment.features)))

    @app.route("/api/data/logs/<path:project>")
    def _project_logs_api(project):

        path = convert_url_to_path(project)
        if not os.path.exists(path):

            json_abort(400, reason='Invalid project')

        is_project_analysis = phenotyper.path_has_saved_project_state(path)

        if not os.path.isfile(path) or not path.endswith(".log"):

            if is_project_analysis:
                logs = glob.glob(
                    os.path.join(path, Paths().analysis_run_log))
                logs += glob.glob(
                    os.path.join(path, Paths().phenotypes_extraction_log))
            else:
                logs = glob.glob(os.path.join(
                    path, Paths().scan_log_file_pattern.format("*")))
                logs += glob.glob(os.path.join(
                    path, Paths().project_compilation_log_pattern.format("*")))

            return jsonify(
                is_project_analysis=is_project_analysis,
                exits=['urls', 'logs'],
                logs=[
                    convert_path_to_url("/logs/project", log_path)
                    for log_path in logs
                ],
                **get_search_results(path, "/api/data/logs"))

    # End of adding routes
