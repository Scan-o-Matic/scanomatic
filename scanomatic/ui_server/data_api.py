from flask import request, Flask, jsonify, send_from_directory
from types import ListType, DictType
import numpy as np
import os
import shutil
from enum import Enum

from scanomatic.data_processing import phenotyper
from scanomatic.image_analysis.grayscale import getGrayscales, getGrayscale
from scanomatic.io.paths import Paths
from scanomatic.image_analysis.support import save_image_as_png
from scanomatic.image_analysis.image_grayscale import get_grayscale
from scanomatic.io.logger import Logger
from scanomatic.models.factories.fixture_factories import FixtureFactory
from scanomatic.models.fixture_models import GrayScaleAreaModel
from scanomatic.image_analysis.image_basics import Image_Transpose
from scanomatic.image_analysis.grid_cell import GridCell
from scanomatic.image_analysis.grid_array import get_calibration_polynomial_coeffs
from scanomatic.models.analysis_model import COMPARTMENTS
from scanomatic.models.factories.analysis_factories import AnalysisFeaturesFactory

from .general import get_fixture_image_by_name, usable_markers, split_areas_into_grayscale_and_plates, \
    get_area_too_large_for_grayscale, get_grayscale_is_valid, usable_plates, image_is_allowed, \
    get_fixture_image


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
    elif isinstance(data, ListType):
        return [json_data(d) for d in data]
    elif isinstance(data, DictType):
        return {json_data(k): json_data(data[k]) for k in data}
    elif hasattr(data, "tolist"):
        return data.tolist()
    elif isinstance(data, Enum):
        return data.name
    else:
        return data


def add_routes(app, rpc_client, is_debug_mode):

    """

    Args:
        app (Flask): The flask app to decorate
        rpc_client (scanomatic.io.rpc_client._ClientProxy): A dynamic rpc-client bridge
        is_debug_mode (bool): If running in debug-mode
    """

    @app.route("/api/data/phenotype", methods=['POST', 'GET'])
    @app.route("/api/data/phenotype/", methods=['POST', 'GET'])
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
                It is possible to send a single curve in a 1-d array or arrays with
                2 or 3 dimensions. These will be reshaped with new dimensions added
                outside of the existing dimensions until i has 4 dimensions.

            "times_data", a time vector with the same length as the inner-most
                dimension of "raw_growth_data". Time should be given in hours.

        Optional keys:

            "smooth_growth_data" must match in shape with "raw_growth_data" and
                will trigger skipping the default curve smoothing.
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

        raw_growth_data = request.json.get("raw_growth_data", [])
        times_data = request.json.get("times_data", [])
        settings = request.json.get("settings", {})
        smooth_growth_data = request.json.get("smooth_growth_data", [])
        inclusion_level = request.json.get("inclusion_level", "Trusted")
        normalize = request.json.get("normalize", False)
        reference_offset = request.json.get("reference_offset", "LowerRight")
        raw_growth_data = _validate_depth(raw_growth_data)

        state = phenotyper.Phenotyper(
            np.array(raw_growth_data), np.array(times_data), run_extraction=False, **settings)

        if smooth_growth_data:
            smooth_growth_data = _validate_depth(smooth_growth_data)
            state.set("smooth_growth_data", np.array(smooth_growth_data))

        state.set_phenotype_inclusion_level(phenotyper.PhenotypeDataType[inclusion_level])
        state.extract_phenotypes(resmoothen=len(smooth_growth_data) == 0)

        curve_segments = state.curve_segments

        if normalize:
            state.set_control_surface_offsets(phenotyper.Offsets[reference_offset])

            return jsonify(smooth_growth_data=json_data(state.smooth_growth_data),
                           phenotypes={
                               pheno.name: [None if p is None else p.tojson() for p in state.get_phenotype(pheno)]
                               for pheno in state.phenotypes},
                           phenotypes_normed={
                               pheno.name: [p.tojson() for p in state.get_phenotype(pheno, normalized=True)]
                               for pheno in state.phenotypes_that_normalize},
                           curve_phases=json_data(curve_segments))

        return jsonify(smooth_growth_data=json_data(state.smooth_growth_data),
                       phenotypes={
                           pheno.name: [None if p is None else p.tojson() for p in state.get_phenotype(pheno)]
                           for pheno in state.phenotypes},
                       curve_phases=json_data(curve_segments))

    @app.route("/api/data/grayscales", methods=['post', 'get'])
    @app.route("/api/data/grayscales/", methods=['post', 'get'])
    def _grayscales():

        return jsonify(grayscales=getGrayscales())

    @app.route("/api/data/grayscale/<fixture_name>", methods=['post', 'get'])
    def _gs_get(fixture_name):

        grayscale_area_model = GrayScaleAreaModel(
            name=request.args.get("grayscale_name", "", type=str),
            x1=request.values.get("x1", type=float),
            x2=request.values.get("x2", type=float),
            y1=request.values.get("y1", type=float),
            y2=request.values.get("y2", type=float))

        if get_area_too_large_for_grayscale(grayscale_area_model):

            return jsonify(source_values=None, target_values=None, grayscale=False,
                           reason="Area too large")

        _logger.info("Grayscale area to be tested {0}".format(dict(**grayscale_area_model)))

        fixture = get_fixture_image_by_name(fixture_name)
        _, values = get_grayscale(fixture, grayscale_area_model, debug=is_debug_mode)
        grayscale_object = getGrayscale(grayscale_area_model.name)
        valid = get_grayscale_is_valid(values, grayscale_object)

        return jsonify(source_values=values, target_values=grayscale_object['targets'],
                       grayscale=valid, reason=None if valid else "No Grayscale")

    @app.route("/api/data/fixture/names")
    @app.route("/api/data/fixture/names/")
    def _fixure_names():

        if rpc_client.online:
            return jsonify(fixtures=rpc_client.get_fixtures(), success=True)
        else:
            return jsonify(fixtures=[], success=False, reason="Scan-o-Matic server offline")

    @app.route("/api/data/fixture/get/<name>")
    def _fixture_data(name=None):
        if not rpc_client.online:
            return jsonify(success=False, reason="Scan-o-Matic server offline")
        elif name in rpc_client.get_fixtures():
            path = Paths().get_fixture_path(name)
            try:
                fixture = FixtureFactory.serializer.load_first(path)
                return jsonify(
                    success=True, grayscale=dict(**fixture.grayscale),
                    plates=[dict(**plate) for plate in fixture.plates],
                    markers=zip(fixture.orientation_marks_x, fixture.orientation_marks_y))
            except IndexError:
                return jsonify(success=False, reason="Fixture without data")
        else:
            return jsonify(success=False, reason="Unknown fixture")

    @app.route("/api/data/fixture/remove/<name>")
    def _fixture_remove(name):

        name = Paths().get_fixture_name(name)
        known_fixtures = tuple(Paths().get_fixture_name(f) for f in rpc_client.get_fixtures())
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

        image = os.path.extsep.join(name, "png")
        _logger.info("Sending fixture image {0}".format(image))
        return send_from_directory(Paths().fixtures, image)

    @app.route("/api/data/fixture/set/<name>")
    def _fixture_set(name):

        if not rpc_client.online:
            return jsonify(success=False, reason="Scan-o-Matic server offline")

        areas = request.json.get("areas")
        markers = request.json.get("markers")
        grayscale_name = request.json.get("grayscale_name")

        name = Paths().get_fixture_name(name)
        if not name:
            return jsonify(success=False, reason="Fixtures need a name")

        _logger.info("Attempting to save {0} with areas {1} and markers {2}".format(name, areas, markers))

        try:
            fixture = get_fixture_image_by_name(name)
        except IOError:
            return jsonify(success=False, reason="Fixture image not on server")

        if not usable_markers(markers, fixture.im):
            return jsonify(success=False, reason="Bad markers")

        grayscale_area_model, plates = split_areas_into_grayscale_and_plates(areas)
        _logger.info("Grayscale {0}".format(grayscale_area_model))
        _logger.info("Plates".format(plates))

        if grayscale_area_model:

            if grayscale_name not in getGrayscales():
                return jsonify(success=False, reason="Unknown grayscale type")
            if get_area_too_large_for_grayscale(grayscale_area_model):
                return jsonify(success=False, reason="Area too large for grayscale")

            grayscale_area_model.name = grayscale_name
            _, values = get_grayscale(fixture, grayscale_area_model)
            grayscale_object = getGrayscale(grayscale_area_model.name)
            valid = get_grayscale_is_valid(values, grayscale_object)

            if not valid:
                return jsonify(success=False, reason="Could not detect grayscale")

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
            return jsonify(success=False, reason="Final compilation doesn't validate")

        FixtureFactory.serializer.dump(fixture_model, fixture_model.path)
        return jsonify(success=True)

    @app.route("/api/data/markers/detect/<fixture_name>")
    def _markers_detect(fixture_name):

        markers = request.values.get('markers', default=3, type=int)
        image = request.files.get('image')
        name = os.path.basename(fixture_name)
        image_name, ext = os.path.splitext(image.filename)
        _logger.info("Working on detecting marker for fixture {0} using image {1} ({2})".format(
            name, image.filename, image_is_allowed(ext)))

        if name and image_is_allowed(ext):

            fixture_file = Paths().get_fixture_path(name)

            path = os.path.extsep.join((fixture_file, ext.lstrip(os.path.extsep)))
            image.save(path)

            fixture = get_fixture_image(name, path)
            fixture.run_marker_analysis(markings=markers)

            save_image_as_png(path)

            return jsonify(markers=fixture['current'].get_marker_positions(),
                           image=os.path.basename(fixture_file))

        _logger.warning("Refused detection (keys files: {0} values: {1})".format(
            request.files.keys(), request.values.keys()))

        return jsonify(markers=[], image="")

    @app.route("/api/data/image/transform/grayscale", methods=['POST'])
    def image_transform_grayscale():

        image = np.array(request.json.get("image", [[]]))
        grayscale_values = np.array(request.json.get("grayscale_values", []))
        grayscale_targets = request.json.get("grayscale_targets", [])
        if not grayscale_targets:
            grayscale_targets = getGrayscale(request.json.get("grayscale_name", ""))['targets']

        transpose_polynomial = Image_Transpose(
            sourceValues=grayscale_values,
            targetValues=grayscale_targets)

        return jsonify(image=transpose_polynomial(image).tolist())

    @app.route("/api/data/image/detect/colony", methods=['POST'])
    def image_detect_colony():

        image = np.array(request.json.get("image", [[]]))
        identifier = ["unknown_image", 0, [0, 0]]  # first plate, upper left colony (just need something

        gc = GridCell(identifier, get_calibration_polynomial_coeffs(), save_extra_data=False)
        gc.source = image.astype(np.float64)
        gc.attach_analysis(
            blob=True, background=True, cell=True,
            run_detect=False)

        gc.detect(remember_filter=False)

        return jsonify(
            blob=gc.get_item(COMPARTMENTS.Blob).filter_array.tolist(),
            background=gc.get_item(COMPARTMENTS.Background).filter_array.tolist()
        )

    @app.route("/api/data/image/analyse/colony", methods=['POST'])
    def image_analyse_colony():

        image = np.array(request.json.get("image", [[]]))
        identifier = ["unknown_image", 0, [0, 0]]  # first plate, upper left colony (just need something

        gc = GridCell(identifier, get_calibration_polynomial_coeffs(), save_extra_data=False)
        gc.source = image.astype(np.float64)
        gc.attach_analysis(
            blob=True, background=True, cell=True,
            run_detect=False)

        gc.analyse(detect=True, remember_filter=False)

        return jsonify(
            blob=gc.get_item(COMPARTMENTS.Blob).filter_array.tolist(),
            background=gc.get_item(COMPARTMENTS.Background).filter_array.tolist(),
            features=json_data(AnalysisFeaturesFactory.deep_to_dict(gc.features))
        )

    @app.route("/api/data/image/transform/cells", methods=['POST'])
    def image_transform_cells():

        image = np.array(request.json.get("image", [[]]))
        background_filter = np.array(request.json.get("background_filter"))
        # TODO: Continue here to get cells per pixel
        # TODO: Then add way to calculate features based on manual filters and cell-image
        raise NotImplemented()

    # End of adding routes
