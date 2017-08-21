from string import letters
from itertools import product
from types import StringTypes

from flask import jsonify, request, send_file
import numpy as np

from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from scanomatic.models.analysis_model import COMPARTMENTS
from scanomatic.image_analysis.grid_cell import GridCell
from scanomatic.image_analysis.grid_array import GridArray
from scanomatic.image_analysis.grayscale import getGrayscale
from scanomatic.image_analysis.image_grayscale import (
    get_grayscale_image_analysis)
from scanomatic.image_analysis import image_basics
from scanomatic.io.paths import Paths
from scanomatic.io.fixtures import Fixtures
from scanomatic.data_processing import calibration
from scanomatic.data_processing.calibration import delete_ccc
from .general import (
    serve_numpy_as_image, get_grayscale_is_valid, valid_array_dimensions,
    json_abort
)

_VALID_CHARACTERS = letters + "-._1234567890"


def add_routes(app):
    """

    :param app: The flask webb app
     :type app: Flask
    :return:
    """

    @app.route("/api/calibration/active", methods=['GET'])
    def get_active_calibrations():

        try:
            identifiers, cccs = zip(*calibration.get_active_cccs().iteritems())
            return jsonify(
                success=True,
                is_endpoint=True,
                identifiers=identifiers,
                species=[
                    ccc[calibration.CellCountCalibration.species]
                    for ccc in cccs],
                references=[
                    ccc[calibration.CellCountCalibration.reference]
                    for ccc in cccs]
            )
        except ValueError:
            return json_abort(
                400,
                success=False,
                is_endpoint=True,
                reason="There are no registered CCC, Scan-o-Matic won't " +
                "work before at least one is added"
            )

    @app.route("/api/calibration/under_construction", methods=['GET'])
    def get_under_construction_calibrations():

        try:
            identifiers, cccs = zip(
                *calibration.get_under_construction_cccs().iteritems())
            return jsonify(
                success=True,
                is_endpoint=True,
                identifiers=identifiers,
                species=[
                    ccc[calibration.CellCountCalibration.species]
                    for ccc in cccs],
                references=[
                    ccc[calibration.CellCountCalibration.reference]
                    for ccc in cccs]
            )
        except ValueError:
            return json_abort(
                400,
                reason="No CCCs are under constructions"
            )

    @app.route("/api/calibration/initiate_new", methods=['POST'])
    def initiate_new_ccc():

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        species = data_object.get("species")
        reference = data_object.get("reference")
        ccc = calibration.get_empty_ccc(species, reference)
        if ccc is None:
            return json_abort(
                400,
                reason="Combination of species and reference not unique"
            )

        success = calibration.add_ccc(ccc)

        if not success:
            return json_abort(
                400,
                reason="Possibly someone just beat you to that combination " +
                "of species and reference!"
            )

        return jsonify(
            success=True,
            is_endpoint=True,
            identifier=ccc[calibration.CellCountCalibration.identifier],
            access_token=ccc[
                calibration.CellCountCalibration.edit_access_token]
        )

    @app.route("/api/calibration/<ccc_identifier>/add_image", methods=['POST'])
    def upload_ccc_image(ccc_identifier):

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        image = request.files.get('image', default=None)
        if image is None:
            return json_abort(
                400,
                reason="Didn't get any image"
            )

        image_identifier = calibration.add_image_to_ccc(
            ccc_identifier, image, access_token=data_object.get("access_token")
        )

        if not image_identifier:
            return json_abort(
                401,
                reason="Refused to save image, probably bad access token"
            )

        return jsonify(
            success=True,
            is_endpoint=True,
            image_identifier=image_identifier
        )

    @app.route("/api/calibration/<ccc_identifier>/image_list", methods=['GET'])
    def list_ccc_images(ccc_identifier):

        image_list = calibration.get_image_identifiers_in_ccc(ccc_identifier)
        if image_list is False:
            return json_abort(
                400,
                reason="No such ccc known"
            )

        return jsonify(
            success=True,
            is_endpoint=True,
            image_identifiers=image_list
        )

    @app.route(
        "/api/calibration/<ccc_identifier>/image/<image_identifier>/get",
        methods=['GET'])
    def download_ccc_image(ccc_identifier, image_identifier):

        im_path = Paths().ccc_image_pattern.format(
            ccc_identifier, image_identifier)
        return send_file(im_path, mimetype='Image/Tiff')

    @app.route(
        "/api/calibration/<ccc_identifier>/image/<image_identifier>/data/set",
        methods=['POST'])
    def set_ccc_image_data(ccc_identifier, image_identifier):
        """ Sets any of the allowed image data fields

        Most easy way to find out which these are is to look at
        /api/calibration/<ccc_identifier>/image/get/<image_identifier>/data

        Note: The request must contain an access_token

        Args:
            ccc_identifier: The ccc identifier
            image_identifier: The image identifier

        Returns:

        """
        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        data_update = {}
        for data_type in calibration.CCCImage:
            if data_type is calibration.CCCImage.identifier:
                continue
            val = data_object.get(data_type.name, None)
            if val:
                if data_type is calibration.CCCImage.marker_x or \
                        data_type is calibration.CCCImage.marker_y and \
                        isinstance(val, StringTypes):

                    try:
                        val = [float(v) for v in val.split(",")]
                    except ValueError:
                        app.logger.warning(
                            "The parameter " +
                            "{0} value '{1}' not understood".format(
                                data_type, val))
                        continue

                data_update[data_type.name] = val

                if data_type is calibration.CCCImage.fixture and \
                        calibration.CCCImage.grayscale_name.name not in \
                        data_object.keys():

                    fixture_settings = Fixtures()[val]
                    if fixture_settings is not None:
                        data_update[
                            calibration.CCCImage.grayscale_name.name] = \
                            fixture_settings.model.grayscale.name

        success = calibration.set_image_info(
            ccc_identifier,
            image_identifier,
            access_token=data_object.get("access_token"),
            **data_update
        )

        if not success:
            return json_abort(
                401,
                reason="Update refused, probably bad access token"
            )

        return jsonify(
            success=True,
            is_endpoint=True
        )

    @app.route(
        "/api/calibration/<ccc_identifier>/image/<image_identifier>/data/get",
        methods=['GET'])
    def get_ccc_image_data(ccc_identifier, image_identifier):

        data = calibration.get_image_json_from_ccc(
            ccc_identifier, image_identifier)
        if data is None:
            return json_abort(
                400,
                reason="The image or CCC don't exist"
            )

        return jsonify(
            success=True,
            is_endpoint=True,
            **{
                k.name: val for k, val in data.iteritems()
                if k is not calibration.CCCImage.plates}
        )

    @app.route(
        "/api/calibration/<ccc_identifier>/image/<image_identifier>/slice/set",
        methods=['POST'])
    def slice_ccc_image(ccc_identifier, image_identifier):

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        data = calibration.get_local_fixture_for_image(
            ccc_identifier, image_identifier)
        if data is None:
            return json_abort(
                400,
                reason="The image or CCC don't exist or not enough info set " +
                "to do slice"
            )

        success = calibration.save_image_slices(
            ccc_identifier, image_identifier,
            grayscale_slice=data["grayscale"],
            plate_slices=data['plates'],
            access_token=data_object.get("access_token"))

        if not success:
            return json_abort(
                401,
                reason="Probably not the correct access token."
            )

        return jsonify(
            success=True,
            is_endpoint=True
        )

    @app.route(
        "/api/calibration/<ccc_identifier>/image/<image_identifier>/slice" +
        "/get/<slice>", methods=['GET'])
    def get_ccc_image_slice(ccc_identifier, image_identifier, slice):
        """

        :param ccc_identifier:
        :param image_identifier:
        :param slice: either 'gs' for grayscale or the plate index 0-3
        :return:
        """
        if slice.lower() == 'gs':
            image = calibration.get_grayscale_slice(
                ccc_identifier, image_identifier)
        else:
            image = calibration.get_plate_slice(
                ccc_identifier, image_identifier, slice, gs_transformed=False)

        if image is None:
            return json_abort(
                400,
                reason="No such image slice exists, has it been sliced?"
            )
        return serve_numpy_as_image(image)

    @app.route(
        "/api/calibration/<ccc_identifier>/image/<image_identifier>" +
        "/grayscale/analyse", methods=['POST'])
    def get_ccc_image_grayscale_analysis(ccc_identifier, image_identifier):

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        gs_image = calibration.get_grayscale_slice(
            ccc_identifier, image_identifier)
        image_data = calibration.get_image_json_from_ccc(
            ccc_identifier, image_identifier)
        try:
            gs_name = image_data[calibration.CCCImage.grayscale_name]
        except TypeError:
            return json_abort(
                400,
                reason='Unknown image or CCC'
            )

        _, values = get_grayscale_image_analysis(
            gs_image, gs_name, debug=False)
        grayscale_object = getGrayscale(gs_name)
        valid = get_grayscale_is_valid(values, grayscale_object)
        if not valid:
            return json_abort(
                400,
                reason='Grayscale results are not valid'
            )

        success = calibration.set_image_info(
            ccc_identifier, image_identifier,
            grayscale_source_values=values,
            grayscale_target_values=grayscale_object['targets'],
            access_token=data_object.get("access_token")
        )

        if success:

            return jsonify(
                success=True,
                is_endpoint=True,
                source_values=values,
                target_values=grayscale_object['targets']
            )

        else:

            return json_abort(
                401,
                reason="Refused to set image grayscale info, probably bad " +
                "access token"
            )

    @app.route(
        "/api/calibration/<ccc_identifier>/image/<image_identifier>/plate" +
        "/<int:plate>/transform", methods=['POST'])
    def get_ccc_image_plate_transform(ccc_identifier, image_identifier, plate):

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        success = calibration.transform_plate_slice(
            ccc_identifier,
            image_identifier,
            plate,
            access_token=data_object.get("access_token")
        )
        if not success:
            return json_abort(
                401,
                reason="Probably bad access token, or not having sliced " +
                "image and analysed grayscale first"
            )

        return jsonify(
            success=True,
            is_endpoint=True
        )

    @app.route(
        "/api/calibration/<ccc_identifier>/image/<image_identifier>/plate/" +
        "<int:plate>/grid/set", methods=['POST'])
    def grid_ccc_image_plate(ccc_identifier, image_identifier, plate):

        def get_xy1_xy2(grid_array):
            outer, inner = grid_array.grid_shape

            xy1 = [[[None] for col in range(inner)] for row in range(outer)]
            xy2 = [[[None] for col in range(inner)] for row in range(outer)]
            warn_once = False
            for row, col in product(range(outer), range(inner)):
                grid_cell = grid_array[(row, col)]

                try:
                    xy1[row][col] = grid_cell.xy1.tolist()
                    xy2[row][col] = grid_cell.xy2.tolist()
                except AttributeError:
                    try:
                        xy1[row][col] = grid_cell.xy1
                        xy2[row][col] = grid_cell.xy2
                    except (TypeError, IndexError, ValueError):
                        if not warn_once:
                            warn_once = True
                            app.logger.error(
                                "Could not parse the xy corner data of grid " +
                                "cells, example '{0}' '{1}'".format(
                                    grid_cell.xy1, grid_cell.xy2)
                            )

            return xy1, xy2

        image_data = calibration.get_image_json_from_ccc(
            ccc_identifier, image_identifier)
        if image_data is None:
            return json_abort(
                400,
                reason="The image or CCC don't exist",
            )

        image = calibration.get_plate_slice(
            ccc_identifier, image_identifier, plate, gs_transformed=False)
        if image is None:
            return json_abort(
                400,
                reason="No such image slice exists, has it been sliced?",
            )

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        pinning_format = data_object.get("pinning_format")
        try:
            pinning_format = tuple(int(v) for v in pinning_format.split(u","))
        except (ValueError, TypeError):
            app.logger.error(
                "Pinning-format not understood ({0})".format(
                    data_object.get("pinning_format"))
            )
            return json_abort(
                400,
                reason="Bad pinning format",
            )

        correction = data_object.get('gridding_correction')
        if correction:
            try:
                correction = tuple(int(v) for v in correction.split(u","))
            except ValueError:
                app.logger.error(
                    "Correction-format not understood ({0})".format(
                        correction)
                )
                return json_abort(
                    400,
                    reason="Bad grid correction {0}".format(correction),
                )
        else:
            correction = None

        analysis_model = AnalysisModelFactory.create()
        analysis_model.output_directory = ""
        grid_array = GridArray(
            (None, plate - 1), pinning_format, analysis_model)

        if not grid_array.detect_grid(image, grid_correction=correction):
            xy1, xy2 = get_xy1_xy2(grid_array)

            return json_abort(
                400,
                grid=grid_array.grid,
                xy1=xy1,
                xy2=xy2,
                reason="Grid detection failed",
            )

        grid = grid_array.grid
        xy1, xy2 = get_xy1_xy2(grid_array)

        grid_path = Paths().ccc_image_plate_grid_pattern.format(
            ccc_identifier, image_identifier, plate)
        np.save(grid_path, grid)

        app.logger.info(
            "xy1 shape {0}, xy2 shape {1}".format(
                np.asarray(xy1).shape, np.asarray(xy2).shape)
        )

        success = calibration.set_plate_grid_info(
            ccc_identifier, image_identifier, plate,
            grid_shape=pinning_format,
            grid_cell_size=grid_array.grid_cell_size,
            access_token=data_object.get("access_token"))

        if not success:
            return json_abort(
                401,
                grid=grid_array.grid,
                xy1=xy1,
                xy2=xy2,
                reason="Probably bad access token, or trying to re-grid " +
                "image after has been used"
            )

        return jsonify(
            success=True,
            is_endpoint=True,
            grid=grid_array.grid,
            xy1=xy1,
            xy2=xy2
        )

    @app.route(
        "/api/data/calibration/<ccc_identifier>/image/<image_identifier>" +
        "/plate/<int:plate>/detect/colony/<int:x>/<int:y>", methods=["POST"])
    def detect_colony(ccc_identifier, image_identifier, plate, x, y):

        image = calibration.get_plate_slice(
            ccc_identifier, image_identifier, plate, True)

        if image is None:
            return json_abort(
                400,
                reason="Image plate slice hasn't been prepared probably"
            )

        grid_path = Paths().ccc_image_plate_grid_pattern.format(
            ccc_identifier, image_identifier, plate)
        try:
            grid = np.load(grid_path)
        except IOError:
            return json_abort(
                400,
                reason="Gridding is missing"
            )

        image_json = calibration.get_image_json_from_ccc(
            ccc_identifier, image_identifier)

        if not image_json or plate not in \
                image_json[calibration.CCCImage.plates]:

            return json_abort(
                400,
                reason="Image id not known or plate not know"
            )

        plate_json = image_json[calibration.CCCImage.plates][plate]
        h, w = plate_json[calibration.CCCPlate.grid_cell_size]

        px_y, px_x = grid[:, grid.shape[1] - y, x]

        colony_im = image[
            int(round(px_y - h / 2)): int(round(px_y + h / 2) + 1),
            int(round(px_x - w / 2)): int(round(px_x + w / 2) + 1)
        ]

        # first plate, upper left colony (just need something):
        identifier = ["unknown_image", 0, [0, 0]]

        grid_cell = GridCell(identifier, None, save_extra_data=False)
        grid_cell.source = colony_im.astype(np.float64)

        transpose_polynomial = image_basics.Image_Transpose(
            sourceValues=image_json[
                calibration.CCCImage.grayscale_source_values],
            targetValues=image_json[
                calibration.CCCImage.grayscale_target_values]
        )

        grid_cell.source[...] = transpose_polynomial(grid_cell.source)

        grid_cell.attach_analysis(
            blob=True, background=True, cell=True,
            run_detect=False)

        grid_cell.detect(remember_filter=False)
        blob = grid_cell.get_item(COMPARTMENTS.Blob).filter_array
        background = grid_cell.get_item(COMPARTMENTS.Background).filter_array
        blob_exists = blob.any()
        blob_pixels = grid_cell.source[blob]
        background_exists = background.any()
        background_reasonable = background.sum() >= 20

        return jsonify(
            success=True,
            blob=blob.tolist(),
            background=background.tolist(),
            image=grid_cell.source.tolist(),
            image_max=grid_cell.source.max(),
            image_min=grid_cell.source.min(),
            blob_max=blob_pixels.max() if blob_exists else -1,
            blob_min=blob_pixels.min() if blob_exists else -1,
            blob_exists=int(blob_exists),
            background_exists=int(background_exists),
            background_reasonable=int(background_reasonable),
            grid_position=(px_y, px_x),
        )

    @app.route(
        "/api/data/calibration/<ccc_identifier>/image/<image_identifier>/" +
        "plate/<int:plate>/compress/colony/<int:x>/<int:y>", methods=["POST"])
    def calibration_compress(ccc_identifier, image_identifier, plate, x, y):
        """Set compressed calibration entry

        Request Keys:
            "image": The grayscale calibrated image
            "blob": The filter indicating what is the colony
            "background": The filter indicating what is the background
            "override_small_background": boolean for allowing less than
                20 pixel backgrounds
        Returns:

        """
        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        image_data = calibration.get_image_json_from_ccc(
            ccc_identifier, image_identifier)
        if image_data is None:
            return json_abort(
                400,
                reason="The image or CCC don't exist"
            )

        try:
            image = np.array(data_object.get("image", [[]]), dtype=np.float64)
        except TypeError:
            return json_abort(
                400,
                reason="Image data is not understandable as a float array"
            )

        try:
            blob_filter = np.array(data_object.get("blob", [[]]), dtype=bool)
        except TypeError:
            return json_abort(
                400,
                reason="Blob filter data is not understandable as a boolean " +
                "array"
            )

        try:
            background_filter = np.array(
                data_object.get("background", [[]]), dtype=bool)
        except TypeError:
            return json_abort(
                400,
                reason="Background filter data is not understandable as a " +
                "boolean array"
            )

        if not valid_array_dimensions(
                2, image, blob_filter, background_filter):
            return json_abort(
                400,
                reason="Supplied data does not have the correct dimensions." +
                " Image-shape is {0}, blob {1}, and bg {2}.".format(
                    image.shape, blob_filter.shape, background_filter.shape) +
                " All need to be identical shape and 2D."
            )

        if (blob_filter & background_filter).any():
            return json_abort(
                400,
                reason="Blob and background filter may not overlap"
            )

        if not blob_filter.any():
            return json_abort(
                400,
                reason="Blob is empty/there's no colony detected"
            )

        if background_filter.sum() < 3:
            return json_abort(
                400,
                reason="Background must be consisting of at least 3 pixels"
            )

        if background_filter.sum() < 20 and not data_object.get(
                "override_small_background", False):

            return json_abort(
                400,
                reason="Background must be at least 20 pixels." +
                " Currently only {0}.".format(background_filter.sum()) +
                " This check can be over-ridden."
            )

        if calibration.set_colony_compressed_data(
                ccc_identifier, image_identifier, plate, x, y,
                included=True,
                image=image, blob_filter=blob_filter,
                background_filter=background_filter,
                access_token=data_object.get("access_token")):

            return jsonify(
                success=True,
                is_endpoint=True
            )

        else:

            return json_abort(
                401,
                reason="Probably invalid access token"
            )

    @app.route('/api/data/calibration/<ccc_identifier>/external_data/upload',
               methods=['POST'])
    def upload_external_data(ccc_identifier):

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        population_size_data = request.files.get(
            'population_size_data', default=None)

        if population_size_data is None:
            return json_abort(
                400,
                reason="Didn't get any data"
            )

        if not calibration.is_valid_token(
                ccc_identifier,
                access_token=data_object.get("access_token")):

            return json_abort(
                401,
                reason="Invalid access token"
            )

        report = {}
        if calibration.add_external_data_to_ccc(
                ccc_identifier,
                population_size_data,
                access_token=data_object.get("access_token"),
                report=report):

            return jsonify(
                success=True,
                is_endpoint=True,
                report=report
            )

        else:

            return json_abort(
                400,
                report=report)

    @app.route(
        '/api/data/calibration/<ccc_identifier>/delete', methods=['POST'])
    def delete_non_deployed_calibration(ccc_identifier):

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        if not calibration.is_valid_token(
                ccc_identifier,
                access_token=data_object.get("access_token")):

            return json_abort(
                401,
                reason="Invalid access token")

        if delete_ccc(ccc_identifier):

            return jsonify(
                success=True,
                is_endpoint=True
            )

        else:

            return json_abort(
                400,
                reason='Unexpected error removing CCC'
            )

    @app.route('/api/data/calibration/<ccc_identifier>/construct/<poly_name>',
               defaults={'power': 5})
    @app.route(
        '/api/data/calibration/<ccc_identifier>/construct/<poly_name>' +
        '/<int: power>')
    def construct_calibration(ccc_identifier, poly_name, power):

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        if not calibration.is_valid_token(
                ccc_identifier,
                access_token=data_object.get("access_token")):

            return json_abort(
                401,
                reason="Invalid access token")

        # TODO: regex not imported?
        poly_name = re.sub(r'[ .,]]', '_', poly_name)
        checked_name = "".join(c for c in poly_name if c in _VALID_CHARACTERS)

        if poly_name != checked_name:
            return json_abort(
                400,
                reason="Name is containing invalid characters"
            )

        response = calibration.constuct_polynomial(
            ccc_identifier,
            poly_name,
            power,
            access_token=data_object.get("access_token")
        )

        if response["validation"] is calibration.CalibrationValidation.OK:
            return jsonify(
                success=True,
                is_endpoint=True,
                **response
            )
        elif response is False:
            return json_abort(
                400,
                reason="Failed to save ccc")

        return json_abort(
            400,
            reason="Construction refused. " +
            "Validation of polynomial says: {}".format(
                response["validation"].name
            ))

    @app.route('/api/data/calibration/<ccc_identifier>/finalize')
    def finalize_calibration(ccc_identifier):
        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        if not calibration.is_valid_token(
                ccc_identifier,
                access_token=data_object.get("access_token")):

            return json_abort(
                401,
                reason="Invalid access token"
            )

        if calibration.activate_ccc(
                ccc_identifier,
                access_token=data_object.get("access_token")):
            jsonify(
                success=True,
                is_endpoint=True
            )
        else:
            return json_abort(
                400,
                reason="Failed to activate ccc"
            )
