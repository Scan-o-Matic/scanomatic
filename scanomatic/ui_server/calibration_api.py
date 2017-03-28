from flask import Flask, jsonify, request, send_file
import numpy as np
import re
from string import letters
from io import BytesIO
import zipfile
import time
import os
from itertools import product
from types import StringTypes

from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from scanomatic.models.analysis_model import COMPARTMENTS, VALUES
from scanomatic.image_analysis.grid_cell import GridCell
from scanomatic.image_analysis.grid_array import GridArray
from scanomatic.image_analysis.grayscale import getGrayscale
from scanomatic.image_analysis.image_grayscale import get_grayscale_image_analysis
from scanomatic.io.paths import Paths
from scanomatic.io.fixtures import Fixtures
from scanomatic.data_processing import calibration
from scanomatic.data_processing.calibration import add_calibration, calculate_polynomial, \
    load_calibration, validate_polynomial, CalibrationValidation, save_data_to_file, remove_calibration, \
    get_data_file_path
from .general import decorate_api_access_restriction, serve_numpy_as_image, get_grayscale_is_valid, \
    valid_array_dimensions

_VALID_CHARACTERS = letters + "-._1234567890"


def add_routes(app):
    """

    :param app: The flask webb app
     :type app: Flask
    :return:
    """

    @app.route("/api/calibration/active", methods=['GET'])
    @decorate_api_access_restriction
    def get_active_calibrations():

        try:
            identifiers, cccs = zip(*calibration.get_active_cccs().iteritems())
            return jsonify(success=True, is_endpoint=True, identifiers=identifiers,
                           species=[ccc[calibration.CellCountCalibration.species] for ccc in cccs],
                           references=[ccc[calibration.CellCountCalibration.reference] for ccc in cccs])
        except ValueError:
            return jsonify(success=False, is_endpoint=True,
                           reason="There are no registered CCC, Scan-o-Matic won't work before at least one is added")

    @app.route("/api/calibration/under_construction", methods=['GET'])
    @decorate_api_access_restriction
    def get_under_construction_calibrations():

        try:
            identifiers, cccs = zip(*calibration.get_under_construction_cccs().iteritems())
            return jsonify(success=True, is_endpoint=True, identifiers=identifiers,
                           species=[ccc[calibration.CellCountCalibration.species] for ccc in cccs],
                           references=[ccc[calibration.CellCountCalibration.reference] for ccc in cccs])
        except ValueError:
            return jsonify(success=False, is_endpoint=True,
                           reason="No CCCs are under constructions")

    @app.route("/api/calibration/initiate_new", methods=['POST'])
    @decorate_api_access_restriction
    def initiate_new_ccc():

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        species = data_object.get("species")
        reference = data_object.get("reference")
        ccc = calibration.get_empty_ccc(species, reference)
        if ccc is None:
            return jsonify(success=False, is_endpoint=True, reason="Combination of species and reference not unique")

        success = calibration.add_ccc(ccc)

        if not success:
            return jsonify(success=False, is_endpoint=True,
                           reason="Possibly someone just beat you to that combination of species and reference!")

        return jsonify(
            success=True,
            is_endpoint=True,
            identifier=ccc[calibration.CellCountCalibration.identifier],
            access_token=ccc[calibration.CellCountCalibration.edit_access_token])

    @app.route("/api/calibration/<ccc_identifier>/add_image", methods=['POST'])
    @decorate_api_access_restriction
    def upload_ccc_image(ccc_identifier):

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        image = request.files.get('image', default=None)
        if image is None:
            return jsonify(success=False, is_endpoint=True, reason="Didn't get any image")

        image_identifier = calibration.add_image_to_ccc(
            ccc_identifier, image, access_token=data_object.get("access_token"))

        if not image_identifier:
            return jsonify(success=False, is_endpoint=True, reason="Refused to save image, probably bad access token")

        return jsonify(success=True, is_endpoint=True, image_identifier=image_identifier)

    @app.route("/api/calibration/<ccc_identifier>/image_list", methods=['GET'])
    @decorate_api_access_restriction
    def list_ccc_images(ccc_identifier):

        image_list = calibration.get_image_identifiers_in_ccc(ccc_identifier)
        if image_list is False:
            return jsonify(success=False, is_endpoint=True, reason="No such ccc known")

        return jsonify(success=True, is_endpoint=True, image_identifiers=image_list)

    @app.route("/api/calibration/<ccc_identifier>/image/<image_identifier>/get", methods=['GET'])
    @decorate_api_access_restriction
    def download_ccc_image(ccc_identifier, image_identifier):

        im_path = Paths().ccc_image_pattern.format(ccc_identifier, image_identifier)
        return send_file(im_path, mimetype='Image/Tiff')

    @app.route("/api/calibration/<ccc_identifier>/image/<image_identifier>/data/set", methods=['POST'])
    @decorate_api_access_restriction
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
                if data_type is calibration.CCCImage.marker_x or data_type is calibration.CCCImage.marker_y and \
                        isinstance(val, StringTypes):

                    try:
                        val = [float(v) for v in val.split(",")]
                    except ValueError:
                        app.logger.warning("The parameter {0} value '{1}' not understood".format(data_type, val))
                        continue

                data_update[data_type.name] = val

                if data_type is calibration.CCCImage.fixture and \
                        calibration.CCCImage.grayscale_name.name not in data_object.keys():

                    fixture_settings = Fixtures()[val]
                    if fixture_settings is not None:
                        data_update[calibration.CCCImage.grayscale_name.name] = fixture_settings.model.grayscale.name

        success = calibration.set_image_info(
            ccc_identifier, image_identifier, access_token=data_object.get("access_token"), **data_update)

        if not success:
            return jsonify(success=False, is_endpoint=True, reason="Update refused, probably bad access token")

        return jsonify(success=True, is_endpoint=True)

    @app.route("/api/calibration/<ccc_identifier>/image/<image_identifier>/data/get", methods=['GET'])
    @decorate_api_access_restriction
    def get_ccc_image_data(ccc_identifier, image_identifier):

        data = calibration.get_image_json_from_ccc(ccc_identifier, image_identifier)
        if data is None:
            return jsonify(success=False, is_endpoint=True, reason="The image or CCC don't exist")

        return jsonify(success=True, is_endpoint=True,
                       **{k.name: val for k, val in data.iteritems() if k is not calibration.CCCImage.plates})

    @app.route("/api/calibration/<ccc_identifier>/image/<image_identifier>/slice/set", methods=['POST'])
    @decorate_api_access_restriction
    def slice_ccc_image(ccc_identifier, image_identifier):

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        data = calibration.get_local_fixture_for_image(ccc_identifier, image_identifier)
        if data is None:
            return jsonify(success=False, is_endpoint=True,
                           reason="The image or CCC don't exist or not enough info set to do slice")

        success = calibration.save_image_slices(
            ccc_identifier, image_identifier,
            grayscale_slice=data["grayscale"],
            plate_slices=data['plates'],
            access_token=data_object.get("access_token"))

        if not success:
            return jsonify(success=False, is_endpoint=True, reason="Probably not the correct access token.")

        return jsonify(success=True, is_endpoint=True)

    @app.route("/api/calibration/<ccc_identifier>/image/<image_identifier>/slice/get/<slice>", methods=['GET'])
    @decorate_api_access_restriction
    def get_ccc_image_slice(ccc_identifier, image_identifier, slice):
        """

        :param ccc_identifier:
        :param image_identifier:
        :param slice: either 'gs' for grayscale or the plate index 0-3
        :return:
        """
        if slice.lower() == 'gs':
            im = calibration.get_grayscale_slice(ccc_identifier, image_identifier)
        else:
            im = calibration.get_plate_slice(ccc_identifier, image_identifier, slice, gs_transformed=False)

        if im is None:
            return jsonify(success=False, is_endpoint=True, reason="No such image slice exists, has it been sliced?")
        return serve_numpy_as_image(im)

    @app.route("/api/calibration/<ccc_identifier>/image/<image_identifier>/grayscale/analyse", methods=['POST'])
    @decorate_api_access_restriction
    def get_ccc_image_grayscale_analysis(ccc_identifier, image_identifier):

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        gs_image = calibration.get_grayscale_slice(ccc_identifier, image_identifier)
        image_data = calibration.get_image_json_from_ccc(ccc_identifier, image_identifier)
        try:
            gs_name = image_data[calibration.CCCImage.grayscale_name]
        except TypeError:
            return jsonify(success=False, is_endpoint=True, reason='Unknown image or CCC')

        _, values = get_grayscale_image_analysis(gs_image, gs_name, debug=False)
        grayscale_object = getGrayscale(gs_name)
        valid = get_grayscale_is_valid(values, grayscale_object)
        if not valid:
            return jsonify(success=False, is_endpoint=True, reason='Grayscale results are not valid')

        success = calibration.set_image_info(
            ccc_identifier, image_identifier,
            grayscale_source_values=values,
            grayscale_target_values=grayscale_object['targets'],
            access_token=data_object.get("access_token"))

        if success:

            return jsonify(success=True, is_endpoint=True, source_values=values,
                           target_values=grayscale_object['targets'])

        else:

            return jsonify(success=False, is_endpoint=True,
                           reason='Refused to set image grayscale info, probably bad access token')

    @app.route("/api/calibration/<ccc_identifier>/image/<image_identifier>/plate/<int:plate>/transform", methods=['POST'])
    @decorate_api_access_restriction
    def get_ccc_image_plate_transform(ccc_identifier, image_identifier, plate):

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        success = calibration.transform_plate_slice(ccc_identifier, image_identifier, plate,
                                                    access_token=data_object.get("access_token"))
        if not success:
            return jsonify(success=False, is_endpoint=True,
                           reason="Probably bad access token or not having sliced image and analysed grayscale first")

        return jsonify(success=True, is_endpoint=True)

    @app.route("/api/calibration/<ccc_identifier>/image/<image_identifier>/plate/<int:plate>/grid/set", methods=['POST'])
    @decorate_api_access_restriction
    def grid_ccc_image_plate(ccc_identifier, image_identifier, plate):

        image_data = calibration.get_image_json_from_ccc(ccc_identifier, image_identifier)
        if image_data is None:
            return jsonify(success=False, is_endpoint=True, reason="The image or CCC don't exist")

        im = calibration.get_plate_slice(ccc_identifier, image_identifier, plate, gs_transformed=False)
        if im is None:
            return jsonify(success=False, is_endpoint=True, reason="No such image slice exists, has it been sliced?")

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        pinning_format = data_object.get("pinning_format")
        try:
            pinning_format = tuple(int(v) for v in pinning_format.split(u","))
        except (ValueError, TypeError):
            app.logger.error("Pinning-format not understood ({0})".format(data_object.get("pinning_format")))
            return jsonify(success=False, is_endpoint=True, reason="Bad pinning format")

        correction = data_object.get('gridding_correction')
        if correction:
            try:
                correction = tuple(int(v) for v in correction.split(u","))
            except ValueError:
                app.logger.error("Correction-format not understood ({0})".format(correction))
                return jsonify(success=False, is_endpoint=True, reason="Bad grid correction {0}".format(correction))
        else:
            correction = None

        analysis_model = AnalysisModelFactory.create()
        analysis_model.output_directory = ""
        ga = GridArray((None, plate - 1), pinning_format, analysis_model)

        if not ga.detect_grid(im, grid_correction=correction):
            return jsonify(
                success=False,
                reason="Grid detection failed",
                is_endpoint=True,
            )

        grid = ga.grid
        inner = len(grid[1])
        outer = len(grid[2])
        xy1 = [[[None] for c in range(inner)] for r in range(outer)]
        xy2 = [[[None] for c in range(inner)] for r in range(outer)]

        for o, i in product(range(outer), range(inner)):

            gc = ga[(i, o)]

            xy1[o][i] = gc.xy1.tolist()
            xy2[o][i] = gc.xy2.tolist()

        grid_path = Paths().ccc_image_plate_grid_pattern.format(ccc_identifier, image_identifier, plate)
        np.save(grid_path, grid)

        app.logger.info("xy1 shape {0}, xy2 shape {1}".format(np.asarray(xy1).shape, np.asarray(xy2).shape))

        success = calibration.set_plate_grid_info(
            ccc_identifier, image_identifier, plate,
            grid_shape=pinning_format,
            grid_cell_size=ga.grid_cell_size,
            access_token=data_object.get("access_token"))

        if not success:
            return jsonify(success=False, is_endpoint=True,
                           reason="Probably bad access token, or trying to re-grid image after has been used")

        return jsonify(success=True, is_endpoint=True,
                       grid=grid,
                       xy1=xy1,
                       xy2=xy2)

    @app.route(
        "/api/data/calibration/<ccc_identifier>/image/<image_identifier>/plate/<int:plate>/detect/colony/<int:x>/<int:y>",
        methods=["POST"])
    @decorate_api_access_restriction
    def detect_colony(ccc_identifier, image_identifier, plate, x, y):

        im = calibration.get_plate_slice(ccc_identifier, image_identifier, plate, True)

        if im is None:
            return jsonify(success=False, is_endpoint=True, reason="Image plate slice hasn't been prepared probably")

        grid_path = Paths().ccc_image_plate_grid_pattern.format(ccc_identifier, image_identifier, plate)
        try:
            grid = np.load(grid_path)
        except IOError:
            return jsonify(success=False, is_endpoint=True, reason="Gridding is missing")

        image_json = calibration.get_image_json_from_ccc(ccc_identifier, image_identifier)

        if not image_json or plate not in image_json[calibration.CCCImage.plates]:

            return jsonify(success=False, is_endpoint=True, reason="Image id not known or plate not know")

        plate_json = image_json[calibration.CCCImage.plates][plate]
        h, w = plate_json[calibration.CCCPlate.grid_cell_size]

        px_y, px_x = grid[:, y, x]

        colony_im = im[
            int(round(px_y - h/2)): int(round(px_y + h/2) + 1),
            int(round(px_x - w / 2)): int(round(px_x + w / 2) + 1)]

        identifier = ["unknown_image", 0, [0, 0]]  # first plate, upper left colony (just need something

        gc = GridCell(identifier, None, save_extra_data=False)
        gc.source = colony_im.astype(np.float64)
        gc.attach_analysis(
            blob=True, background=True, cell=True,
            run_detect=False)

        gc.detect(remember_filter=False)

        return jsonify(
            success=True,
            blob=gc.get_item(COMPARTMENTS.Blob).filter_array.tolist(),
            background=gc.get_item(COMPARTMENTS.Background).filter_array.tolist(),
            image=colony_im.tolist(),
            grid_position=(px_y, px_x),
        )

    @app.route(
        "/api/data/calibration/<ccc_identifier>/image/<image_identifier>/plate/<int:plate>/compress/colony/<int:x>/<int:y>",
        methods=["POST"])
    @decorate_api_access_restriction
    def calibration_compress(ccc_identifier, image_identifier, plate, x, y):
        """Set compressed calibration entry

        Request Keys:
            "image": The grayscale calibrated image
            "blob": The filter indicating what is the colony
            "background": The filter indicating what is the background
        Returns:

        """
        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        image_data = calibration.get_image_json_from_ccc(ccc_identifier, image_identifier)
        if image_data is None:
            return jsonify(success=False, is_endpoint=True, reason="The image or CCC don't exist")

        try:
            image = np.array(data_object.get("image", [[]]), dtype=np.float64)
        except TypeError:
            return jsonify(success=False, is_endpoint=True,
                           reason="Image data is not understandable as a float array")

        try:
            blob_filter = np.array(data_object.get("blob", [[]]), dtype=bool)
        except TypeError:
            return jsonify(success=False, is_endpoint=True,
                           reason="Blob filter data is not understandable as a boolean array")

        try:
            background_filter = np.array(data_object.get("background", [[]]))
        except TypeError:
            return jsonify(success=False, is_endpoint=True,
                           reason="Background filter data is not understandable as a boolean array")

        if not valid_array_dimensions(2, image, blob_filter, background_filter):
            return jsonify(success=False, is_endpoint=True,
                           reason="Supplied data does not have the correct dimensions." +
                           " Image-shape is {0}, blob-shape {1}, and background-shape {2}.".format(
                               image.shape, blob_filter.shape, background_filter.shape) +
                           " All need to be identical shape and 2D."
                           )

        if (blob_filter & background_filter).any():
            return jsonify(success=False, is_endpoint=True,
                           reason="Blob and background filter may not overlap")

        if not blob_filter.any():
            return jsonify(success=False, is_endpoint=False,
                           reason="Blob is empty/there's no colony detected")

        if not background_filter.any():
            return jsonify(success=False, is_endpoint=False,
                           reason="Background is empty/there's no background detected")

        if calibration.set_colony_compressed_data(
                ccc_identifier, image_identifier, plate, x, y,
                included=True,
                image=image, blob_filter=blob_filter, background_filter=background_filter):

            return jsonify(sucecss=True, is_endpoint=True)

        else:

            return jsonify(success=False, is_endpoint=False, reason="Probably invalid access token")

    """
    DEPRECATION WARNING BELOW
    """

    @app.route("/api/calibration/add/<name>")
    @app.route("/api/calibration/add/<name>/<int:degree>")
    @decorate_api_access_restriction
    def calibration_add(name, degree=5):

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        entries = data_object.get("entries", [])
        poly = calculate_polynomial(entries, degree)

        validity = validate_polynomial(entries, poly)
        if validity != CalibrationValidation.OK:
            return jsonify(success=False, reason=validity.name)

        name = re.sub(r'[ .,]]', '_', name)
        name = "".join(c for c in name if c in _VALID_CHARACTERS)

        if not name:
            return jsonify(success=False, reason="Name contains no valid characters ({0})".format(_VALID_CHARACTERS))

        save_data_to_file(entries, label=name)

        data_path = None

        add_calibration(name, poly, data_path)

        return jsonify(success=True, poly=poly, name=name)

    @app.route("/api/calibration/get")
    @app.route("/api/calibration/get/<name>")
    @app.route("/api/calibration/get/<name>/<int:degree>")
    @decorate_api_access_restriction
    def calibration_get(name="", degree=None):

        try:
            return jsonify(success=True, poly=list(load_calibration(name, degree)))
        except (TypeError, AttributeError):
            return jsonify(success=False, reason="Can't find polynomial '{0}' (Degree: {1})".format(name, degree))

    @app.route("/api/calibration/remove/<name>")
    @app.route("/api/calibration/remove/<name>/<int:degree>")
    @decorate_api_access_restriction
    def calibration_remove(name, degree=None):

        if remove_calibration(label=name, degree=degree):
            return jsonify(success=True)
        else:
            return jsonify(success=False,
                           reason="No calibration found matching criteria (name={0}, degree={1})".format(
                               name,
                               "*" if degree is None else degree
                           ))

    @app.route("/api/calibration/export")
    @app.route("/api/calibration/export/<name>")
    @decorate_api_access_restriction
    def calibration_export(name=''):

        data_path = get_data_file_path(label=name)

        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:

            data = zipfile.ZipInfo(os.path.basename(data_path))
            data.date_time = time.localtime(time.time())[:6]
            data.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(data, open(data_path, 'r').read())

        memory_file.flush()
        memory_file.seek(0)
        if not name:
            name = 'default'

        return send_file(memory_file, attachment_filename='calibration.{0}.zip'.format(name), as_attachment=True)
