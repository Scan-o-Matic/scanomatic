from flask import Flask, jsonify, request, send_file
import numpy as np
import re
from string import letters
from io import BytesIO
import zipfile
import time
import os

from scanomatic.data_processing import calibration
from scanomatic.io.fixtures import Fixtures
from scanomatic.image_analysis.image_grayscale import get_grayscale_image_analysis
from scanomatic.data_processing.calibration import add_calibration, CalibrationEntry, calculate_polynomial, \
    load_calibration, validate_polynomial, CalibrationValidation, save_data_to_file, remove_calibration, \
    get_data_file_path
from scanomatic.image_analysis.grayscale import getGrayscale
from .general import decorate_api_access_restriction, serve_numpy_as_image, get_grayscale_is_valid

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
            identifiers, cccs = zip(calibration.get_active_cccs().iteritems())
            return jsonify(success=True, is_endpoint=True, identifiers=identifiers,
                           species=[ccc[calibration.CellCountCalibration.species] for ccc in cccs],
                           references=[ccc[calibration.CellCountCalibration.reference] for ccc in cccs])
        except ValueError:
            return jsonify(success=False, is_endpoint=True,
                           reason="There are no registered CCC, Scan-o-Matic won't work before at least one is added")

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

    @app.route("/api/calibration/<ccc_identifier>/image_list", methods=['POST'])
    @decorate_api_access_restriction
    def upload_ccc_image(ccc_identifier):

        image_list = calibration.get_image_identifiers_in_ccc(ccc_identifier)
        if image_list is False:
            return jsonify(success=False, is_endpoint=True, reason="No such ccc known")

        return jsonify(success=True, is_endpoint=True, image_identifiers=image_list)

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
            plates=data['plates'],
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

        gs_image = calibration.get_grayscale_slice(ccc_identifier, image_identifier)
        gs_name = gs_image[calibration.CCCImage.grayscale_name]

        _, values = get_grayscale_image_analysis(gs_image, gs_name, debug=False)
        grayscale_object = getGrayscale(gs_name)
        valid = get_grayscale_is_valid(values, grayscale_object)
        if not valid:
            return jsonify(success=True, is_endpoint=True, reason='Grayscale results are not valid')

        calibration.CCCImage.grayscale_target_values
        calibration.set_image_info(ccc_identifier, image_identifier,
                                   grayscale_source_values=values,
                                   grayscale_target_values=grayscale_object['targets'])

        return jsonify(success=True, is_endpoint=True, source_values=values, target_values=grayscale_object['targets'])

    @app.route("/api/calibration/<ccc_identifier>/image/<image_identifier>/plate/<plate>/transform", methods=['POST'])
    @decorate_api_access_restriction
    def get_ccc_image_plate_transform(ccc_identifier, image_identifier, plate):

        success = calibration.transform_plate_slice(ccc_identifier, image_identifier, plate)
        if not success:
            return jsonify(success=False, is_endpoint=True,
                           reason="Probably bad access token or not having sliced image and analysed grayscale first")

        return jsonify(success=True, is_endpoint=True)

    @app.route("/api/calibration/compress")
    @decorate_api_access_restriction
    def calibration_compress():
        """Get compressed calibration entry

        Request Keys:
            "image": The grayscale calibrated image
            "image_filter": The filter indicating what is the colony
        Returns:

        """
        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        image = np.array(data_object.get("image", [[]]))
        image_filter = np.array(data_object.get("filter", [[]]))
        image_name = data_object.get("image_name", "")
        colony_name = data_object.get("colony_name", "")
        target_value = float(data_object.get("target_value", 0))
        colony = image[image_filter].ravel()
        keys, counts = zip(*{k: (colony == k).sum() for k in np.unique(colony).tolist()}.iteritems())
        return jsonify(success=True, entry={CalibrationEntry.target_value.name: target_value,
                                            CalibrationEntry.source_value_counts.name: counts,
                                            CalibrationEntry.source_values.name: keys,
                                            CalibrationEntry.image.name: image_name,
                                            CalibrationEntry.colony_name.name: colony_name})

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

        memory_file.seek(0)
        if not name:
            name = 'default'

        return send_file(memory_file, attachment_filename='calibration.{0}.zip'.format(name), as_attachment=True)