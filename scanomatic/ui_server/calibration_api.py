from flask import Flask, jsonify, request, send_file
import numpy as np
import re
from string import letters
from io import BytesIO
import zipfile
import time
import os

from scanomatic.data_processing import calibration

from scanomatic.data_processing.calibration import add_calibration, CalibrationEntry, calculate_polynomial, \
    load_calibration, validate_polynomial, CalibrationValidation, save_data_to_file, remove_calibration, \
    get_data_file_path

from .general import decorate_api_access_restriction

_VALID_CHARACTERS = letters + "-._1234567890"


def add_routes(app):
    """

    :param app: The flask webb app
     :type app: Flask
    :return:
    """

    @app.route("/api/calibration/active")
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

    @app.route("/api/calibration/initiate_new")
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

    @app.route("/api/calibration/<ccc_identifier>/image/add")
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