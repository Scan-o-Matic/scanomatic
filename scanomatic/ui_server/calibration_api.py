from flask import Flask, jsonify, request, send_file
import numpy as np
import re
from string import letters
from io import BytesIO
import zipfile
import time

from scanomatic.data_processing.calibration import add_calibration, CalibrationEntry, calculate_polynomial, \
    load_calibration, validate_polynomial, CalibrationValidation, save_data_to_file, remove_calibration, \
    get_data_file_path

_VALID_CHARACTERS = letters + "-._1234567890"


def add_routes(app):
    """

    :param app: The flask webb app
     :type app: Flask
    :return:
    """


    @app.route("/api/calibration/compress")
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
    def calibration_get(name="", degree=None):

        try:
            return jsonify(success=True, poly=list(load_calibration(name, degree)))
        except (TypeError, AttributeError):
            return jsonify(success=False, reason="Can't find polynomial '{0}' (Degree: {1})".format(name, degree))

    @app.route("/api/calibration/remove/<name>")
    @app.route("/api/calibration/remove/<name>/<int:degree>")
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
    def calibration_export(name=None):

        data_path = get_data_file_path(label=name)

        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:

            data = zipfile.ZipInfo(data_path)
            data.date_time = time.localtime(time.time())[:6]
            data.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(data, data_path)

        memory_file.seek(0)
        return send_file(memory_file, attachment_filename='{0}.zip'.format(name), as_attachment=True)