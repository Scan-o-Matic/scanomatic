from __future__ import absolute_import

from flask import jsonify, request

from scanomatic.io.app_config import Config
from .general import json_abort


def add_routes(app):

    @app.route("/api/settings", methods=['post'])
    def _settings_api():

        app_conf = Config()

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        app_conf.computer_human_name = data_object["computer_human_name"]
        app_conf.mail.warn_scanning_done_minutes_before = data_object[
            "mail"]["warn_scanning_done_minutes_before"]

        bad_data = []
        success = app_conf.validate(bad_data)
        if success:
            app_conf.save_current_settings()
        else:
            return json_abort(400, reason="Bad data for {0}".format(bad_data))
        return jsonify()
