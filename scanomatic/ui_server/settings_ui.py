from flask import (
    send_from_directory, redirect, request, render_template, jsonify)

from scanomatic.io.paths import Paths
from scanomatic.io.app_config import Config
from scanomatic.io.power_manager import POWER_MANAGER_TYPE


def add_routes(app):

    @app.route("/ccc")
    def _ccc():
        return send_from_directory(Paths().ui_root, Paths().ui_ccc_file)

    @app.route("/maintain")
    def _maintain():
        return send_from_directory(Paths().ui_root, Paths().ui_maintain_file)

    @app.route("/settings", methods=['get', 'post'])
    def _settings():

        app_conf = Config()

        action = request.args.get("action")
        if action == "update":

            data_object = request.get_json(silent=True, force=True)
            if not data_object:
                data_object = request.values

            app_conf.number_of_scanners = data_object["number_of_scanners"]
            app_conf.power_manager.number_of_sockets = data_object[
                "power_manager"]["sockets"]
            app_conf.power_manager.host = data_object["power_manager"]["host"]
            app_conf.power_manager.mac = data_object["power_manager"]["mac"]
            app_conf.power_manager.name = data_object["power_manager"]["name"]
            app_conf.power_manager.password = data_object[
                "power_manager"]["password"]
            app_conf.power_manager.host = data_object["power_manager"]["host"]
            app_conf.power_manager.type = POWER_MANAGER_TYPE[
                data_object["power_manager"]["type"]]
            app_conf.computer_human_name = data_object["computer_human_name"]
            app_conf.mail.warn_scanning_done_minutes_before = data_object[
                "mail"]["warn_scanning_done_minutes_before"]

            bad_data = []
            success = app_conf.validate(bad_data)
            app_conf.save_current_settings()
            return jsonify(
                success=success,
                reason=None if success else "Bad data for {0}".format(
                    bad_data))
        elif action:
            return jsonify(success=False, reason="Not implemented")

        return render_template(
            Paths().ui_settings_template, **app_conf.model_copy())

    @app.route("/fixtures", methods=['post', 'get'])
    def _fixtures():

        return send_from_directory(Paths().ui_root, Paths().ui_fixture_file)
