from flask import send_from_directory, render_template

from scanomatic.io.paths import Paths
from scanomatic.io.app_config import Config


def add_routes(app):

    @app.route("/ccc")
    def _ccc():
        return send_from_directory(Paths().ui_root, Paths().ui_ccc_file)

    @app.route("/maintain")
    def _maintain():
        return send_from_directory(Paths().ui_root, Paths().ui_maintain_file)

    @app.route("/settings", methods=['get'])
    def _settings():

        app_conf = Config()

        return render_template(
            Paths().ui_settings_template, **app_conf.model_copy())

    @app.route("/fixtures", methods=['get'])
    def _fixtures():

        return send_from_directory(Paths().ui_root, Paths().ui_fixture_file)
