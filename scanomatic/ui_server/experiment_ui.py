"""..."""
from flask import send_from_directory

from scanomatic.io.paths import Paths


def add_routes(app):

    @app.route("/feature_extract", methods=['get'])
    def _feature_extract():

        return send_from_directory(
            Paths().ui_root, Paths().ui_feature_extract_file)

    @app.route("/analysis", methods=['get'])
    def _analysis():

        return send_from_directory(Paths().ui_root, Paths().ui_analysis_file)

    @app.route("/experiment", methods=['get'])
    def _experiment():

        return send_from_directory(Paths().ui_root, Paths().ui_experiment_file)

    @app.route("/compile", methods=['get'])
    def _compile():

        return send_from_directory(Paths().ui_root, Paths().ui_compile_file)
