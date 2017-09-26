from flask import send_from_directory

from scanomatic.io.paths import Paths


def add_routes(app):

    @app.route("/qc_norm")
    def _qc_norm():
        return send_from_directory(Paths().ui_root, Paths().ui_qc_norm_file)
