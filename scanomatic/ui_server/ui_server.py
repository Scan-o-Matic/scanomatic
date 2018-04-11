from __future__ import absolute_import

import os

from flask import Flask, send_from_directory

from scanomatic.io.app_config import Config
from scanomatic.io.paths import Paths
from scanomatic.io.rpc_client import get_client
from scanomatic.io.imagestore import ImageStore
from scanomatic.data.util import get_database_url

from . import database
from . import qc_api
from . import analysis_api
from . import compilation_api
from . import calibration_api
from . import scan_api
from . import management_api
from . import tools_api
from . import data_api
from . import experiment_api
from . import status_api
from . import ui_pages
from . import scanners_api
from . import scan_jobs_api
from . import scans_api
from .flask_prometheus import Prometheus


def create_app():
    app = Flask(__name__)
    Prometheus(app)
    app.config['imagestore'] = ImageStore(Config().paths.projects_root)
    app.config['DATABASE_URL'] = get_database_url()
    database.setup(app)
    rpc_client = get_client(admin=True)
    add_resource_routes(app, rpc_client)
    ui_pages.add_routes(app)
    management_api.add_routes(app, rpc_client)
    tools_api.add_routes(app)
    qc_api.add_routes(app)
    analysis_api.add_routes(app)
    compilation_api.add_routes(app)
    scan_api.add_routes(app)
    status_api.add_routes(app, rpc_client)
    data_api.add_routes(app, rpc_client)
    app.register_blueprint(
        calibration_api.blueprint, url_prefix="/api/calibration")
    experiment_api.add_routes(app, rpc_client)
    app.register_blueprint(
        scan_jobs_api.blueprint, url_prefix="/api/scan-jobs"
    )
    app.register_blueprint(
        scanners_api.blueprint, url_prefix="/api/scanners"
    )
    app.register_blueprint(scans_api.blueprint, url_prefix="/api/scans")
    return app


def add_resource_routes(app, rpc_client):
    """

    :param app: The flask webb app
     :type app: Flask
    :return:
    """

    @app.route("/images/<image_name>")
    def _image_base(image_name=None):
        if image_name:
            return send_from_directory(Paths().images, image_name)

    @app.route("/style/<style>")
    def _css_base(style=None):
        if style:
            return send_from_directory(Paths().ui_css, style)

    @app.route("/js/<js>")
    def _js_base(js=None):
        if js:
            return send_from_directory(Paths().ui_js, js)

    @app.route("/js/<group>/<js>")
    def _js_external(group, js):
        return send_from_directory(os.path.join(Paths().ui_js, group), js)

    @app.route("/fonts/<font>")
    def _font_base(font=None):
        if font:
            return send_from_directory(Paths().ui_font, font)
