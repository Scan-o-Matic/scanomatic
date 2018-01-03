from __future__ import absolute_import

import os
import time
import webbrowser
from socket import error
from threading import Thread, Timer

import requests
from flask import Flask, send_from_directory
from flask_cors import CORS

from scanomatic.io.app_config import Config
from scanomatic.io.logger import Logger, LOG_RECYCLE_TIME
from scanomatic.io.paths import Paths
from scanomatic.io.rpc_client import get_client
from scanomatic.io.backup import backup_file
from scanomatic.io.scanstore import ScanStore

from . import qc_api
from . import analysis_api
from . import compilation_api
from . import calibration_api
from . import scan_api
from . import management_api
from . import tools_api
from . import data_api
from . import settings_api
from . import experiment_api
from . import status_api
from . import ui_pages

_URL = None
_LOGGER = Logger("UI-server")
_DEBUG_MODE = None


def init_logging():

    _LOGGER.pause()
    backup_file(Paths().log_ui_server)
    _LOGGER.set_output_target(
        Paths().log_ui_server,
        catch_stdout=_DEBUG_MODE is False, catch_stderr=_DEBUG_MODE is False)
    _LOGGER.surpress_prints = _DEBUG_MODE is False
    _LOGGER.resume()


def launch_server(host, port, debug):

    global _URL, _DEBUG_MODE
    _DEBUG_MODE = debug
    app = Flask("Scan-o-Matic UI", template_folder=Paths().ui_templates)
    app.config['scanstore'] = ScanStore(Config().paths.projects_root)

    rpc_client = get_client(admin=True)

    if rpc_client.local and rpc_client.online is False:
        rpc_client.launch_local()

    if port is None:
        port = Config().ui_server.port
    if host is None:
        host = Config().ui_server.host

    _URL = "http://{host}:{port}".format(host=host, port=port)
    init_logging()
    _LOGGER.info("Requested to launch UI-server at {0} being debug={1}".format(
        _URL, debug))

    app.log_recycler = Timer(LOG_RECYCLE_TIME, init_logging)
    app.log_recycler.start()

    add_resource_routes(app, rpc_client)

    ui_pages.add_routes(app)

    management_api.add_routes(app, rpc_client)
    tools_api.add_routes(app)
    qc_api.add_routes(app)
    analysis_api.add_routes(app)
    compilation_api.add_routes(app)
    scan_api.add_routes(app)
    status_api.add_routes(app, rpc_client)
    data_api.add_routes(app, rpc_client, debug)
    app.register_blueprint(
        calibration_api.blueprint, url_prefix="/api/calibration")
    settings_api.add_routes(app)
    experiment_api.add_routes(app, rpc_client)

    if debug:
        CORS(app)
        _LOGGER.warning(
            "\nRunning in debug mode, causes sequrity vunerabilities:\n" +
            " * Remote code execution\n" +
            " * Cross-site request forgery\n" +
            "   (https://en.wikipedia.org/wiki/Cross-site_request_forgery)\n" +
            "\nAnd possibly more issues"

        )
    try:
        app.run(port=port, host=host, debug=debug)
    except error:
        _LOGGER.warning(
            "Could not bind socket, probably server is already running and" +
            " this is nothing to worry about." +
            "\n\tIf old server is not responding, try killing its process." +
            "\n\tIf something else is blocking the port," +
            " try setting another port" +
            " (see `scan-o-matic --help` for instructions).")
        return False
    return True


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


def launch_webbrowser(delay=0.0):

    if delay:
        _LOGGER.info("Will open webbrowser in {0} s".format(delay))
        time.sleep(delay)

    if _URL:
        webbrowser.open(_URL)
    else:
        _LOGGER.error("No server launched")


def launch(host, port, debug, open_browser_url=True):
    if open_browser_url:
        _LOGGER.info("Getting ready to open browser")
        Thread(target=launch_webbrowser, kwargs={"delay": 2}).start()
    else:
        _LOGGER.info("Will not open browser")

    launch_server(host, port, debug)


def ui_server_responsive():

    port = Config().ui_server.port
    if not port:
        port = 5000
    host = Config().ui_server.host
    if not host:
        host = 'localhost'
    try:
        return requests.get("http://{0}:{1}".format(host, port)).ok
    except requests.ConnectionError:
        return False
