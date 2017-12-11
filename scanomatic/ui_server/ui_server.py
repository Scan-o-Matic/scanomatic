from __future__ import absolute_import

import os
import glob
import time
import webbrowser
from socket import error
from threading import Thread, Timer

import requests
from flask import (
    Flask, send_from_directory, redirect, jsonify, render_template)
from flask_cors import CORS

from scanomatic.io.app_config import Config
from scanomatic.io.logger import Logger, LOG_RECYCLE_TIME
from scanomatic.io.paths import Paths
from scanomatic.io.rpc_client import get_client
from scanomatic.io.backup import backup_file
from scanomatic.data_processing import phenotyper

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
from . import help_ui
from . import settings_ui
from . import status_ui
from . import qc_norm_ui
from . import experiment_ui
from .general import (
    serve_log_as_html, convert_url_to_path, get_search_results,
    convert_path_to_url
)

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

    add_base_routes(app, rpc_client)

    help_ui.add_routes(app)
    settings_ui.add_routes(app)
    status_ui.add_routes(app, rpc_client)
    qc_norm_ui.add_routes(app)
    experiment_ui.add_routes(app)

    management_api.add_routes(app, rpc_client)
    tools_api.add_routes(app)
    qc_api.add_routes(app)
    analysis_api.add_routes(app)
    compilation_api.add_routes(app)
    scan_api.add_routes(app)
    data_api.add_routes(app, rpc_client, debug)
    app.register_blueprint(
        calibration_api.blueprint, url_prefix="/api/calibration")
    settings_api.add_routes(app)
    experiment_api.add_routes(app, rpc_client, _LOGGER)

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


def add_base_routes(app, rpc_client):
    """

    :param app: The flask webb app
     :type app: Flask
    :return:
    """

    @app.route("/")
    def _root():
        return render_template(Paths().ui_root_file, debug=app.debug)

    @app.route("/home")
    def _show_homescreen():
        return redirect("/status")

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

    @app.route("/fonts/<font>")
    def _font_base(font=None):
        if font:
            return send_from_directory(Paths().ui_font, font)

    @app.route("/logs/system/<log>")
    def _logs(log):
        """
        Args:
            log: The log-type to be returned {'server' or 'ui_server'}.

        Returns: html-document (or json on invalid log-parameter).

        """
        if log == 'server':
            log_path = Paths().log_server
        elif log == "ui_server":
            log_path = Paths().log_ui_server
        else:
            return jsonify(
                success=False,
                is_endpoint=True,
                reason="No system log of that type")

        return serve_log_as_html(log_path, log.replace("_", " ").capitalize())

    @app.route("/logs/project/<path:project>")
    def _project_logs(project):

        path = convert_url_to_path(project)

        if not os.path.exists(path):

            return jsonify(success=True,
                           is_project=False,
                           is_endpoint=False,
                           exits=['urls'],
                           **get_search_results(path, "/logs/project"))

        is_project_analysis = phenotyper.path_has_saved_project_state(path)

        if not os.path.isfile(path) or not path.endswith(".log"):

            if is_project_analysis:
                logs = glob.glob(
                    os.path.join(path, Paths().analysis_run_log))
                logs += glob.glob(
                    os.path.join(path, Paths().phenotypes_extraction_log))
            else:
                logs = glob.glob(os.path.join(
                    path, Paths().scan_log_file_pattern.format("*")))
                logs += glob.glob(os.path.join(
                    path, Paths().project_compilation_log_pattern.format("*")))

            return jsonify(
                success=True,
                is_project=False,
                is_endpoint=False,
                is_project_analysis=is_project_analysis,
                exits=['urls', 'logs'],
                logs=[
                    convert_path_to_url("/logs/project", log_path)
                    for log_path in logs
                ],
                **get_search_results(path, "/logs/project"))

        include_levels = 3 if is_project_analysis else 2

        return serve_log_as_html(
            path, os.sep.join(path.split(os.path.sep)[-include_levels:]))

    @app.route("/scanners/<scanner_query>")
    def _scanners(scanner_query=None):
        if scanner_query is None or scanner_query.lower() == 'all':
            return jsonify(
                scanners=rpc_client.get_scanner_status(), success=True)
        elif scanner_query.lower() == 'free':
            return jsonify(
                scanners={
                    s['socket']: s['scanner_name'] for s in
                    rpc_client.get_scanner_status()
                    if 'owner' not in s or not s['owner']},
                success=True)
        else:
            try:
                return jsonify(
                    scanner=(
                        s for s in rpc_client.get_scanner_status()
                        if scanner_query
                        in s['scanner_name']).next(),
                    success=True)
            except StopIteration:
                return jsonify(
                    scanner=None, success=False,
                    reason="Unknown scanner or query '{0}'".format(
                        scanner_query))


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
