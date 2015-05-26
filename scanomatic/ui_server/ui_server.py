__author__ = 'martin'

import time
from flask import Flask, request, send_from_directory, redirect, jsonify, abort
import webbrowser
from threading import Thread
from socket import error
from subprocess import Popen
import os

from scanomatic.io.app_config import Config
from scanomatic.io.paths import Paths
from scanomatic.io.logger import Logger
from scanomatic.io.rpc_client import get_client
from scanomatic.imageAnalysis.first_pass_image import FixtureImage
from scanomatic.imageAnalysis.support import save_image_as_png
from scanomatic.models.fixture_models import GrayScaleAreaModel
from scanomatic.imageAnalysis.grayscale import getGrayscales

_url = None
_logger = Logger("UI-server")
_ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.tiff'}


def _launch_scanomatic_rpc_server():
    Popen(["scan-o-matic_server"])


def _allowed_image(ext):
    return ext.lower() in _ALLOWED_EXTENSIONS

def get_fixture_image(name, image_path):

    fixture = FixtureImage()
    fixture.name = name
    fixture.set_image(image_path=image_path)
    return fixture

def launch_server(is_local=None, port=None, host=None, debug=False):

    global _url

    app = Flask("Scan-o-Matic UI")
    rpc_client = get_client(admin=True)

    if rpc_client.local and rpc_client.online is False:
        _launch_scanomatic_rpc_server()

    if port is None:
        port = Config().ui_port

    if is_local is True or (Config().ui_local and is_local is None):
        host = "localhost"
        is_local = True
    elif host is None:
        host = "0.0.0.0"
        is_local = False

    _url = "http://{host}:{port}".format(host=host, port=port)

    @app.route("/")
    def _root():
        return """<!DOCTYPE: html>
        <html>
        <head>
            <link rel="stylesheet" type="text/css" href="style/main.css">
            <title>Scan-o-Matic</title>
            <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.11.0/jquery.min.js"></script>
        </head>
        <body>
        <img id='logo' src='images/help_logo.png'>
        <script>
        $("#logo").bind("load", function () { $(this).hide().fadeIn(4000); });
        </script>
        <ul>
        <li><a href="/help">Help</a></li>
        <li><a href="/wiki">Wiki</a></li>
        <li><a href="/fixtures">Fixtures</a></li>
        </ul>
        </body>
        </html>
        """

    @app.route("/help")
    def _help():
        return send_from_directory(Paths().ui_root, Paths().help_file)

    @app.route("/wiki")
    def _wiki():
        return redirect("https://github.com/local-minimum/scanomatic/wiki")

    @app.route("/images/<image_name>")
    def _help_logo(image_name=None):
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

    @app.route("/fixtures/<name>")
    def _fixture_data(name=None):
        if rpc_client.online and name in rpc_client.get_fixtures():
            return "Not implemented sending fixture data"
        else:
            return ""

    @app.route("/grayscales", methods=['post', 'get'])
    def _grayscales():

        if request.args.get("names"):

            return jsonify(grayscales=getGrayscales())

        return ""

    @app.route("/fixtures", methods=['post', 'get'])
    def _fixtures():

        if request.args.get("names"):

            if rpc_client.online:
                return ",".join(rpc_client.get_fixtures())
            else:
                return ""

        elif request.args.get("update"):
            return "Not implemented saving/creating fixtures...sorry"

        elif request.args.get("grayscale"):

            name = request.args.get("fixture", "", type=str)

            if name:
                fixture_file = Paths().get_fixture_path(name)
                grayscale_area_model = GrayScaleAreaModel(
                    name=request.args.get("grayscale", "", type=str),
                    x1=request.form.get("x1", type=int),
                    x2=request.form.get("x2", type=int),
                    y1=request.form.get("y1", type=int),
                    y2=request.form.get("y2", type=int))
                ext = "tiff"
                image_path = os.path.extsep.join((fixture_file, ext))
                fixture = get_fixture_image(name, image_path)
                fixture['current'].model.grayscale = grayscale_area_model
                fixture.analyse_grayscale()
                return jsonify(source_values=fixture['current'].model.grayscale.values)
            else:
                return abort(500)

        elif request.args.get("detect"):

            markers = request.values.get('markers', default=3, type=int)
            image = request.files.get('image')
            name = os.path.basename(request.values.get("name", '', type=str))
            image_name, ext = os.path.splitext(image.filename)
            _logger.info("Working on detecting marker for fixture {0} using image {1} ({2})".format(
                name, image.filename, _allowed_image(ext)))

            if name and _allowed_image(ext):

                fixture_file = Paths().get_fixture_path(name)

                path = os.path.extsep.join((fixture_file, ext.lstrip(os.path.extsep)))
                image.save(path)

                fixture = get_fixture_image(name, path)
                fixture.run_marker_analysis(markings=markers)

                save_image_as_png(path)

                return jsonify(markers=str(fixture['current'].get_marker_positions()),
                               image=os.path.basename(fixture_file))

            _logger.warning("Refused detection (keys files: {0} values: {1})".format(
                request.files.keys(), request.values.keys()))

            return jsonify(markers="[]", image="")

        elif request.args.get("image"):

            image = os.path.extsep.join((os.path.basename(request.args.get("image")), "png"))
            _logger.info("Sending fixture image {0}".format(image))
            return send_from_directory(Paths().fixtures, image)

        return send_from_directory(Paths().ui_root, Paths().fixture_file)

    try:
        if is_local:
            if debug:
                _logger.info("Running in debug mode.")
            app.run(port=port, debug=debug)
        else:
            if debug:
                _logger.warning("Debugging is only allowed on local servers")
            app.run(port=port, host=host)

    except error:
        _logger.warning("Could not bind socket, probably server is already running and this is nothing to worry about."
                        + "\n\tIf old server is not responding, try killing its process."
                        + "\n\tIf something else is blocking the port, try setting another port using --help.")
        return False
    return True


def launch_webbrowser(delay=0.0):

    if delay:
        _logger.info("Will open webbrowser in {0} s".format(delay))
        time.sleep(delay)

    if _url:
        webbrowser.open(_url)
    else:
        _logger.error("No server launched")


def launch(open_browser_url=True, **kwargs):

    if open_browser_url:
        Thread(target=launch_webbrowser, kwargs={"delay": 2}).start()
    launch_server(**kwargs)