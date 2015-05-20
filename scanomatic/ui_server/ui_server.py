__author__ = 'martin'

import time
from flask import Flask, request, send_from_directory, redirect
import webbrowser
from threading import Thread
from socket import error
from subprocess import Popen

from scanomatic.io.app_config import Config
from scanomatic.io.paths import Paths
from scanomatic.io.logger import Logger
from scanomatic.io.rpc_client import get_client

_url = None
_logger = Logger("UI-server")


def _launch_scanomatic_rpc_server():
    Popen(["scan-o-matic_server"])


def launch_server(is_local=None, port=None, host=None):

    global _url

    app = Flask("Scan-o-Matic UI")
    rpc_client = get_client(admin=True)

    if rpc_client.local and rpc_client.online is False:
        _launch_scanomatic_rpc_server()

    if port is None:
        port = Config().ui_port

    if is_local is True or (Config().ui_local and is_local is None):
        host = "localhost"
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

    @app.route("/fixtures", methods=['post', 'get'])
    def _fixtures():

        if request.args.get("names"):

            if rpc_client.online:
                return ",".join(rpc_client.get_fixtures())
            else:
                return ""

        elif request.args.get("update"):
            return "Not implemented saving/creating fixtures...sorry"

        elif request.args.get("detect"):
            return ""

        return send_from_directory(Paths().ui_root, Paths().fixture_file)

    try:
        if is_local:
            app.run(port=port)
        else:
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