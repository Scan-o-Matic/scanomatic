__author__ = 'martin'

import time
from flask import Flask, render_template, send_from_directory
import webbrowser
from threading import Thread
from socket import error
import os

from scanomatic.io.app_config import Config
from scanomatic.io.paths import Paths
from scanomatic.io.logger import Logger

_url = None
_logger = Logger("UI-server")


def launch_server(is_local=None, port=None, host=None):

    global _url

    app = Flask("Scan-o-Matic UI")

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
        return app.name

    @app.route("/help")
    def _help():
        return send_from_directory(Paths().root, Paths().help_file)

    @app.route("/images/help_logo.png")
    def _help_logo():
        return send_from_directory(Paths().images, "help_logo.png")

    @app.route("/style.css")
    def _css_base():
        return send_from_directory(Paths().root, "style.css")

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


def launch(**kwargs):

    Thread(target=launch_webbrowser, kwargs={"delay": 2}).start()
    launch_server(**kwargs)