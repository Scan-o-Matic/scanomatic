__author__ = 'martin'

import time
from flask import Flask
import webbrowser
from threading import Thread
from socket import error

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
        import socket
        host = socket.gethostbyname(socket.gethostname())

    _url = "http://{host}:{port}".format(host=host, port=port)

    @app.route("/")
    def root():
        return app.name

    @app.route("/help")
    def help():
        return app.send_static_file(Paths().help)

    try:
        app.run(port=port)
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