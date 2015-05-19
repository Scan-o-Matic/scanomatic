__author__ = 'martin'

import time
from flask import Flask
import webbrowser
from threading import Thread

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
    def placeholder():
        return app.send_static_file(Paths().help)

    app.run(port=port)

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