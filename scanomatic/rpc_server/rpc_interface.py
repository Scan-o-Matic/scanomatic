__author__ = 'martin'

import socket
import sys
import os
from SimpleXMLRPCServer import SimpleXMLRPCServer

from scanomatic.io.app_config import Config
from scanomatic.generics.singleton import Singleton
import scanomatic.io.logger as logger
from scanomatic.rpc_server.server import Server


class RPC_Interface(Singleton):

    def __init__(self):

        self._rpc_server = None
        self._server = Server()
        self._logger = logger.Logger("RPC Server")
        self._startServer()

    def _startServer(self):

        app_config = Config()
        host = app_config.rpc_host
        port = app_config.rpc_port

        if self._rpc_server is not None:
            raise Exception("Server is already running")

        try:
            self._rpc_server = SimpleXMLRPCServer((host, port), logRequests=False)
        except socket.error:
            self._logger.critical(
                "Sever is already running or the " +
                "port {0} is in use by other program".format(
                    port))

            sys.exit(1)

        self._rpc_server.register_introspection_functions()

        self._running = True
        self._mainThread = None

        self._logger.info("Server (pid {0}) listens to {1}:{2}".format(
            os.getpid(), host, port))

        for m in dir(self):
            if not(m.startswith("_")):
                self._rpc_server.register_function(getattr(self, m), m)

    def server_shutdown(self, user_id, forceJobsToStop=False):

        if forceJobsToStop:
            return self._server.shutdown(user_id)
        else:
            return self._server.safe_shutdown(user_id)

    def get_status(self, user_id=None):

        pass

    def get_scanner_status(self, user_id=None):

        pass

    def get_queue_status(self, user_id=None):

    def get_job_status(self, user_id, job_id):

        pass

    def communicate(self, user_id, job_id, communication_type, **content):

        pass


