__author__ = 'martin'

import socket
import sys
import os

from scanomatic.io.app_config import Config
from scanomatic.generics.singleton import Singleton
import scanomatic.io.logger as logger
from scanomatic.rpc_server.server import Server
from scanomatic.rpc_server.stoppable_rpc_server import Stoppable_RPC_Server
from scanomatic.generics.abstract_model_factory import AbstractModelFactory


def _verify_admin(f):

    def _verify_global_admin(interface_builder, user_id, *args, **kwargs):

        if (user_id == interface_builder._admin):

            return  f(user_id, *args, **kwargs)

        else:

            interface_builder.logger.warning("User {0} unauthorized attempt at accessing {1}".format(user_id, f))
            return False

    return _verify_global_admin


class Interface_Builder(Singleton):

    def __init__(self):

        self._rpc_server = None
        self._som_server = None
        self._logger = logger.Logger("Server Interface Builder")
        self._start_som_server()
        self._start_rpc_server()

    def _start_som_server(self):

        self._som_server = Server()

    def _start_rpc_server(self):

        app_config = Config()
        host = app_config.rpc_host
        port = app_config.rpc_port

        if self._rpc_server is not None:
            raise Exception("Server is already running")

        try:
            self._rpc_server = Stoppable_RPC_Server((host, port), logRequests=False)
        except socket.error:
            self._logger.critical(
                "Sever is already running or the " +
                "port {0} is in use by other program".format(
                    port))

            sys.exit(1)

        self._rpc_server.register_introspection_functions()

        self._logger.info("Server (pid {0}) listens to {1}:{2}".format(
            os.getpid(), host, port))

        for m in dir(self):
            if (m.startswith("_server_")):
                self._rpc_server.register_function(getattr(self, m), m[8:])

    def _remove_rpc_server(self):

        self._rpc_server.stop()
        del self._rpc_server
        self._som_server.logger.info("Server no longer accepting requests")

    def _remove_som_server(self, forceJobsToStop):

        if forceJobsToStop:
            successful_stop = self._som_server.shutdown()
        else:
            successful_stop = self._som_server.safe_shutdown()

        del self._som_server

        return successful_stop

    @_verify_admin
    def _server_shutdown(self, user_id, forceJobsToStop=False):

        self._remove_rpc_server()

        val = self._remove_som_server(forceJobsToStop=forceJobsToStop)

        if val:
            self._som_server.logger.info("Server is shut down")
        else:
            self._som_server.logger.error("Unknown error shutting down Scan-o-Matic server")

        return  val

    @_verify_admin
    def _server_restart(self, user_id, force_jobs_to_stop=False):

        self._remove_rpc_server()

        if self._remove_som_server(forceJobsToStop=force_jobs_to_stop):
            self._start_som_server()
            self._start_rpc_server()
        else:
            return False
        return True


    def _server_get_status(self, user_id=None):

        return AbstractModelFactory.to_dict(self._som_server.get_server_status())

    def _server_get_scanner_status(self, user_id=None):

        pass

    def _server_get_queue_status(self, user_id=None):

        pass

    def _server_get_job_status(self, user_id, job_id):

        pass

    def _server_communicate(self, user_id, job_id, communication, **communication_content):

        pass
