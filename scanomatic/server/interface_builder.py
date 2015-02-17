__author__ = 'martin'

import socket
import sys
import os
from time import sleep

from scanomatic.io.app_config import Config
from scanomatic.generics.singleton import Singleton
import scanomatic.io.logger as logger
from scanomatic.server.server import Server
from scanomatic.server.stoppable_rpc_server import Stoppable_RPC_Server
from scanomatic.generics.abstract_model_factory import AbstractModelFactory
import scanomatic.generics.decorators as decorators

_SOM_SERVER = None
_RPC_SERVER = None


def _verify_admin(f):
    def _verify_global_admin(interface_builder, user_id, *args, **kwargs):

        global _SOM_SERVER
        global _RPC_SERVER

        if _SOM_SERVER and user_id == _SOM_SERVER.admin:

            return f(interface_builder, user_id, *args, **kwargs)

        else:

            _RPC_SERVER.logger.warning("User {0} unauthorized attempt at accessing {1}".format(user_id, f))
            return False

    return _verify_global_admin


class Interface_Builder(Singleton):

    def __init__(self):

        self.logger = logger.Logger("Server Manager")
        self._start_som_server()
        self._start_rpc_server()

    @staticmethod
    def _start_som_server():

        global _SOM_SERVER
        if _SOM_SERVER is None:
            _SOM_SERVER = Server()
        else:
            _SOM_SERVER.logger.warning("Attempt to launch second instance of server")

    def _start_rpc_server(self):

        global _RPC_SERVER
        app_config = Config()
        host = app_config.rpc_host
        port = app_config.rpc_port

        if _RPC_SERVER is not None and _RPC_SERVER.running:
            _RPC_SERVER.logger.warning("Attempt to launch second instance of server")
            return False

        try:
            _RPC_SERVER = Stoppable_RPC_Server((host, port), logRequests=False)
        except socket.error:
            self.logger.critical(
                "Sever is already running or the " +
                "port {0} is in use by other program".format(
                    port))
            self._remove_som_server(False)
            sys.exit(1)

        _RPC_SERVER.register_introspection_functions()

        _RPC_SERVER.logger.info("Server (pid {0}) listens to {1}:{2}".format(
            os.getpid(), host, port))

        for m in dir(self):
            if m.startswith("_server_"):
                _RPC_SERVER.register_function(getattr(self, m), m[8:])

        _RPC_SERVER.serve_forever()

    @decorators.threaded
    def _remove_rpc_server(self):

        global _RPC_SERVER
        global _SOM_SERVER

        _RPC_SERVER.stop()
        _RPC_SERVER = None

        if _SOM_SERVER:
            L = _SOM_SERVER.logger
        else:
            L = self.logger

        L.info("Server no longer accepting requests")

    @staticmethod
    def _remove_som_server(wait_for_jobs_to_stop):

        global _SOM_SERVER
        if wait_for_jobs_to_stop:
            successful_stop = _SOM_SERVER.shutdown()
        else:
            successful_stop = _SOM_SERVER.safe_shutdown()

        return successful_stop

    @_verify_admin
    def _server_shutdown(self, user_id, wait_for_jobs_to_stop=False):

        self._remove_rpc_server()

        val = self._remove_som_server(wait_for_jobs_to_stop=wait_for_jobs_to_stop)

        if val:
            self.logger.info("Server is shut down")
        else:
            self.logger.error("Unknown error shutting down Scan-o-Matic server")

        return val

    @_verify_admin
    def _server_restart(self, user_id, wait_for_jobs_to_stop=False):

        self._remove_rpc_server()

        self._restart_server_thread(wait_for_jobs_to_stop)

        return True

    @decorators.threaded
    def _restart_server_thread(self, wait_for_jobs_to_stop):

        global _SOM_SERVER
        self._remove_som_server(wait_for_jobs_to_stop=wait_for_jobs_to_stop)

        while _SOM_SERVER.serving:
            sleep(0.1)

        del _SOM_SERVER
        _SOM_SERVER = None

        self._start_som_server()
        self._start_rpc_server()

    @staticmethod
    def _server_get_status(user_id=None):

        global _SOM_SERVER
        return AbstractModelFactory.to_dict(_SOM_SERVER.get_server_status())

    def _server_get_scanner_status(self, user_id=None):

        pass

    def _server_get_queue_status(self, user_id=None):

        pass

    def _server_get_job_status(self, user_id, job_id):

        pass

    @_verify_admin
    def _server_communicate(self, user_id, job_id, communication, **communication_content):
        """Used to communicate with active jobs.

        Args:
            userID (str):   The ID of the user, this must match the
                            current ID of the server admin or request
                            will be refused.

            jobID (str):    The ID for the job to communicate with.

            title (str):    The name, as understood by the job, for what you
                            want to do. The following are universally
                            understood::

                setup:  Passing settings before the job has been started.
                        This ``title`` is preferrably coupled with ``kwargs``
                        while the other universally used titles have no use
                        for extra parameters.
                start:  Starting the job's execution
                pause:  Temporarily pausing the job
                resume: Temporarily resuming the job
                stop:   Stopping the job
                status: Requesting that the job sends back the current status

            kwargs (dict):  Extra parameters to send with the communication.

        Returns:

            bool.   ``True`` if communication was allowed (user was admin and
                    title was accepted imperative) else ``False``
        """

        global _SOM_SERVER
        job = _SOM_SERVER.get_job(job_id)

        if job is not None:
            try:
                ret = job.pipe.send(communication, **communication_content)
                self.logger.info("The job {0} got message {1}".format(
                    job.identifier, communication))
                return ret
            except AttributeError:
                self.logger.error("The job {0} has no valid call {1}".format(
                    job.identifier, communication))
                return False

        else:
            self.logger.error("The job {0} is not running".format(job_id))
            return False

    @_verify_admin
    def _server_reestablish_process(self, user_id, job_id, label, job_type, pid):
        """Interface for orphaned daemons to re-gain contact with server.

        Parameters
        ==========

        userID : str
            The ID of the user requesting to create a job.
            This must match the current ID of the server admin or
            the request will be refused.
            **NOTE**: If using a rpc_client from scanomatic.io the client
            will typically prepend this parameter

        jobID : str

            The job identifier of the job that wants to regain contact.
            This job must be known to the server

        jobType: int

            The type of job the job is.

        label : str

            User-friendly string with info about the job

        pid : int

            The process id of the orphaned daemon

        Returns
        =======

        multiprocessing.Connection or False
            Returns the part of the pipe used by the child-process if
            re-establishment is allowed, else False

        """
        pass


        # if jobID in self._jobs:
        #
        # return self._jobs.fakeProcess(jobID, label, jobType, pid)
        #
        # else:
        #
        # self._logger.warning(
        #         "Unknown job "+
        #         "'{0}'({1}, pid {2}) tried to claim it exists".format(
        #             label, jobID, pid))
        #
        #     return False