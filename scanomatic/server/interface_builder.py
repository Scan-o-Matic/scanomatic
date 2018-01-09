import socket
import sys
import os
from time import sleep
from functools import wraps

from scanomatic.io.app_config import Config
from scanomatic.generics.singleton import SingeltonOneInit
import scanomatic.io.logger as logger
from scanomatic.server.server import Server
from scanomatic.server.stoppable_rpc_server import Stoppable_RPC_Server
import scanomatic.generics.decorators as decorators
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from scanomatic.models.factories.features_factory import FeaturesFactory
from scanomatic.models.factories.compile_project_factory import CompileProjectFactory
import scanomatic.models.rpc_job_models as rpc_job_models
from scanomatic.io.rpc_client import sanitize_communication

_SOM_SERVER = None
""":type : scanomatic.server.server.Server"""
_RPC_SERVER = None


def _verify_admin(f):

    @wraps(f)
    def _verify_global_admin(interface_builder, user_id, *args, **kwargs):

        global _SOM_SERVER
        global _RPC_SERVER

        if _SOM_SERVER and user_id == _SOM_SERVER.admin:

            return f(interface_builder, user_id, *args, **kwargs)

        else:

            _RPC_SERVER.logger.warning("User {0} unauthorized attempt at accessing {1}".format(user_id, f))
            return False

    return _verify_global_admin


def _report_invalid(logger, factory, model, title):
    """

    :type logger: scanomatic.io.logger.Logger
    :type factory: scanomatic.generics.abstract_model_factory.AbstractModelFactory
    :type model: scanomatic.generics.model.Model
    :return: None
    """

    for param in factory.get_invalid_names(model):

        logger.warning("{title} got invalid parameter {param} value '{value}'".format(
           title=title, param=param, value=model[param]))


class InterfaceBuilder(SingeltonOneInit):

    def __one_init__(self):

        self.logger = logger.Logger("Server Manager")
        self._start_som_server()
        self._start_rpc_server()

    @staticmethod
    def _start_som_server():

        global _SOM_SERVER
        if _SOM_SERVER is None:
            _SOM_SERVER = Server()
            _SOM_SERVER.start()
        else:
            _SOM_SERVER.logger.warning("Attempt to launch second instance of server")

    def _start_rpc_server(self):

        global _RPC_SERVER
        app_config = Config()
        host = app_config.rpc_server.host
        port = app_config.rpc_server.port

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

        if _RPC_SERVER:
            _RPC_SERVER.stop()
        else:
            self.logger.warning("Trying to stop a server that is not running")
        _RPC_SERVER = None

        if _SOM_SERVER:
            logger_instance = _SOM_SERVER.logger
        else:
            logger_instance = self.logger

        logger_instance.info("Server no longer accepting requests")

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
            self.logger.info("Server is shutting down")
        else:
            self.logger.error("Unknown error shutting down Scan-o-Matic server")

        return sanitize_communication(val)

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
        """Gives a dictionary of the servers status

        Kwargs:
            user_id (str):   The ID of the user requesting status
                            The full purpose of userID is to maintain
                            method interface for all exposed RPC methods

        Returns:
            dictionary. Key value pairs for the different aspects of the
                        server status.
        """
        global _SOM_SERVER
        return sanitize_communication(_SOM_SERVER.get_server_status())

    def _server_get_queue_status(self, user_id=None):

        global _SOM_SERVER
        return sanitize_communication(_SOM_SERVER.queue.status)

    def _server_get_job_status(self, user_id=None):
        """Gives a list or statuses.

        First entry is always the status of the server followed by
        an item for each job.

        Kwargs:
            userID (str):   The ID of the user requesting status
                            The full purpose of userID is to maintain
                            method interface for all exposed RPC methods

        Returns:
            list.   Each item in the list is a dictionary.
                    For information about the job dictionaries and
                    their structure, see ``self.getStatus``.
                    The first item of the list will be a dictionary
                    containing general information about the server.::

            ServerUpTime:  (str) Either the message 'Server Not Running'
                           or a string with like "XXh, YYm, ZZ.ZZs"
                           expressing the time that the server has been
                           running.

        """
        global _SOM_SERVER
        return sanitize_communication(_SOM_SERVER.jobs.status)

    @_verify_admin
    def _server_communicate(self, user_id, job_id, communication, communication_content={}):
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
                email:  Add recipient of notifications to a job
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
        """:type : scanomatic.models.rpc_job_models.RPCjobModel"""

        if job is not None:
            if job.status is rpc_job_models.JOB_STATUS.Queued:
                return sanitize_communication(_SOM_SERVER.queue.remove(job))
            else:
                try:
                    ret = _SOM_SERVER.jobs[job].pipe.send(communication, **communication_content)
                    self.logger.info("The job {0} got message {1}".format(
                        job.id, communication))
                    return sanitize_communication(ret)
                except (AttributeError, TypeError):
                    self.logger.error("The job {0} has no valid call {1} with payload {2}".format(
                        job.id, communication, communication_content))
                    return False

        else:
            self.logger.error("The job {0} is not running".format(job_id))
            return False

    @_verify_admin
    def _server_create_compile_project_job(self, user_id, compile_project_model):

        global _SOM_SERVER

        compile_project_model = CompileProjectFactory.create(**compile_project_model)

        if not CompileProjectFactory.validate(compile_project_model):

            _report_invalid(
                _SOM_SERVER.logger,
                CompileProjectFactory,
                compile_project_model,
                "Request compile project")
            return False

        return sanitize_communication(_SOM_SERVER.enqueue(compile_project_model, rpc_job_models.JOB_TYPE.Compile))

    @_verify_admin
    def _server_remove_from_queue(self, user_id, job_id):
        """Removes job from queue

        Parameters
        ==========

        user_id : str
            The ID of the user requesting to create a job.
            This must match the current ID of the server admin or
            the request will be refused.
            **NOTE**: If using a rpc_client from scanomatic.io the client
            will typically prepend this parameter

        jobID : str
            The ID of job to be removed

        Returns
        =======

        bool
            Success status

        See Also
        ========

        flush_queue
            Remove all queued jobs
        """

        global _SOM_SERVER
        return sanitize_communication(_SOM_SERVER.queue.remove(job_id))

    @_verify_admin
    def _server_flush_queue(self, user_id):
        """Clears the queue

        Parameters
        ==========

        user_id : str
            The ID of the user requesting to create a job.
            This must match the current ID of the server admin or
            the request will be refused.
            **NOTE**: If using a rpc_client from scanomatic.io the client
            will typically prepend this parameter

        Returns
        =======

        bool
            Success status

        See Also
        ========

        remove_from_queue
            Remove individual job from queue
        """

        global _SOM_SERVER
        queue = _SOM_SERVER.queue

        while queue:
            job = queue.get_highest_priority()
            queue.remove(job)

        return True

    @_verify_admin
    def _server_create_analysis_job(self, user_id, analysis_model):
        """Enques a new analysis job.

        Parameters
        ==========

        userID : str
            The ID of the user requesting to create a job.
            This must match the current ID of the server admin or
            the request will be refused.
            **NOTE**: If using a rpc_client from scanomatic.io the client
            will typically prepend this parameter

        analysis_model : dict
            A dictionary representation of a scanomatic.models.analysis_model.AnalysisModel

        Returns
        =======

        bool
            Success of putting job in queue
        """

        global _SOM_SERVER

        analysis_model = AnalysisModelFactory.create(**analysis_model)
        if not AnalysisModelFactory.validate(analysis_model):
            _report_invalid(_SOM_SERVER.logger, AnalysisModelFactory, analysis_model, "Request analysis")
            return False

        return sanitize_communication( _SOM_SERVER.enqueue(analysis_model, rpc_job_models.JOB_TYPE.Analysis))

    @_verify_admin
    def _server_create_feature_extract_job(self, user_id, feature_extract_model):
        """Enques a new feature extraction job.

        Parameters
        ==========

        userID : str
            The ID of the user requesting to create a job.
            This must match the current ID of the server admin or
            the request will be refused.
            **NOTE**: If using a rpc_client from scanomatic.io the client
            will typically prepend this parameter


        feature_extract_model : dict
            A dictionary representation of scanomatic.models.features_model.FeaturesModel

        Returns
        =======
            bool.   ``True`` if job request was successfully enqueued, else
                    ``False``
        """

        global _SOM_SERVER

        feature_extract_model = FeaturesFactory.create(**feature_extract_model)
        if not FeaturesFactory.validate(feature_extract_model):
            _report_invalid(_SOM_SERVER.logger, FeaturesFactory, feature_extract_model, "Request feature extraction")
            return False

        return sanitize_communication(_SOM_SERVER.enqueue(feature_extract_model, rpc_job_models.JOB_TYPE.Features))
