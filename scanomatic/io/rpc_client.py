import xmlrpclib
import enum
from subprocess import Popen
import socket
from types import StringTypes
from types import GeneratorType, TypeType

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.app_config as app_config
import scanomatic.io.logger as logger

#
# METHODS
#


def sanitize_communication(obj):

    if isinstance(obj, dict):
        return {str(k): sanitize_communication(v) for k, v in obj.iteritems() if v is not None}
    elif isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, set):
        return type(obj)(False if v is None else sanitize_communication(v) for v in obj)
    elif isinstance(obj, enum.Enum):
        return obj.name
    elif isinstance(obj, GeneratorType):
        return tuple(False if v is None else sanitize_communication(v) for v in obj)
    elif obj is None:
        return False
    elif isinstance(obj, TypeType):
        return str(obj)
    else:
        return obj


def get_client(host=None, port=None, admin=False, log_level=None):

    appCfg = app_config.Config()
    if port is None:
        port = appCfg.rpc_server.port
    if host is None:
        host = appCfg.rpc_server.host

    cp = _ClientProxy(host, port, log_level=log_level)
    if admin:

        cp.userID = appCfg.rpc_server.admin

    return cp

#
# CLASSES
#


class _ClientProxy(object):

    def __init__(self, host, port, userID=None, log_level=None):

        self._logger = logger.Logger("Client Proxy")
        if log_level is not None:
            self._logger.level = log_level
        self._userID = userID
        self._adminMethods = ('communicateWith',
                              'createFeatureExtractJob',
                              'createAnalysisJob',
                              'removeFromQueue',
                              'reestablishMe',
                              'flushQueue',
                              'serverRestart',
                              'serverShutDown')

        self._client = None
        self._host = None
        self._port = None

        self.host = host
        self.port = port

    def __getattr__(self, key):

        if key == 'launch_local':
            if self.online is False and self.local:
                self._logger.info("Launching new local server")
                return lambda: Popen(["scan-o-matic_server"])
            else:
                return lambda: self._logger.warning("Can't launch because server is {0}".format(['not local', 'online'][self.online]))

        elif key in self._allowedMethods():
            m = self._userIDdecorator(getattr(self._client, key))
            m.__doc__ = (self._client.system.methodHelp(key) +
                         ["", "\n\nNOTE: userID is already supplied"][
                             self._userID is not None])
            return m
        else:
            raise AttributeError("Client doesn't support attribute {0}".format(
                key))

    def __dir__(self):

        return list(self._allowedMethods())

    def _setupClient(self):

        if (self._host is None or self._port is None):
            self._client = None
            self._logger.info("No client active")
        else:
            address = "{0}:{1}/".format(self._host, self._port)
            self._logger.info("Communicates with '{0}'".format(address))
            self._client = xmlrpclib.ServerProxy(address)

    def _userIDdecorator(self, f):

        def _wrapped(*args, **kwargs):

            if self._userID is not None:
                args = (self._userID,) + args

            args = sanitize_communication(args)
            kwargs = sanitize_communication(kwargs)

            self._logger.debug("Sanitized args {0} and kwargs {1}".format(args, kwargs))

            return f(*args, **kwargs)

        return _wrapped

    def _allowedMethods(self):

        retTup = tuple()

        if not(self._client is None or
                hasattr(self._client, "system") is False):

            try:
                retTup = (v for v in self._client.system.listMethods() if
                          not v.startswith("system.") and not (
                          self._userID is None and v in self._adminMethods))
            except socket.error:
                self._logger.warning("Connection Refused for '{0}:{1}'".format(
                    self.host, self.port))
                return ("launch_local",)

        return retTup

    @property
    def working_on_job_or_has_queue(self):

        return self.online and (self.get_job_status() or self.get_queue_status())

    @property
    def online(self):

        if self._client is not None:
            try:
                return bool(dir(self._client.system.listMethods()))
            except socket.error:
                return False
        return False

    @property
    def local(self):

        return "127.0.0.1" in self.host or "localhost" in self.host

    @property
    def userID(self):

        return self._userID

    @userID.setter
    def userID(self, value):

        self._userID = value

    @property
    def host(self):
        return self._host

    @host.setter
    def host(self, value):

        if not isinstance(value, StringTypes):
            value = str(value)

        value = "{0}{1}".format(
            ['', 'http://'][not value.startswith('http://')], value)

        self._host = value
        self._setupClient()

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, value):

        self._port = value
        self._setupClient()
