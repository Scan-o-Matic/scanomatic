"""Module for obtaining a valid rpc-client"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import xmlrpclib
from functools import partial

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.app_config as app_config
import scanomatic.io.logger as logger

#
# METHODS
#


def get_client(host=None, port=None, admin=False):

    appCfg = app_config.Config()
    if (port is None):
        port = appCfg.rpc_port
    if (host is None):
        host = appCfg.rpc_host

    cp = _ClientProxy(host, port)
    if admin:

        cp.userID = appCfg.rpc_admin

    return cp

#
# CLASSES
#


class _ClientProxy(object):

    def __init__(self, host, port, userID=None):

        self._logger = logger.Logger("Client Proxy")
        self._userID = userID
        self._adminMethods = ('communicateWith',
                              'createFeatureExtractJob',
                              'serverRestart',
                              'serverShutDown')

        self._client = None
        self._host = None
        self._port = None

        self.host = host
        self.port = port

    def __getattr__(self, key):

        if key in self._allowedMethods():
            return self._userIDdecorator(getattr(self._client, key))
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

        if self._userID is not None:
            return partial(f, self._userID)
        return f

    def _allowedMethods(self):

        return (v for v in self._client.system.listMethods() if
                not v.startswith("system.") and not (
                self._userID is None and v in self._adminMethods))

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

        if not isinstance(value, str):
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
