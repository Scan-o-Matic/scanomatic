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

#
# METHODS
#


def get_client(host="127.0.0.1", port=None, admin=False):

    appCfg = app_config.Config()
    if (port is None):
        #self._serverCfg = ConfigParser(allow_no_value=True)
        #self._serverCfg.readfp(open(self._paths.config_rpc))
        port = appCfg.rpc_port

    sClient = xmlrpclib.ServerProxy("{0}{1}:{2}/".format(
        ['', 'http://'][not host.startswith('http://')], host, port))

    cp = _ClientProxy(sClient)
    if admin:

        cp.userID = appCfg.rpc_admin

    return cp

#
# CLASSES
#


class _ClientProxy(object):

    def __init__(self, serverClient, userID=None):

        self._userID = userID
        self._adminMethods = ('communicateWith',
                              'createFeatureExtractJob',
                              'serverRestart',
                              'serverShutDown')

        self._client = serverClient

    def __getattr__(self, key):

        if key in self._allowedMethods():
            return self._userIDdecorator(getattr(self._client, key))
        else:
            raise AttributeError("Client doesn't support attribute {0}".format(
                key))

    def __dir__(self):

        return list(self._allowedMethods())

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
