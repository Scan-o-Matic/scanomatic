"""The RPC-server is the master controller of all Scan-o-Matic default
operations.
"""
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
import threading
import time
from SimpleXMLPRCServer import SimpleXMLRPCServer
from ConfigParser import ConfigParser

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
import scanomatic.io.paths as paths

#
# CLASSES
#


class SOM_RPC(object):

    def __init__(self):

        self._logger = logger.Logger("Scan-o-Matic RPC Server")

        self._paths = paths.Paths()

        self._serverCfg = ConfigParser(allow_no_value=True)
        self._serverCfg.readfp(open(self._paths.config_rpc))

        self._admin = self._paths.config_rpc_admin

        self._server = None
        self._running = False

    def _startServer(self):

        if (self._server is not None):
            raise Exception("Server is already running")

        host = self._serverCfg.get('Connection', 'host')
        port = self._serverCfg.get('Connection', 'port')

        self._server = SimpleXMLRPCServer((host, port))

        self._running = True
        self._mainThread = None

        self._logger.info("Server listens to {0}:{1}".format(host, port))

        [self._server.register_function(getattr(self, m), m) for m
         in dir(self) if not(m.startswith("_") or m in
                             self._serverCfg.options('Hidden Methods'))]

    def _main(self):

        while self._running:

            #TODO: Gather information here

            time.sleep(0.21)

        self._server.shutDown()
        self._niceQuitProcesses()

    def _niceQuitProcesses(self):

        #TODO: Ask if everyone is ready to die

        self._shutDownComplete = True

    def serverRestart(self, userID):

        if userID == self._admin:
            self.serverShutDown()
            self.run()

    def serverShutDown(self, userID):

        if userID == self._admin:
            self._running = False
            self._shutDownComplete = False
            while (self._mainThread is not None and
                   self._mainThread.isalive):

                time.sleep(0.05)

            self._server = None

    def run(self):

        if self._running is True:

            raise Exception("Server is already running")

        self._logger.info("Starting server")
        self._startServer()

        self._mainThread = threading.Thread(target=self._main)
        self._mainThread.start()

        self._server.serve_forever()
        self._logger.info("Server serves forever")
