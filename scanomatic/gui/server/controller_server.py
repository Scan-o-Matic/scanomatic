"""The Server Controller"""
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

import time
from subprocess import Popen

#
# INTERNAL DEPENDENCIES
#

#Own Model and View
import model_server
import view_server

#Other modules
import scanomatic.gui.generic.controller_generic as controller_generic
import scanomatic.io.rpc_client as rpc_client


class Controller(controller_generic.Controller):

    def __init__(self, parent=None):

        model = model_server.Model.LoadStageModel()

        model['rpc-client'] = rpc_client.get_client(admin=True)

        view = view_server.Server_Status(model, self)

        super(Controller, self).__init__(parent, view=view, model=model)

    def update(self, *args):

        model = self._model

        if not model['server-offline']:

            if not model['serverOnline']:

                if model['server-online-check-time'] == -1:
                    model['server-online-check-time'] = time.time()
                    Popen('scan-o-matic_server')  # , stderr=PIPE, stdout=PIPE)
                    self._logger.info("Server Launched Attempted")

                elif model['server-online-check-time'] < -1:
                    if time.time() + model['server-online-check-time'] > 30:
                        self._view.error(
                            model['status-local-server-error'])
                        model['server-online-check-time'] = 0
                elif model['server-online-check-time'] == 0:
                    pass
                elif time.time() - model['server-online-check-time'] > 30:
                    model['server-online-check-time'] = -time.time()
            else:
                model['server-online-check-time'] = -1

    def connected(self):

        return self._model['serverOnline']

    def shutDown(self):

        model = self._model

        if (not model['serverOnline']):

            self._view.error(model['server-shutdown-error'])

        elif (self._view.warning(model['server-shutdown-warning'], yn=True)):

            model['server-offline'] = True
            model['rpc-client'].serverShutDown()

    def startUp(self):

        model = self._model
        if (self._model['serverOnline']):
            self._view.error(model['server-startup-error'])
        else:
            model['server-online-check-time'] = -1
            model['server-offline'] = False

    def addExtractionJob(self, path, tag):

        model = self._model
        if hasattr(model['rpc-client'], 'createFeatureExtractJob'):

            return model['rpc-client'].createFeatureExtractJob(path, tag)

        return False
