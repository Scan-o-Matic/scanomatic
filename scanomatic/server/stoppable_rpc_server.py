__author__ = 'martin'

from SimpleXMLRPCServer import SimpleXMLRPCServer

import scanomatic.generics.decorators as decorators
import scanomatic.io.logger as logger


class Stoppable_RPC_Server(SimpleXMLRPCServer):

    def __init__(self, *args, **kwargs):

        SimpleXMLRPCServer.__init__(self, *args, **kwargs)
        self.logger = logger.Logger("RPC Server")
        self._keepAlive = True

    def restart(self):

        if self._keepAlive:
            self.stop()

        self.serve_forever(poll_interval=self.timeout)

    def stop(self):

        self._keepAlive = False

    @decorators.threaded
    def serve_forever(self, poll_interval=0.5):

        self.timeout = poll_interval
        while self._keepAlive:
            self.handle_request()