import os
import time
from subprocess import Popen
#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
import scanomatic.io.rpc_client as rpc_client

#
# CLASSES
#


class _PipeEffector(object):

    REQUEST_ALLOWED = "__ALLOWED_CALLS__"

    def __init__(self, pipe, loggerName="Pipe effector"):

        self._logger = logger.Logger(loggerName)

        #The actual communications object
        self._pipe = pipe

        #Calls this side accepts
        self._allowedCalls = dict()

        #Calls that the other side will accept according to other side
        self._allowedRemoteCalls = None

        #Flag indicating if other side is missing
        self._hasContact = True

        #Sends that faild get stored here
        self._sendBuffer = []

        #Calls that should trigger special reaction if pipe is not working
        #Reaction will depend on if server or client side
        self._failVunerableCalls = []

        self._pid = os.getpid()

    def setFailVunerableCalls(self, *calls):

        self._failVunerableCalls = calls

    def setAllowedCalls(self, allowedCalls):
        """Allowed Calls must be iterable with a get item function
        that understands strings"""

        self._allowedCalls = allowedCalls
        self._sendOwnAllowedKeys()

    def add_allowed_calls(self, additional_calls):

        self._allowedCalls.update(additional_calls)
        self._sendOwnAllowedKeys()

    def _sendOwnAllowedKeys(self):

        self._logger.info("Informing other side of pipe about my allowed calls")
        self.send(self.REQUEST_ALLOWED, *self._allowedCalls.keys())

    def poll(self):

        while self._hasContact and self._pipe.poll():

            response = None
            try:
                dataRecvd = self._pipe.recv()
            except EOFError:
                self._logger.warning("Lost contact in pipe")
                self._hasContact = False
                return

            self._logger.debug("Pipe recieved {0}".format(dataRecvd))

            try:
                request, args, kwargs = dataRecvd
            except (IndexError, TypeError):

                self._logger.error(
                    "Recieved a malformed data package '{0}'".format(dataRecvd))


            if request == self.REQUEST_ALLOWED:
                self._logger.info("Got information about other side's allowed calls '{0}'".format(args))
                if self._allowedRemoteCalls is None:
                    self._sendOwnAllowedKeys()
                    self._allowedRemoteCalls = args
            else:
                if self._allowedCalls is None:
                    self._logger.info("No allowed calls")
                    self._sendOwnAllowedKeys()
                elif request in self._allowedCalls:
                    response = self._allowedCalls[request](*args, **kwargs)
                else:
                    self._logger.warning("Request {0} not an allowed call".format(request))
                    self._logger.info("Allowed calls are '{0}'".format(self._allowedCalls.keys()))

            try:
                if response not in (None, True, False):
                    if (isinstance(response, dict) and
                            request == "status"):
                        self.send(request, **response)
                        self._logger.info("Sent status response {0}".format(response))
                    else:
                        self.send(response[0], *response[1], **response[2])
                        self._logger.info("Sent response {0}".format(response))

                else:
                    self._logger.debug("No response to request")

            except Exception, e:

                self._logger.error("Could not send response '{0}' ({1})".format(
                    response, (e, e.message)))

    def _failSend(self, callName, *args, **kwargs):
        """Stores send request in buffer to be sent upon new connection

        If `callName` exists in buffer, it is replaced by the newer send
        request with the same `callName`

        Parameters
        ==========

        callName : str
            Identification string for the type of action or information
            requested or passed through the sent objects

        *args, **kwargs: objects, optional
            Any serializable objects to be passed allong

        Returns
        =======

        bool
            Success status
        """
        for i, (cN, a, kw) in enumerate(self._sendBuffer):
            if cN == callName:
                self._sendBuffer[i] = (callName, args, kwargs)
                return True 

        self._sendBuffer.append((callName, args, kwargs))
        return True

    def send(self, callName, *args, **kwargs):

        if self._hasContact and self._sendBuffer:
            while self._sendBuffer:
                cN, a, kw = self._sendBuffer.pop()
                if not self._send(cN, *a, **kw) and not self._hasContact:
                    self._failSend(cN, a, kw)
                    break

        success = self._send(callName, *args, **kwargs) 

        if not success and not self._hasContact:

            self._failSend(callName, *args, **kwargs)
             
        return success

    def _send(self, callName, *args, **kwargs):

        if (self._allowedRemoteCalls is None or
                callName == self.REQUEST_ALLOWED or
                callName in self._allowedRemoteCalls):

            try:
                self._pipe.send((callName, args, kwargs))
            except Exception, e:
                self._logger.warning("Failed to send {0} ({1})".format(
                    (callName, args, kwargs), (e, e.message)))
                self._hasContact = False
                return False

            return True

        else:
            self._logger.warning("Other side won't accept '{0}'. ".format(
                callName) + "Known calls are '{0}'".format(
                    self._allowedRemoteCalls))
            return False


class ParentPipeEffector(_PipeEffector):

    def __init__(self, pipe):

        super(ParentPipeEffector, self).__init__(
            pipe, loggerName="Parent Pipe Effector")

        self._status = {'pid': self._pid}

        self._allowedCalls['status'] = self._setStatus

    @property
    def status(self):

        return self._status

    def _setStatus(self, *args, **kwargs):

        if 'pid' not in kwargs:
            kwargs['pid'] = self._pid

        self._status = kwargs


class ChildPipeEffector(_PipeEffector):

    def __init__(self, pipe, procEffector=None):

        """

        :type procEffector: scanomatic.server.proc_effector.ProcessEffector
        """
        super(ChildPipeEffector, self).__init__(
            pipe, loggerName="Child Pipe Effector")

        if (procEffector is None):
            self._procEffector = None
            self._allowedCalls = {}
        else:
            self.procEffector = procEffector

    @property
    def keepAlive(self):

        return True if self._procEffector is None else self._procEffector.keep_alive

    def _failSend(self, callName, *args, **kwargs):

        #Not loose calls
        super(ChildPipeEffector, self)._failSend(callName, *args, **kwargs)

        if (callName in self._failVunerableCalls):

            rC = rpc_client.get_client(admin=True)

            if not rC.online:

                self._logger.info("Re-booting server process")
                Popen('scan-o-matic_server')
                time.sleep(2)

            if rC.online:

                pipe = rC.reestablishMe(
                    self.procEffector.label,
                    self.procEffector.label,
                    self.procEffector.TYPE,
                    os.getpid())

                if (pipe is False):

                    self._logger.critical(
                        "Server refused to acckowledge me, " +
                        "nothing left to do but die")
                    self.procEffector.stop()
                    return False

                else:
                    self._pipe = pipe
                    self._hasContact = True
        
        return True
    
    @property
    def procEffector(self):

        return self._procEffector

    @procEffector.setter
    def procEffector(self, procEffector):

        """

        :type procEffector: scanomatic.server.proc_effector.ProcessEffector
        """
        self._procEffector = procEffector
        self.setAllowedCalls(procEffector.allow_calls)
        self.setFailVunerableCalls(*procEffector.fail_vunerable_calls)
        procEffector.pipe_effector = self

    def sendStatus(self, status):

        self.send('status', **status)
