#!/usr/bin/env python
"""Event objects for subprocess communications"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.999"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import time
import inspect
import logging

#
# METHOD
#


def whoCalled(fn):

    def wrapped(*args, **kwargs):
        frames = []
        frame = inspect.currentframe().f_back
        while frame.f_back:
            frames.append(inspect.getframeinfo(frame)[2])
            frame = frame.f_back
        frames.append(inspect.getframeinfo(frame)[2])

        print "==={0}\n{1}\n{2}\n{3}\nCalled by {4}\n____".format(
            time.time(), fn, args, kwargs, ">".join(frames[::-1]))

        fn(*args, **kwargs)

    return wrapped


#
# CLASS
#


class Event(object):

    def __init__(self, requestFunction, responseTargetFunction,
                 responseDefualt, responseTimeOut=None, **requestParameters):

        self._logger = logging.getLogger("Event {0}".format(requestFunction))

        #Request
        self._requestObject = requestFunction.im_self
        self._requestFunction = requestFunction
        self._requestParameters = requestParameters

        self._requestTime = None

        #Response
        self._responseTargetFunctions = [responseTargetFunction]
        self._responseDefualts = {responseTargetFunction: responseDefualt}
        self._responseTimeOuts = {responseTargetFunction: responseTimeOut}

        self._response = None
        self._hasResponded = False

    def isSameRequest(self, otherEvent):
        """Test if to events have identical requests.

        This check both that the request-functions are identical
        and that the passed parameters are the same.

        :param otherEvent: The event that is tested against
        :return: If they are same.
        """

        if not isinstance(otherEvent, Event):
            self._logger.warning("{0} not an {1}".format(otherEvent, Event))
            return False

        return ((self._requestFunction is otherEvent._requestFunction) and
                (self._requestParameters == otherEvent._requestParameters))

    def hasAllTargets(self, otherEvent):
        """Test if all targets in other event are in present event.

        :param otherEvent: The event that is tested
        :return: If otherEvent's targets all exist in persent event
        """

        if not isinstance(otherEvent, Event):
            self._logger.error("{0} not an {1}".format(otherEvent, Event))
            return False

        return set(self._responseTargetFunctions).issuperset(
            otherEvent._responseTargetFunctions)

    def addTargets(self, otherEvent):
        """Adds all targets in otherEvent not present in current event.

        :param otherEvent: The event to add from
        """

        if not isinstance(otherEvent, Event):
            self._logger.error("{0} not an {1}".format(otherEvent, Event))
            return False

        if otherEvent is self:

            self._logger.error("Illegal Operation, cannot add self to self")
            return False

        for responseTargetFunction in otherEvent._responseTargetFunctions:

            if responseTargetFunction not in self._responseTargetFunctions:

                self._responseTargetFunctions.append(
                    responseTargetFunction)

                self._responseDefualts[responseTargetFunction] = \
                    otherEvent._responseDefualts[responseTargetFunction]

                self._responseTimeOuts[responseTargetFunction] = \
                    otherEvent._responseTimeOuts[responseTargetFunction]

    def sendRequest(self):
        """Sends the request, should be called by the event-handler"""

        if self._requestTime is None:

            self._requestTime = time.time()
            self._requestFunction(self.recieveResponse,
                                  **self._requestParameters)

        else:

            self.logger.error("Request can only be sent once!")

    def recieveResponse(self, *response):
        """Callback for the communicator's responses

        :param *response: An argument list of response-values
        """

        if len(self._responseTargetFunctions) == 0:

            self._logger.error(
                "Recieved response without anyone to send it to")

        self._response = response
        self._hasResponded = True

    def isDone(self):
        """If all is said and done and event can be trashed (by event handler)

        :return: Done-state
        """

        return (self._hasResponded and
                (len(self._responseTargetFunctions) == 0))

    def _removeTarget(self, responseTargetFunction):
        """Removes a response target from event

        :param responseTargetFunction: The target function
        """

        del self._responseTargetFunctions[
            self._responseTargetFunctions.index(responseTargetFunction)]

        del self._responseDefualts[responseTargetFunction]
        del self._responseTimeOuts[responseTargetFunction]

    def check(self):
        """Checks with communicator and deals with callbacks and timeouts"""

        #Make the communicator check its communications
        self._requestObject.update()

        if self._hasResponded and self._response is not None:
            for responseTargetFunction in self._responseDefualts.keys():

                responseTargetFunction(self._requestObject, *self._response)

                self._removeTarget(responseTargetFunction)

        elif self._hasResponded is False:

            checkTime = time.time()

            for responseTargetFunction in self._responseDefualts.keys():

                timeOut = self._responseTimeOuts[responseTargetFunction]

                if ((timeOut is not None) and
                        (checkTime > timeOut + self._requestTime)):

                    responseTargetFunction(
                        self._requestObject,
                        self._responseDefualts[responseTargetFunction])

                    self._removeTarget(responseTargetFunction)

                    self._hasResponded = True
