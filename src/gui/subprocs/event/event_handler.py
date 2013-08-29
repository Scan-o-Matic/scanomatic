#!/usr/bin/env python
"""Event Handler that keeps all the Event objects"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import inspect

#
# INTERNAL DEPENDENCIES
#

from src.gui.subprocs.event.event import Event

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

        print "===\n{0}\n{1}\n{2}\nCalled by {3}\n____".format(
            fn, args, kwargs, ">".join(frames[::-1]))

        fn(*args, **kwargs)

    return wrapped

#
# CLASS
#


class EventHandler(object):

    def __init__(self):
        """The Event Handler gathers all subprocess events.

        It invokes all the communications and checks their status.

        :param logger: A logging-object
        """

        self._events = set()

    def addEvent(self, event):
        """Adds (or merges) an event.

        If event is added then the event's request is sent

        :param event: Event to be added
        """

        if not isinstance(event, Event):
            raise TypeError("{0} not an {1}".format(event, Event))

        if not self._mergeEvent(event):

            if event not in self._events:

                self._events.add(event)
                event.sendRequest()

            else:

                raise ValueError("Cannot have duplicate instances in "
                                 "event handler ({0})".format(event))

    def update(self):
        """Runs an update loop through the events.

        Method is good target for timout-calls or such.

        :return: True
        """

        for e in tuple(self._events):

            e.check()
            if e.isDone():
                self._removeEvent(e)

        return True

    def _mergeEvent(self, event):
        """Checks if the event can be merged with other event

        :param event: Event to be merged if possible.
        :return: If event got merged
        """

        for e in self._events:

            if e.isSameRequest(event):

                if not e.hasAllTargets(event):

                    e.addTargets(event)

                return True

        return False

    def _removeEvent(self, event):
        """Removes an event.

        This method does no checks to make sure the event is done!
        It also destroys the event

        :param event: The event to be removed.
        """

        if event in self._events:
            self._events.remove(event)

        del event
