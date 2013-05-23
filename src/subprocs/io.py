#!/usr/bin/env python
"""Communication statements"""
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

import time

#
# INTERNAL DEPENDENCIES
#

import src.resource_logger as resource_logger

#
# EXCEPTIONS
#


class BadFormedMessage(Exception):
    pass

#
# CLASSES
#


class Unbuffered_IO:

    def __init__(self, stream):
        """This class provides and unbuffered IO-writer"""
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, data):

        if isinstance(data, str):
            data = (data, )

        for row in data:

            if not isinstance(row, str):

                row = str(row)

            if not row.endswith(Proc_IO.NEWLINE):

                row += Proc_IO.NEWLINE

            self.write(row)

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

    def close(self):

        self.stream.close()


class Proc_IO(object):

    #MAIN PROC STATUS STATEMENTS
    IS_PAUSED = "__IS_PAUSED__"
    IS_RUNNING = "__IS_RUNNING__"

    #MAIN PROC CONTROL
    PAUSE = "__PAUSE__"
    PAUSING = "__PAUSING__"
    TERMINATE = "__TERMINATE__"
    TERMINATING = "__TERMINATING__"
    UNPAUSE = "__UNPAUSE__"

    #PING
    PING = "__PING__"

    #MAIN PROC INFO REQUESTS
    INFO = "__PARAM__"
    CURRENT = "__CURRENT__"
    TOTAL = "__TOTAL__"
    PROGRESS = "__PROGRESS__"
    REFUSED = "__REFUSED__"
    STATUS = "__STATUS__"

    #COMMUNICATION MESSAGE DECORATORS
    _MESSAGE_START = "__START__"
    _MESSAGE_END = "__DONE__"

    #ERROR/BAD CALL
    UNKNOWN = "__UNKNOWN__"

    #HELPS
    NEWLINE = "\n"
    VALUE_EXTEND = " {0}"

    def __init__(self, send_file_path, recieve_file_path, recieve_pos=None,
                 send_file_state='w', logger=None):

        if logger is None:
            logger = resource_logger.Fallback_Logger()

        self._logger = logger

        unbuffered_send = open(send_file_path, send_file_state, 0)
        self._send_path = send_file_path
        self._send_fh = Unbuffered_IO(unbuffered_send)
        self._recieve_path = recieve_file_path
        self._sending = False
        self._recieve_pos = recieve_pos

    def _set_recieve_pos(self):
        try:
            fh = open(self._recieve_path, 'r')
            fh.read()
            self._recieve_pos = fh.tell()
            fh.close()
        except:
            self._logger.info(
                "No recieve file ('{0}') existing before".format(
                    self._recieve_path))

    def _parse_recieved_and_callback(self, lines, recieve_callback):

        while self._MESSAGE_START in lines and self._MESSAGE_END:

            m_start = lines.index(self._MESSAGE_START)
            m_end = lines.index(self._MESSAGE_END) + len(self._MESSAGE_END)

            if m_start < m_end:
                msg = lines[m_start: m_end]
                recieve_callback(msg)
            else:
                raise BadFormedMessage(msg)

            self._recieve_pos += m_end
            lines = lines[m_end:]

    def get_sending_path(self):

        return self._send_path

    def recieve(self, recieve_callback):
        """Recieves messages and passes them to callback.

        :param recieve_callback: Method that takes a string as parameter
            (the string being the decorated message).
        """
        #Seek to last_pos
        if self._recieve_pos is None:
            self._set_recieve_pos()

        #Recieving
        fh_pos = self._recieve_pos

        try:
            fh = open(self._recieve_path, 'r')
            if fh_pos is not None:
                fh.seek(fh_pos)
            lines = fh.read()
            fh.close()
        except:
            lines = ""
            self._logger.error('{0} could not read {1}'.format(
                self, self._recieve_path))

        #Parsing for messages
        self._parse_recieved_and_callback(lines, recieve_callback)

    def send(self, msg):
        """Safe sending one message at a time"""

        while self._sending is True:
            time.sleep(0.02)

        self._sending = True

        if self._send_fh is None:
            for row in msg:
                print row
        else:
            self._send_fh.writelines(msg)

        self._sending = False

    def decorate(self, msg, timestamp=None):
        """Adds a start and stop row to message"""

        #CREATING STAMP IF NOT PASSED
        if timestamp is None:
            timestamp = time.time()

        #SETTING CORRECT TYPES OF MSG
        if isinstance(msg, str):
            msg = (msg, )
        else:
            msg = tuple(msg)

        #ADDING START ROW WITH STAMP
        decorated_msg = (self._MESSAGE_START +
                         self.VALUE_EXTEND.format(timestamp), ) + msg

        #ADDING END ROW
        decorated_msg += (self._MESSAGE_END, )

        return decorated_msg

    def undecorate(self, msg):
        """Removes start en end rows.

        :return timestamp: The passed stamp of the message
        :return msg: The undecorated message
        """
        if self._MESSAGE_START in msg:

            tmpmsg = msg[msg.index(self._MESSAGE_START) +
                         len(self._MESSAGE_START):]

            try:
                timestamp, undecorated_msg = tmpmsg.split(self.NEWLINE, 1)
            except:
                raise BadFormedMessage("Malformed Message Start:\n{0}".format(
                    msg))

            timestamp = timestamp.strip()

        else:
            raise BadFormedMessage("No Message Start:\n{0}".format(msg))

        try:
            undecorated_msg, overflow = undecorated_msg.split(
                self._MESSAGE_END, 1)
        except:
            raise BadFormedMessage("No Message End:\n{0}".format(msg))

        return timestamp, undecorated_msg

    def close_send_file(self):

        if self._send_fh is not None:
            self._send_fh.close()
            self._send_fh = None
