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
# CLASSES
#


class SUBPROC_COMMUNICATIONS(object):

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
    PING = "__ECHO__"

    #MAIN PROC INFO REQUESTS
    INFO = "__PARAM__"
    CURRENT = "__CURRENT__"
    TOTAL = "__TOTAL__"
    PROGRESS = "__PROGRESS__"
    REFUSED = "__REFUSED__"
    STATUS = "__STATUS__"

    #END OF COMMUNICATION
    COMMUNICATION_END = "__DONE__"

    #ERROR/BAD CALL
    UNKNOWN = "__UNKNOWN__"

    #HELPS
    NEWLINE = "\n"
    VALUE_EXTEND = " {0}"
