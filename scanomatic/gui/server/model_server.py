"""The Server Model"""
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


#
# INTERNAL DEPENDENCIES
#

import scanomatic.gui.generic.new_model_generic as new_model_generic


#
# MODEL
#

class Model(new_model_generic.Model):

    _PRESETS_STAGE = {
        'status-server': 'Server Status:',
        'status-server-checking': 'Checking...',
        'status-server-running': 'Connected',
        'status-server-launching': 'Launching...',
        'status-server-local-error': 'Failed to launch!',
        'status-server-remote-no-connection': 'No Connection!',
        'status-queue': "Jobs in queue:",
        'status-jobs': "Jobs running:",
        'status-scanners': "Free scanners:",
        'status-unknown-count': "?",

        'status-local-server-error': """Local server could not be started,
try updating your program""",

        'status-not-implemented-error': """Detailed status not yet
        re-implemented.""",
        'server-online-check-time': -1,
        'rpc-client': None
    }

    def serverOnline(self):

        c = self['rpc-client']
        return c is None and False or c.online

    def serverLocal(self):

        c = self['rpc-client']
        return c is None and False or c.local

    def serverPort(self):

        c = self['rpc-client']
        return c is None and -1 or c.port

    def serverHost(self):

        c = self['rpc-client']
        return c is None and "" or c.host

    def serverLaunchChecking(self):

        return self['server-online-check-time'] not in (-1, 0)

    def queueLength(self):

        c = self['rpc-client']
        if c is not None and c.online:
            return len(c.getJobsInQueue())
        return 0

    def jobsLength(self):

        c = self['rpc-client']
        if c is not None and c.online:
            return len(c.getActiveJobs())
        return 0

    def scannersFree(self):

        #c = self['rpc-client']
        return -1
