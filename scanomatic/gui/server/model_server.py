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
        'status-server-offline': "Offline",
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

        'status-not-implemented-error':
        """Detailed status not yet re-implemented.""",

        'server-host-label': 'Host:',
        'server-cpu-label': 'CPU',
        'server-mem-label': 'MEM',
        'server-local-label': 'Local',

        'status-server-remote-no-connection':
        "Can't Reach Host",

        'server-shutdown-error':
        "Can't shut down a server that is not running",

        'server-shutdown-warning':
        "This will shut down ALL jobs running on the server. Are you sure?",

        'server-startup-error':
        "A server is already running, can't have two!",

        'server-online-check-time': -1,
        'server-resource-check-time': -1,
        'server-resource-check-spacing': 5,
        'rpc-client': None,
        'server-offline': False,
        'cpu-ok': False,
        'mem-ok': False,
        'up-time': -1,
    }

    _PRESETS_QUEUE = {
        'current-queue': [],
        'rpc-client': None,
        'queue-title': "Jobs Queue on Scan-o-Matic Server",
        'page-title': "Queue",
        'queue-flush': 'Flush',
        'queue-flushed': "Everything in queue removed",
        'queue-flushed-refused': "Refused, probably lacking admin-permissions",
        'queue-removed': "Job removed",
        'queue-removed-refused': "Refused, either job not in queue or insufficient permissions",
        'queue-item-prio': "Priority",
        'queue-item-remove': "Remove",
    }

    def serverOnline(self):

        c = self['rpc-client']
        if c:
            return c.online
        else:
            return False

    def serverLocal(self):

        c = self['rpc-client']
        if c:
            return c.local
        else:
            return True

    def serverPort(self):

        c = self['rpc-client']
        if c:
            return c.port
        else:
            return -1

    def serverHost(self):

        c = self['rpc-client']
        if c:
            return c.host
        return ""

    def serverLaunchChecking(self):

        return self['server-online-check-time'] not in (-1, 0)

    def queueLength(self):

        c = self['rpc-client']
        if c and c.online:
            return len(c.getJobsInQueue())
        return -1

    def jobsLength(self):

        c = self['rpc-client']
        if c and c.online:
            return len(c.getActiveJobs())
        return -1

    def scannersFree(self):

        #c = self['rpc-client']
        return -1
