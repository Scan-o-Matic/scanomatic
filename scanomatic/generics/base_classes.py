"""Base classes that hook up class with resource classes"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

from scanomatic.io.paths import Paths
from scanomatic.io.app_config import Config


class PathUser(object):

    def __init__(self):

        self._path = Paths() 


class AppConfigUser(object):

    def __init__(self):

        self._appConfig = Config()


class PathAndAppConfigUser(object):

    def __init__(self):

        self._path = Paths() 
        self._appConfig = Config()
