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
