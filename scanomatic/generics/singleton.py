"""Singleton class"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


class Singleton(object):

    _INSTANCE = None

    def __new__(cls, *args):

        if cls is Singleton:

            if cls._INSTANCE is None:

                cls._INSTANCE = super(Singleton, cls).__new__(Singleton, *args)

            return cls._INSTANCE

        else:

            return super(Singleton, cls).__new__(cls)
