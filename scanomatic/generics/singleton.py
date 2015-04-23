"""Singleton class"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


import warnings

_INSTANCES = {}
_INITED = {}


class Singleton(object):

    def __new__(cls, *args):

        global _INSTANCES
        if cls not in _INSTANCES:
            if cls is Singleton:
                _INSTANCES[cls] = super(Singleton, cls).__new__(Singleton, *args)
            else:
                _INSTANCES[cls] = super(Singleton, cls).__new__(cls)
        return _INSTANCES[cls]


class SingeltonOneInit(Singleton):

    def __new__(cls, *args):

        global _INITED
        instance = super(SingeltonOneInit, cls).__new__(cls)
        if instance not in _INITED:
            _INITED[instance] = False
        return instance

    def __init__(self, *args, **kwargs):
        global _INITED
        if self not in _INITED or not _INITED[self]:
            self.__one_init__(*args, **kwargs)
            _INITED[self] = True

    def __one_init__(self, *args, **kwargs):

        warnings.warn("One init not overwritten on {0}".format(type(self)))