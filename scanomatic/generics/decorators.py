import scanomatic.io.logger as logger

import time
import datetime
import multiprocessing
from inspect import ismethod
from threading import Thread


class _ClassPropertyDescriptor(object):

    def __init__(self, fget, fset=None):
        self._fget = fget
        self._fset = fset

    def __get__(self, obj, cls=None):
        if cls is None:
            cls = type(obj)
        return self._fget.__get__(obj, cls)()

    def __set__(self, obj, value):
        if not self._fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self._fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self._fset = func
        return self


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return _ClassPropertyDescriptor(func)


def _get_id_tuple(f, args, kwargs, mark=object()):

    l = [id(f)]
    for arg in args:
        l.append(id(arg))
    l.append(id(mark))
    for k, v in kwargs:
        l.append(k)
        l.append(id(v))
    return tuple(l)

_MEMOIZED = {}


def memoize(f):

    def memoized(*args, **kwargs):
        key = _get_id_tuple(f, args, kwargs)
        if key not in _MEMOIZED:
            _MEMOIZED[key] = f(*args, **kwargs)
        return _MEMOIZED[key]
    return memoized


_TIME_LOGGER = logger.Logger("Time It")


def timeit(f):

    def timer(*args, **kwargs):

        t = time.time()
        _TIME_LOGGER.info("Calling {0}".format(f))
        ret = f(*args, **kwargs)
        _TIME_LOGGER.info("Call to {0} lasted {1}".format(f, str(datetime.timedelta(seconds=time.time() - t))))
        return ret

    return timer

_PATH_LOCK = dict()


def path_lock(f):

    def _acquire(path):

        try:
            while not _PATH_LOCK[path].acquire(False):
                time.sleep(0.05)
        except KeyError:
            raise IndentationError("Path '{0}' not registered as lock".format(path))

    def locking_wrapper_method(self_cls, path, *args, **kwargs):
        _acquire(path)
        ret = f(self_cls, path, *args, **kwargs)
        _PATH_LOCK[path].release()
        return ret

    def locking_wrapper_function(path, *args, **kwargs):
        _acquire(path)
        ret = f(path, *args, **kwargs)
        _PATH_LOCK[path].release()
        return ret

    if ismethod(f):
        return locking_wrapper_method
    else:
        return locking_wrapper_function


def register_path_lock(path):
    _PATH_LOCK[path] = multiprocessing.Lock()


def threaded(f):

    def _threader(*args, **kwargs):

        thread = Thread(target=f, args=args, kwargs=kwargs)
        thread.start()

    return  _threader