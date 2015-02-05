

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

_memoized = {}


def memoize(f):

    def memoized(*args, **kwargs):
        key = _get_id_tuple(f, args, kwargs)
        if key not in _memoized:
            _memoized[key] = f(*args, **kwargs)
        return _memoized[key]
    return memoized