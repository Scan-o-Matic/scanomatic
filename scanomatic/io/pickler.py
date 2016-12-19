from cPickle import UnpicklingError, Unpickler, load
from StringIO import StringIO
import os


def unpickle(path):
    """Unpickles data safely

    Args:
        path (str): Path to file

    Returns: Unpickled object

    """

    return load(safe_load(path))


def unpickles(data):
    """Unpickles data safely

    Args:
        data (str): Data

    Returns: Unpickled object

    """
    try:
        # return Unpickler(StringIO(data)).load()
        raise ImportError
    except (ImportError, UnpicklingError):

        version_compatibility = _RefactoringPhases()
        io_buffer = SafeStringIO(data, version_compatibility)
        version_compatibility.io = io_buffer
        return Unpickler(io_buffer).load()


def safe_load(path, return_string=False):

    version_compatibility = _RefactoringPhases()
    fh = SafeProxyFileObject(path, version_compatibility)
    version_compatibility.io = fh
    if return_string:
        return os.linesep.join(fh.readlines())
    else:
        return fh


def unpickle_with_unpickler(unpickler, path, *args, **kwargs):
    """Compatibility unpickler that handles problems

    This is used for

    1) Dealing with errors in line-endings in data sent between different OS
    2) Dealing with SoM refactoring import errors

    Args:
        unpickler (func): A function that unpickles a string of data
        path (str): Path to the data to be unpickled
        *args: Optional extra arguments for the unpickler
        **kwargs: Opional extra keyword arguments for the unpickler

    Returns: Unpickled object
    """

    return unpickler(safe_load(path), *args, **kwargs)


class _RefactoringPhases(object):
    def __init__(self):
        """Rewrites pickled data to match refactorings
        """
        self._next = None
        self.io = None

    def __call__(self, line):
        """

        Args:
            line (str): A pickled line

        Returns:

        """
        if self._next is None:

            if line.endswith("scanomatic.data_processing.curve_phase_phenotypes"):

                tell = self.io.tell()
                _next = self.io.readline()
                self.io.seek(tell)

                if _next.startswith('VectorPhenotypes'):
                    return line[:-49] + "scanomatic.data_processing.phases.features"
                elif _next.startswith('CurvePhasePhenotypes'):
                    return line[:-49] + "scanomatic.data_processing.phases.analysis"
                elif _next.startswith('CurvePhases'):
                    return line[:-49] + "scanomatic.data_processing.phases.segmentation"
                elif _next.startswith('CurvePhaseMetaPhenotypes'):
                    return line[:-49] + "scanomatic.data_processing.phases.features"

            return line
        else:
            ret = self._next
            self._next = None
            return ret


class SafeFileObject(file):

    def __init__(self, name, *validation_functions):

        file.__init__(self, name, mode='rb')
        self._validation_functions = validation_functions

    def readline(self):
        line = file.readline(self).rstrip("\r\n")
        for validation_func in self._validation_functions:
            line = validation_func(line)
        return line

    def readlines(self):
        def yielder():
            size = os.fstat(self.fileno()).st_size
            while size != self.tell():
                yield self.readline()

        return [l for l in yielder()]


class SafeProxyFileObject(object):

    def __init__(self, name, *validation_functions):

        self.__dict__["__file"] = open(name, mode='rb')
        self.__dict__["__validation_functions"] = validation_functions

    def readline(self):
        line = self.__dict__['__file'].readline().rstrip("\r\n")
        for validation_func in self.__dict__['__validation_functions']:
            line = validation_func(line)

        return line + "\n"

    def readlines(self):

        def yielder():
            size = os.fstat(self.fileno()).st_size
            while size != self.tell():
                yield self.readline()

        return [l for l in yielder()]

    def __getattr__(self, item):

        return getattr(self.__dict__['__file'], item)


class SafeStringIO(StringIO):

    def __init__(self, data, *validation_functions):

        StringIO.__init__(self, data)
        self._validation_functions = validation_functions

    def readline(self):

        line = StringIO.readline(self).rstrip("\r\n")
        for validation_func in self._validation_functions:
            line = validation_func(line)
        return line

    def readlines(self):

        def yielder():

            while self.len != self.pos:
                yield self.readline()

        return [l for l in yielder()]
