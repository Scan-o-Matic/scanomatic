import importlib
import sys


class ExpiringModule(object):

    def __init__(self, module, run_code=""):
        """A module that gets deleted after use

        Args:
            module: The name of the module or submodule to import
            run_code: Any code to be evaluated while importing, note that if you wish to refer to the module you are
                importing, then you do it by the general name 'mod'

        Usage:

            This example imports matplotlib and whipes all traces when done.
            It imports it, setting the backend before importing pyplot

            with ExpiringModule('matplotlib', run_code='mod.use("Agg")') as _:
                with ExpiringModule('matplotlib.pyplot') as plt:
                     # Do your plotting here
                     pass

        Note:
            It is not safe to use while threading if other modules are imported and/or globals set while
            performing the `with` statement.

        Returns: ExpiringModule

        """
        self._module = module
        self._runcode = run_code
        self.__variables = []
        self.__modules = []

    def __enter__(self):

        self.__variables = [v for v in globals().iterkeys()]
        self.__modules = [v for v in sys.modules]

        mod = importlib.import_module(self._module)

        if self._runcode:
            eval(self._runcode)

        self.__variables = [v for v in globals().iterkeys() if v not in self.__variables]
        self.__modules = [v for v in sys.modules if v not in self.__modules]

        return mod

    def __exit__(self, exc_type, exc_val, exc_tb):

        for m in self.__modules:
            del sys.modules[m]

        for v in self.__variables:
            del v