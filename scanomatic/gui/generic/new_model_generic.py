"""The Main Controller"""
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

import copy

#
# INTERNAL DEPENDENCIES
#


class Model(object):

    def __init__(self, presets=dict(), doCopy=False):

        if doCopy:
            presets = copy.deepcopy(presets)

        self._values = presets

    def __getitem__(self, key):

        if key in self._values:

            return self._values[key]

        elif (hasattr(self, key) and key[0].lower() == key[0] and
              not key.startswith("_")):

            return getattr(self, key)()

        else:

            raise KeyError

    def __setitem__(self, key, value):

        if hasattr(self, key):

            if key in ("plate",):

                key = "_" + key

            else:

                raise ValueError("Key '{0}' is read-only".format(key))

        if key in self._values:

            self._values[key] = value

        else:

            raise ValueError("Key '{0}' unknown".format(key))

    @classmethod
    def LoadAppModel(cls, doCopy=False):

        if not(hasattr(cls, '_PRESETS_STAGE')
               and hasattr(cls, '_PRESETS_APP')):

            raise TypeError("{0} does not support Application Models".format(
                cls))

        return cls(presets=dict(cls._PRESETS_APP.items() +
                                cls._PRESETS_STAGE.items()),
                   doCopy=doCopy)

    @classmethod
    def LoadStageModel(cls, doCopy=False):

        if not(hasattr(cls, '_PRESETS_STAGE')):

            raise TypeError("{0} does not support Stage Models".format(
                cls))

        return cls(presets=cls._PRESETS_STAGE, doCopy=doCopy)
