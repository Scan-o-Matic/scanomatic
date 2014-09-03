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

    _PRESETS_UI = {

        'start-text': 'Run',

        #FIXTURES
        'plate-label': 'Plate {0}',

        'pinning-matrices': {'A: 8 x 12 (96)': (8, 12),
                            'B: 16 x 24 (384)': (16, 24),
                            'C: 32 x 48 (1536)': (32, 48),
                            'D: 64 x 96 (6144)': (64, 96),
                            '--Empty--': None},

        'pinning-matrices-reversed': {(8, 12): 'A: 8 x 12 (96)',
                            (16, 24): 'B: 16 x 24 (384)',
                            (32, 48): 'C: 32 x 48 (1536)',
                            (64, 96): 'D: 64 x 96 (6144)',
                            None: '--Empty--'},

        'pinning-default': (32, 48),

        #ERRORS
        'error-not-implemented': "That feature hasn't been implemented yet!"
    }

    def __init__(self, presets=dict(), doCopy=True):

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

            raise KeyError("{0} has no key '{1}'".format(self, key))

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

    @staticmethod
    def _MergePresets(name, preset):

        return dict(hasattr(Model, name) and getattr(Model, name) or {},
                    **preset)

    @classmethod
    def LoadAppModel(cls, doCopy=True):

        if not(hasattr(cls, '_PRESETS_STAGE')
               and hasattr(cls, '_PRESETS_APP')):

            raise TypeError("{0} does not support Application Models".format(
                cls))

        return cls(
            presets=dict(
                Model._MergePresets("_PRESETS_APP", cls._PRESETS_APP),
                **Model._MergePresets("_PRESETS_STAGE", cls._PRESETS_STAGE)),
                   doCopy=doCopy)

    @classmethod
    def LoadStageModel(cls, stage='STAGE', doCopy=True):


        if not(hasattr(cls, '_PRESETS_' + stage)):

            raise TypeError("{0} does not support Stage {1} Model".format(
                cls, stage))

        return cls(
            presets=Model._MergePresets("_PRESETS_" + stage,
                                        getattr(cls, '_PRESETS_' + stage)),
            doCopy=doCopy)
