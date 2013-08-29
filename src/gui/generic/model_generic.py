#
# DEPENDENCIES
#

import copy

#
# INTERNAL DEPENDENCIES
#

import src.resource_path as resource_path

#
# EXCEPTIONS
#


class Invalid_Key_Assignment(Exception):
    pass


class Unknown_Key_Request(Exception):
    pass

#
# FUNCTIONS
#

def get_model():

    m = Model(private_model=generic_gui_model)
    return m

#
# MODEL CLASS
#


class Model(object):

    def __init__(self, private_model=None, generic_model=None,
        paths=None):

        if private_model is None:
            self._pm = dict()
            self.build_private_model()
        else:
            self._pm = private_model

        self._counters = dict()

        if paths is None:
            self._paths = resource_path.Paths()
        else:
            self._paths = paths

        self._gm = generic_model

    def __getitem__(self, key):

        if key in self._counters.keys():
            return self._count_key(key)
        elif key in self._pm.keys():
            return self._pm[key]
        elif self._gm is not None:
            return self._gm[key]
        else:
            raise Unknown_Key_Request("'{0}' not found in {1}".format(key, self))
            return None

    def __setitem__(self, key, val):

        if key in self._counters.keys():

            raise Invalid_Key_Assignment(
                "'{0}' cannot be set directly as it is a counter".format(key))

        else:
            self._pm[key] = val

    def add_counter(self, key, counted_key):

        self._counters[key] = counted_key

    def remove_counter(self, key):

        del self._counter[key]

    def _count_key(self, key):

        return len(self.__getitem__(self._counters[key]))

    def keys(self):

        return self._pm.keys()

    def items(self):

        return self._pm.items()

    def values(self):

        return self._pm.values()

    def copy_model(self):

        m = Model(private_model=copy.deepcopy(self._pm),
                generic_model=self._gm)

        return m

    def copy_part(self, keys):

        pm = dict()

        for k in keys:

            pm[k] = copy.deepcopy(self._pm[k])

        m = Model(private_model=pm, generic_model=self._gm)
        return m

    def build_private_model():

        self._pm = dict()

#
# GENERIC GUI MODEL VALUES
#

generic_gui_model = {

#COMMON OBJECTS
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
