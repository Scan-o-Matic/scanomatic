#
# DEPENDENCIES
#

import copy

#
# FUNCTIONS
#

def copy_model(model):

    tmp_model = dict()

    for k in model.keys():

        tmp_model[k] = model[k]

    return copy.deepcopy(tmp_model)


def copy_part_of_model(model, keys):

    tmp_model = dict()

    for k in keys:

        tmp_model[k] = model[k]

    return copy.deepcopy(model)


def link_to_part_of_model(model, keys):

    return Model_Link(model, keys)

#
# MODEL LINK
#

class Model_Link(object):

    def __init__(self, source_model, link_keys):

        self._source_model = source_model
        self._private_model = dict()
        self._link_keys = link_keys

    def __getattr__(self, key):

        if key in self._link_keys:

            return self._source_model[key]

        else:

            return self._private_model[key]

    def __setattr__(self, key, val):

        if key in self._link_keys:

            self._source_model[key] = val

        else:

            self._private_model[key] = val

    def keys(self):

        return link_keys + self._private_model.keys()


