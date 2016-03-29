import numpy as np
from operator import or_, eq
from itertools import imap, repeat
from enum import Enum


def cumulative_apply(f, iterable):

    out = None
    for pos, val in enumerate(iterable):
        if pos == 0:
            out = val
        else:
            out = f(out, val)
    return out


class PositionMark(Enum):

    OK = 0
    NoGrowth = 1
    BadData = 2
    Empty = 3


class FilterArray(object):

    def __init__(self, data, filter):

        assert data.shape == filter.shape, "Data and filter don't match"

        self.__dict__["__numpy_data"] = data
        self.__dict__["__numpy_filter"] = filter

    @property
    def filter(self):

        return self.__dict__["__numpy_filter"]

    @property
    def mask(self):

        return self.__dict__["__numpy_filter"] > 0

    def filter_to_mask(self, *filters):

        return cumulative_apply(
            or_,
            imap(eq,
                 repeat(self.__dict__["__numpy_filter"], len(filters)),
                 (f.value if hasattr(f, "value") else f for f in filters)))

    def masked(self, *filters):

        return np.ma.MaskedArray(self.__dict__["__numpy_data"], mask=self.filter_to_mask(*filters))

    def __getattr__(self, item):

        return getattr(np.ma.MaskedArray(self.__dict__["__numpy_data"], mask=self.mask), item)