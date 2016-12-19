import numpy as np
from operator import or_, eq
from itertools import imap, repeat
from enum import Enum


def fold(f, iterable):

    out = None
    for pos, val in enumerate(iterable):
        if pos == 0:
            out = val
        else:
            out = f(out, val)
    return out


class Filter(Enum):
    """Marks for data positions.

    Attributes:
        Filter.OK: Data is good.
        Filter.NoGrowth: Something was here but didn't grow
        Filter.BadData: Somethingnt was here, grew but technical problems caused bad data.
        Filter.Empty: Nothing was ever placed here
        Filter.UndecidedProblem: Data is not good, but reason unknown, typically set when phenotype algorithms
            generate non-finite values.
    """
    OK = 0
    NoGrowth = 1
    BadData = 2
    Empty = 3
    UndecidedProblem = 4


class FilterArray(object):

    def __init__(self, data, filter):

        self.__dict__["__numpy_data"] = data
        self.__dict__["__numpy_filter"] = filter

    @property
    def filter(self):

        return self.__dict__["__numpy_filter"]

    @property
    def mask(self):

        return self.__dict__["__numpy_filter"] > 0

    def filter_to_mask(self, *filters):

        return fold(
            or_,
            imap(eq,
                 repeat(self.__dict__["__numpy_filter"], len(filters)),
                 (f.value if hasattr(f, "value") else f for f in filters)))

    def masked(self, *filters):
        if not filters:
            filters = (Filter.NoGrowth, Filter.BadData, Filter.UndecidedProblem, Filter.Empty)

        return np.ma.MaskedArray(self.__dict__["__numpy_data"], mask=self.filter_to_mask(*filters))

    def where_mask_layer(self, filter):

        return np.where(self.__dict__["__numpy_filter"] == filter.value)

    def filled(self, fill_value=np.nan):

        return np.ma.MaskedArray(self.__dict__["__numpy_data"], mask=self.mask).filled(fill_value=fill_value)

    def tojson(self, use_filled=False):

        if use_filled:
            val = self.filled()
        else:
            val = self.data
        filt = np.isnan(val)
        val = val.astype(np.object)
        val[filt] = None
        return val.tolist()

    def __getattr__(self, item):

        return getattr(np.ma.MaskedArray(self.__dict__["__numpy_data"], mask=self.mask), item)

    def __getitem__(self, item):

        return np.ma.MaskedArray(self.__dict__["__numpy_data"], mask=self.mask)[item]