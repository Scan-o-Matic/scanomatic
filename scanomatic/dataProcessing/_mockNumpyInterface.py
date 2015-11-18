"""Contains basic aspects of numpy interface such that it in
basic aspect can be used in the same way while derived classes
can implement specific behaviours.
"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

from types import StringTypes

class NumpyArrayInterface(object):

    def __init__(self, dataObject):
        """dataObject is an object, preferrably a numpy array
        which derived classes work on.

        The data object is assumed to have at least three dimension
        (Or be one-dimensional with elements having at least two).

        """
        self._dataObject = dataObject

    def _posStringToTuple(self, posStr):

        plate, coords = [p.strip() for p in posStr.split(":")]
        x, y = [int(c) for c in coords.split("-")]
        return int(plate), x, y

    def __getitem__(self, key):

        if isinstance(key, StringTypes):
            plate, x, y = self._posStringToTuple(key)
            return self._dataObject[plate][x, y]
        elif isinstance(key, int):
            return self._dataObject[key]
        else:
            return self._dataObject[key[0]][key[1:]]

    def __iter__(self):

        for i in xrange(len(self._dataObject)):

            yield self.__getitem__(i)

    def __len__(self):

        return self._dataObject.shape[0]

    @property
    def shape(self):
        return self._dataObject.shape

    @property
    def ndim(self):
        return self._dataObject.ndim
