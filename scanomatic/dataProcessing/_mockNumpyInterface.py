from types import StringTypes


class NumpyArrayInterface(object):

    def __init__(self, dataObject):
        """dataObject is an object, preferrably a numpy array
        which derived classes work on.

        The data object is assumed to have at least three dimension
        (Or be one-dimensional with elements having at least two).

        """
        self._smooth_growth_data = dataObject

    def _position_2_string_tuple(self, posStr):

        plate, coords = [p.strip() for p in posStr.split(":")]
        x, y = [int(c) for c in coords.split("-")]
        return int(plate), x, y

    def __getitem__(self, key):

        if isinstance(key, StringTypes):
            plate, x, y = self._position_2_string_tuple(key)
            return self._smooth_growth_data[plate][x, y]
        elif isinstance(key, int):
            return self._smooth_growth_data[key]
        else:
            return self._smooth_growth_data[key[0]][key[1:]]

    def __iter__(self):

        for i in xrange(len(self._smooth_growth_data)):

            yield self.__getitem__(i)

    def __len__(self):

        return self._smooth_growth_data.shape[0]

    @property
    def shape(self):
        return self._smooth_growth_data.shape

    @property
    def ndim(self):
        return self._smooth_growth_data.ndim
