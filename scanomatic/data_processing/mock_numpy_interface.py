from types import StringTypes

# TODO: Clean up use?


class NumpyArrayInterface(object):

    def __init__(self, data):
        """data is an object, preferably a numpy array
        which derived classes work on.

        The data object is assumed to have at least three dimension
        (Or be one-dimensional with elements having at least two).

        """
        self._data = data

    @staticmethod
    def _position_2_string_tuple(position_string):

        plate, coords = [p.strip() for p in position_string.split(":")]
        x, y = [int(c) for c in coords.split("-")]
        return int(plate), x, y

    def __getitem__(self, key):

        if isinstance(key, StringTypes):
            plate, x, y = self._position_2_string_tuple(key)
            return self._data[plate][x, y]
        elif isinstance(key, int):
            return self._data[key]
        else:
            return self._data[key[0]][key[1:]]

    def __iter__(self):

        for i in xrange(len(self._data)):

            yield self.__getitem__(i)

    def __len__(self):

        return self._data.shape[0]

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim
