import csv
from collections import defaultdict
import numpy as np
import chardet
import unicodedata
import codecs
from itertools import izip
from scanomatic.data_processing.phenotyper import Phenotyper
from enum import Enum


class Preprocessing(Enum):

    AsLoaded = (1, 0)
    Warringer2003_S_cerevisae = (0.191, 0, 0.499, 1, 0)
    Precog2016_S_cerevisiae = (0.82673123484708266, 0, 1, 0)
    Precog2016_E_coli = (0.75389848795692815, 0, 1, 0)
    Precog2016_Sz_pombe = (0.64672463774234579, 0, 1, 0)
    Precog2016_P_pastori = (0.5653284345804932, 0, 1, 0)
    Precog2016_C_albicans = (0.5790256635480614, 0, 1, 0)

    def __call__(self, data):

        return np.poly1d(self.value)(data)


def _get_row_length(row):

    return len(tuple(e for e in row if e and e != ' '))


def _get_row_is_possible_at_length(row, length):

    return any(e.strip() for e in row) and len(row) >= length


def _count_row_lengths(data):

    counts = defaultdict(int)
    for i in data:
        counts[_get_row_length(data[i])] += 1
    return counts


def _get_count_mode(counts):

    max_val = 0
    max_key = None

    for key in counts:
        if counts[key] > max_val:
            max_val = counts[key]
            max_key = key
    return max_key


def _parse_non_data(data, mode, include_until=-1):

    return tuple(data[i] for i in data if _get_row_length(data[i]) < mode or i < include_until)


def _parse_data(data, mode, time_scale, start_row=0):

    all_data = tuple(data[i][:mode] for i in data if _get_row_is_possible_at_length(data[i], mode) and i >= start_row)
    data = np.array(all_data[1:], dtype=np.object)

    if data.size == 0:
        print("No data to parse")
        raise ValueError("No data")

    try:
        time = data[:, 0].astype(np.float) / time_scale
    except ValueError:
        def f(v):
            return sum(float(a) * b for a, b in izip(v.split(":"), (1, 1/60., 1/3600.)))

        print("Failed direct parsing for time in decimal format, attempting clock format")
        time = np.frompyfunc(f, 1, 1)(data[:, 0])

    except IndexError:
        print("Failed parsing data {0}".format(data))
        raise

    column_start = max(1, data.shape[1] - 200)

    headers = all_data[0][column_start:]

    try:
        data = data[:, column_start:].astype(np.float)
    except ValueError:
        try:
            data = np.array([[val if val else None for val in row[column_start:]] for row in data], dtype=np.float)
        except ValueError:
            print("Failed to interpret data ({0} rows, type ({1})) as floats".format(len(data), data.dtype))
            raise

    return headers, time.astype(np.float), data


def csv_loader(path):

    with open(path, 'r') as fh:
        data = fh.read()
        data = codecs.decode(data, chardet.detect(data)['encoding'])
        data = unicodedata.normalize('NFKD', data).encode('ascii', 'ignore')

    dialect = csv.Sniffer().sniff(data)

    data = data.split(dialect.lineterminator)

    if len(data) == 1:
        for linedelim in ('\r\n', '\n\r', '\n', '\r'):
            data = data[0].split(linedelim)
            if len(data) > 1:
                break

    # Sniffing has to be repeated to not become confused by non-commented comment lines
    dialect = csv.Sniffer().sniff(data[10])
    data = {i: v for i, v in enumerate(csv.reader(data, dialect=dialect))}
    return data


def parse(path, time_scale=36000):

    data = csv_loader(path)
    mode_length = _get_count_mode(_count_row_lengths(data))
    rows = max(data.keys()) + 1
    i = 0
    ret = None
    while i < rows:
        try:
            ret = _parse_data(data, mode_length, time_scale, start_row=i)
            break
        except ValueError:
            i += 1

    return ret, _parse_non_data(data, mode_length, include_until=i)


def load(path=None, data=None, times=None, time_scale=36000, reshape=True,
         preprocess=Preprocessing.Precog2016_S_cerevisiae):
    """Loads a phenotyper object for a bioscreen experiment

    :param path: Path to the bioscreen data file
    :param data: If not path is supplied, data should be the growth data
    :param times: If not path is supplied, should be times-vector in hours
    :param time_scale: How many units per hour time is reported if not in hours
    :param reshape: If data should be reshaped as 2 ten by ten plates.
    :param preprocess: If imported data should have their values transformed.
        Typically by some polynomial. For standard options use one of the
        `Preprocessing` values. You can also supply your own function that
        takes the element-wise value and returns a corresponding pre-processed value:
        ```yprim = f(y)```
    :return:
    """
    if path:
        try:
            (_, times, data), _ = parse(path, time_scale=time_scale)
        except TypeError:
            print("*** Could not parse data***")
            return None

    data = preprocess(data)
    data = data.T
    if data.shape[0] == 200:
        if reshape:
            data = np.array([data[:100].reshape(10, 10, *data.shape[1:]),
                             data[100:].reshape(10, 10, *data.shape[1:])])
        else:
            data = np.array([[data[:100]], [data[100:]]])
    elif data.shape[0] == 100:
        if reshape:
            data = np.array([data.reshape(10, 10, *data.shape[1:])])
        else:
            data = np.array([data])
    else:
        data = data.reshape((1, ) * (4 - data.ndim) + data.shape)

    return Phenotyper(data, times_data=times)
