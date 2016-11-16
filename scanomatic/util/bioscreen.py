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

    def __call__(self, data):

        return np.poly1d(self.value)(data)


def _count_row_lengths(data):

    counts = defaultdict(int)
    for i in data:
        counts[len(data[i])] += 1
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

    return tuple(data[i] for i in data if len(data[i]) != mode or i < include_until)


def _parse_data(data, mode, time_scale, start_row=-1):

    all_data = tuple(data[i] for i in data if len(data[i]) == mode and i > start_row)
    data = np.array(all_data[1:], dtype=np.object)

    try:
        time = data[:, 0].astype(np.float) / time_scale
    except ValueError:
        def f(v):
            return sum(float(a) * b for a, b in izip(v.split(":"), (1, 1/60., 1/3600.)))

        time = np.frompyfunc(f, 1, 1)(data[:, 0])
    except IndexError:
        print("Failed parsing data {0}".format(data))
        raise

    colstart = max(1, data.shape[1] - 200)

    return all_data[0][colstart:], time.astype(np.float), data[:, colstart:].astype(np.float)


def csv_loader(path):

    with open(path, 'r') as fh:
        data = fh.read()
        data = codecs.decode(data, chardet.detect(data)['encoding'])
        data = unicodedata.normalize('NFKD', data).encode('ascii', 'ignore')

    dialect = csv.Sniffer().sniff(data)
    data = data.split(dialect.lineterminator)

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

    if path:
        (_, times, data), _ = parse(path, time_scale=time_scale)

    data = preprocess(data)
    data = data.T
    if data.shape[0] == 200:
        if reshape:
            data = np.array([data[:100].reshape(10, 10, *data.shape[1:]),
                             data[100:].reshape(10, 10, *data.shape[1:])])
        else:
            data = np.array([[data[:100]], [data[100:]]])
    else:
        data = data.reshape((1, ) * (4 - data.ndim) + data.shape)

    return Phenotyper(data, times_data=times)
