import csv
from collections import defaultdict
import numpy as np

from scanomatic.data_processing.phenotyper import Phenotyper


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


def _parse_non_data(data, mode):

    return tuple(data[i] for i in data if len(data[i]) != mode)


def _parse_data(data, mode, time_scale):

    all_data = tuple(data[i] for i in data if len(data[i]) == mode)
    data = np.array(all_data[1:], dtype=np.float)
    return all_data[0], data[:, 0] / time_scale, data[:, 1:]


def parse(path, time_scale=36000):

    with open(path, 'r') as fh:
        data = fh.readlines()

    dialect = csv.Sniffer().sniff(data[10])
    data = {i: v for i, v in enumerate(csv.reader(data, dialect=dialect))}
    mode_length = _get_count_mode(_count_row_lengths(data))
    return _parse_data(data, mode_length, time_scale), _parse_non_data(data, mode_length)


def load(path=None, data=None, times=None, time_scale=36000, reshape=True):

    if path:
        (_, times, data), _ = parse(path)

    data = data.T
    if data.shape[0] == 200:
        if reshape:
            data = np.array([[data[:100].reshape(10, 10)], [data[100:].reshape(10, 10)]])
        else:
            data = np.array([[data[:100]], [data[100:]]])
    else:
        data = data.reshape((1, ) * (4 - data.ndim) + data.shape)

    return Phenotyper(data, times_data=times)
