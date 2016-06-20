import numpy as np
from enum import Enum
import json
from itertools import izip
import os
import shutil
from scipy.optimize import curve_fit
import time
from datetime import datetime
from dateutil import tz
from scipy.stats import linregress


from scanomatic.io.logger import Logger
from scanomatic.io.paths import Paths

# TODO: Need removal too
# TODO: Consider how to send pull request on a polynomial

_logger = Logger("Calibration")


class CalibrationEntry(Enum):
    image = 0
    """:type : CalibrationEntry"""
    colony_name = 1
    """:type : CalibrationEntry"""
    target_value = 2
    """:type : CalibrationEntry"""
    source_values = (3, 0)
    """:type : CalibrationEntry"""
    source_value_counts = (3, 1)
    """:type : CalibrationEntry"""


class CalibrationValidation(Enum):
    OK = 0
    """:type : CalibrationValidation"""
    BadSlope = 1
    """:type : CalibrationValidation"""
    BadIntercept = 2
    """:type : CalibrationValidation"""
    BadStatistics = 3
    """:type : CalibrationValidation"""


def validate_polynomial(data, poly):

    expanded, targets, _, _ = _get_expanded_data(data)
    expanded_sums = np.array(tuple(v.sum() for v in poly(expanded)))
    slope, intercept, _, p_value, stderr = linregress(expanded_sums, targets)

    try:
        np.testing.assert_almost_equal(slope, 1.0, decimal=3)
    except AssertionError:
        return CalibrationValidation.BadSlope

    if abs(intercept) > 75000:
        return CalibrationValidation.BadIntercept

    if stderr > 0.05 or p_value > 0.1:
        return CalibrationValidation.BadStatistics

    return CalibrationValidation.OK


def _eval_deprecated_format(entry, key):

    if isinstance(key, CalibrationEntry):
        key = key.value

    if isinstance(key, int):
        return entry[key]
    elif key:
        return _eval_deprecated_format(entry[key[0]], key[1:])
    else:
        return entry


def _parse_data(entry):

    try:
        entry = json.loads(entry)
    except ValueError:
        # Try parsing old format
        entry = {k.name: _eval_deprecated_format(eval(entry), k) for k in CalibrationEntry}

    return {CalibrationEntry[k]: v for k, v in entry.iteritems()}


def _valid_entry(entry):

    if entry is None:
        return False
    return CalibrationEntry.target_value in entry and CalibrationEntry.source_value_counts in entry and CalibrationEntry.source_values in entry


def _jsonify_entry(entry):

    return {k.name: v for k, v in entry.iteritems()}


def _jsonify(data):

    return json.dumps([_jsonify_entry(e) for e in data])


def _get_file_path(file_path=None, label=''):
    if file_path is None:
        if label:
            file_path = Paths().analysis_calibration_data.format(label + ".")
        else:
            file_path = Paths().analysis_calibration_data.format(label)

    return file_path


def save_data_to_file(data, file_path=None, label=''):

    file_path = _get_file_path(file_path, label)
    with open(file_path, 'w') as fh:
        json.dump(data, fh)


def load_data_file(file_path=None, label=''):

    file_path = _get_file_path(file_path, label)
    try:
        with open(file_path, 'r') as fs:

            try:
                data_store = json.load(fs)

            except ValueError:

                data_store = {
                    CalibrationEntry.target_value: [],
                    CalibrationEntry.source_values: [],
                    CalibrationEntry.source_value_counts: []}

                for i, line in enumerate(fs):

                    try:
                        entry = _parse_data(line)
                    except (ValueError, TypeError):
                        entry = None

                    if _valid_entry(entry):
                        data_store[CalibrationEntry.source_value_counts].append(
                            entry[CalibrationEntry.source_value_counts])
                        data_store[CalibrationEntry.source_values].append(
                            entry[CalibrationEntry.source_values])
                        data_store[CalibrationEntry.target_value].append(
                            entry[CalibrationEntry.target_value])

                    else:
                        _logger.warning("Could not parse line {0}: '{1}' in {2}".format(i, line.strip(), file_path))

    except IOError:
        raise IOError("File at {0} not found".format(file_path))

    return data_store


def _get_calibration_optimization_function(degree=5):

    arr = np.zeros((degree + 1,), np.float)

    def poly(x, c1, cn):
        arr[-2] = c1
        arr[0] = cn
        return tuple(v.sum() for v in np.polyval(arr, x))

    return poly


def get_calibration_polynomial(coefficients_array):

    return np.poly1d(coefficients_array)


def _get_expanded_data(data_store):

    measures = min(len(data_store[k]) for k in
                   (CalibrationEntry.target_value, CalibrationEntry.source_values, CalibrationEntry.source_value_counts))

    x = np.empty((measures,), dtype=object)
    y = np.zeros((measures,), dtype=np.float64)
    x_min = None
    x_max = None

    values = data_store[CalibrationEntry.source_values]
    counts = data_store[CalibrationEntry.source_value_counts]
    targets = data_store[CalibrationEntry.target_value]

    for pos in range(measures):

        x[pos] = _expand_compressed_vector(values[pos], counts[pos], dtype=np.float64)
        y[pos] = targets[pos]

        if x_min is None or x_min > x[pos].min():

            x_min = x[pos].min()

        if x_max is None or x_max < x[pos].max():

            x_max = x[pos].max()

    return x, y, x_min, x_max


def _expand_compressed_vector(values, counts, dtype):

    return np.hstack((np.repeat(value, count) for value, count in izip(values, counts))).astype(dtype)


def poly_as_text(poly):

    def coeffs():
        for i, coeff in enumerate(poly[::-1]):
            yield "{0:.2E} x^{1}".format(coeff, i)

    return "y = {0}".format(" + ".join(coeffs()))


def calculate_polynomial(data_store, degree=5):

    x, y, _, _ = _get_expanded_data(data_store)

    poly = _get_calibration_optimization_function(degree)

    p0 = np.zeros((2,), np.float)

    (c1, cn), pcov = curve_fit(poly, x, y, p0=p0)

    poly_vals = np.zeros((degree + 1))
    poly_vals[-2] = c1
    poly_vals[0] = cn

    _logger.info("Data produced polynomial {0} with 1 sigma per term (x^1, x^{2}) {1}".format(
        poly_as_text(poly_vals), np.sqrt(np.diag(pcov)), degree))

    return poly_vals


def load_calibrations(file_path=None):

    if file_path is None:
        file_path = Paths().analysis_polynomial

    try:

        with open(file_path, 'r') as fh:

            try:
                data = json.load(fh)
            except ValueError:
                data = {}
                fh.seek(0)
                for i, l in enumerate(fh):
                    try:
                        key, value = eval(l)
                        data[key] = value
                    except (TypeError, ValueError):
                        _logger.info("Skipping line {0}: '{0}' (can't parse)".format(i, l.strip()))

    except IOError:
        _logger.warning("Could not locate file '{0}'".format(file_path))
        data = {}

    return data


def load_calibration(label="", poly_degree=None, file_path=None):

    data = load_calibrations(file_path)
    if poly_degree is not None:
        label = "{0}_{1}".format(label, poly_degree)

    for k in data:

        if k.startswith(label):

            if poly_degree is None:
                _logger.info("Using polynomial {0}: {1}".format(k, poly_as_text(data[k])))

            return data[k]


def _safe_copy_file_if_needed(file_path):

    # Make copy of previous state
    if os.path.isfile(file_path):

        local_zone = tz.gettz()
        stamp = datetime.fromtimestamp(time.time(), local_zone).isoformat()

        target = "{0}.{1}.polynomials".format(file_path.rstrip("polynomials"), stamp)
        shutil.copy(file_path, target)


def add_calibration(label, poly, file_path=None):

    if file_path is None:
        file_path = Paths().analysis_polynomial

    _safe_copy_file_if_needed(file_path)

    data = load_calibrations(file_path)

    key = "{0}_{1}".format(label, len(poly) - 1)
    if key in data:
        _logger.warning("Replacing previous calibration {0}: {1}".format(key, data[key]))

    data[key] = poly.tolist() if hasattr(poly, 'tolist') else poly

    with open(file_path, 'w') as fh:

        json.dump(data, fh)


def remove_calibration(label, degree=None, file_path=None):

    if file_path is None:
        file_path = Paths().analysis_polynomial

    data = load_calibrations(file_path)
    keys = tuple(data.keys())
    has_changed = False

    for key in keys:

        if degree:

            if key == "{0}_{1}".format(label, degree):
                del data[key]
                has_changed = True
                break
        elif key.startswith("{0}_".format(label)):
            del data[key]
            has_changed = True

    if has_changed:
        _safe_copy_file_if_needed(file_path)
        with open(file_path, 'w') as fh:
            json.dump(data, fh)

        return True

    else:
        _logger.warning("No polynomial was found matching the criteria (label={0}, degree={1}".format(
            label, degree))
        return False
