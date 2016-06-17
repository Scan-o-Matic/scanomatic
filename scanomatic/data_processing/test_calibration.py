from . import calibration
import numpy as np
from scipy.stats import linregress

# TODO: Validation should be a calibration method

data = calibration.load_data_file()


def test_load_data():

    assert calibration.load_data_file() is not None


def test_load_calibration():

    assert calibration.load_calibration() is not None


def test_expand_data_lenghts():

    counts = data[calibration.CalibrationEntry.source_value_counts]
    exp_vals, _, _, _ = calibration._get_expanded_data(data)
    assert all(np.sum(c) == len(v) for c, v in zip(counts, exp_vals)), list((np.sum(c), len(v)) for c, v in zip(counts, exp_vals))


def test_expand_data_sums():

    counts = data[calibration.CalibrationEntry.source_value_counts]
    values = data[calibration.CalibrationEntry.source_values]
    data_sums = np.array(tuple(np.sum(np.array(c) * np.array(v)) for c, v in zip(counts, values)))

    exp_vals, _, _, _ = calibration._get_expanded_data(data)
    expanded_sums = np.array(tuple(v.sum() for v in exp_vals), dtype=np.float)
    np.testing.assert_allclose(expanded_sums, data_sums)


def test_expand_data_targets():

    _, targets, _, _ = calibration._get_expanded_data(data)
    np.testing.assert_allclose(targets.astype(np.float), data[calibration.CalibrationEntry.target_value])


def test_expand_vector_length():

    counts = data[calibration.CalibrationEntry.source_value_counts][0]
    values = data[calibration.CalibrationEntry.source_values][0]
    expanded = calibration._expand_compressed_vector(values, counts, np.float)
    assert sum(counts) == expanded.size


def test_expanded_vector_sum():
    counts = data[calibration.CalibrationEntry.source_value_counts][0]
    values = data[calibration.CalibrationEntry.source_values][0]
    data_sum = np.sum(np.array(counts) * np.array(values))
    expanded = calibration._expand_compressed_vector(values, counts, np.float)
    np.testing.assert_allclose(expanded.sum(), data_sum)


def test_calibration_opt_func():

    poly = calibration._get_calibration_optimization_function(2)
    assert poly([2], 1, 1)[0] == 6
    assert poly([2], 2, 0)[0] == 4
    assert poly([2], 0, 2)[0] == 8
    poly = calibration._get_calibration_optimization_function(4)
    assert poly([1], 1, 1)[0] == 2
    assert poly([1], 2, 0)[0] == 2
    assert poly([1], 0, 2)[0] == 2
    assert poly([2], 0, 1)[0] == 16


def test_calculate_polynomial():
    degree = 4
    poly = calibration.calculate_polynomial(data, degree=degree)
    assert poly.size == degree + 1
    np.testing.assert_allclose(poly[1:-2], 0)
    np.testing.assert_allclose(poly[-1], 0)
    assert np.unique(poly).size > 1


def test_polynomial():

    poly_coeffs = calibration.calculate_polynomial(data)
    poly = calibration.get_calibration_polynomial(poly_coeffs)
    expanded, targets, _, _ = calibration._get_expanded_data(data)
    expanded_sums = np.array(tuple(v.sum() for v in poly(expanded)))
    slope, intercept, _, p_value, stderr = linregress(expanded_sums, targets)

    np.testing.assert_almost_equal(slope, 1.0, decimal=3)
    assert abs(intercept) < 75000
    assert stderr < 0.05
    assert p_value < 0.1

