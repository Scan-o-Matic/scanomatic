#!/usr/bin/env python
"""Produces the calibration curve from calibration data"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Olle Nerman"]
__license__ = "GPL v3.0"
__version__ = "0.999"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import random
import shutil
import os
import time

import src.resource_path as resource_path
import src.gui.analysis.model_analysis as model_analysis


def expand_compressed_vector(values, counts):

    return [item for sublist in
            [[v] * n for v, n in zip(values, counts)] for item in sublist]


def vector_polynomial_sum_dropped_coeffs(X, *coefficient_array):

    coefficient_array = np.array((
        coefficient_array[0], 0, 0,
        coefficient_array[1], 0), dtype=np.float64)

    Y = np.zeros(len(X))

    for pos in xrange(len(X)):

        Y[pos] = (np.sum(np.polyval(coefficient_array, X[pos])))

    return Y


def vector_polynomial_sum_dropped_coeffs2(X, *coefficient_array):

    coefficient_array = np.array((
        coefficient_array[0], 0, 0, 0,
        coefficient_array[1], 0), dtype=np.float64)

    Y = np.zeros(len(X))

    for pos in xrange(len(X)):
        Y[pos] = (np.sum(np.polyval(coefficient_array, X[pos])))

    return Y


def vector_polynomial_sum(X, *coefficient_array):

    Y = np.zeros(len(X))

    for pos in xrange(len(X)):
        Y[pos] = (np.sum(np.polyval(coefficient_array, X[pos])))

    return Y


def load_data_file(file_path):

    try:

        fs = open(file_path)

    except:

        raise Exception("There's no file at {0}".format(file_path))

    labelTargetPos = None
    labelSourceValuePos = None
    labelSourceCountPos = None
    data_store = {
        labelTargetValue: [],
        labelSourceValues: [],
        labelSourceCounts: []}

    for lineIndex, line in enumerate(fs):

        usefulLine = False
        line = line.strip('\n').replace('"', '').split(sep)

        if labelTargetValue in line:
            labelTargetPos = line.index(labelTargetValue)
            usefulLine = True
        if labelSourceValues in line:
            labelSourceValuePos = line.index(labelSourceValues)
            usefulLine = True
        if labelSourceCounts in line:
            labelSourceCountPos = line.index(labelSourceCounts)
            usefulLine = True

        if not usefulLine and None in (labelTargetPos, labelSourceCountPos,
                                       labelSourceValuePos):

            raise Exception("Skipping line {0}".format(sep.join(line)))

        elif not usefulLine:

            try:

                data_store[labelTargetValue].append(eval(line[labelTargetPos]))
                data_store[labelSourceValues].append(
                    eval(line[labelSourceValuePos]))
                data_store[labelSourceCounts].append(
                    eval(line[labelSourceCountPos]))

            except:

                raise Exception("Bad line #{0}: {1}".format(
                    lineIndex, sep.join(line)))

    fs.close()

    measures = len(data_store[labelTargetValue])

    return data_store, measures


def get_expanded_data(data_store, measures):

    X = np.empty((measures,), dtype=object)
    Y = np.zeros((measures,), dtype=np.float64)
    x_min = None
    x_max = None

    for pos in range(measures):

        X[pos] = np.asarray(expand_compressed_vector(
            data_store[labelSourceValues][pos],
            data_store[labelSourceCounts][pos]),
            dtype=np.float64)

        Y[pos] = data_store[labelTargetValue][pos]

        if x_min is None or x_min > X[pos].min():

            x_min = X[pos].min()

        if x_max is None or x_max < X[pos].max():

            x_max = X[pos].max()

    return X, Y, x_min, x_max

paths = resource_path.Paths()

sep = "\t"
labelTargetValue, labelSourceValues, labelSourceCounts = \
    model_analysis.specific_log_book['calibration-measure-labels']

if __name__ == "__main__":

    #
    #   SETTING UP PATHS
    #

    if len(sys.argv) == 1:
        file_path = paths.analysis_calibration_data
    else:
        file_path = sys.argv[-1]

    file_out_path = paths.analysis_polynomial

    #
    #   READING IN FILE
    #

    data_store, measures = load_data_file(file_path)

    #
    #   EXPANDING VECTORS
    #

    X, Y, x_min, x_max = get_expanded_data(data_store, measures)
    #coeff_guess is the initial guess for solution
    #This is important as it sets the degree of the polynomial
    #An array of length 5 equals a 4th deg pol

    coeff_guess = np.asarray((1, 1))

    #X and Y should be arrays both, should contain as each element
    #a vector (array) of measuers
    #The length of Y should equal the length of the first dimension of X
    #The curve fit will return the best solution for the polynomial

    popt, pcov = curve_fit(
        vector_polynomial_sum_dropped_coeffs2, X, Y, p0=coeff_guess)

    popt = [popt[0], 0, 0, 0, popt[1], 0]
    do_save = True

    if os.path.isfile(file_out_path):
        shutil.copy(file_out_path, "{0}.{1}.old".format(file_out_path,
                                                        int(time.time())))

    try:

        fs = open(file_out_path, 'w')

    except:

        do_save = False

    if do_save:

        fs.write(str(['smart_poly_cal_' + str(len(coeff_guess) + 2) + "_deg",
                      list(popt)]) + "\n")

        fs.close()

    #create a polynomial for the coefficient vector
    pa = np.poly1d(popt)
    x_points = np.linspace(x_min, x_max, 100)
    plt.clf()
    plt.plot(x_points, pa(x_points), label="y=ax**5 + bx")

    coeff_guess = np.asarray((1, 1))
    popt, pcov = curve_fit(vector_polynomial_sum_dropped_coeffs, X, Y,
                           p0=coeff_guess)
    popt = [popt[0], 0, 0, popt[1], 0]
    p1 = np.poly1d(popt)
    plt.plot(x_points, p1(x_points), label="y=ax**4 + bx")

    coeff_guess = np.asarray((1, 1, 1, 1))
    popt, pcov = curve_fit(vector_polynomial_sum, X, Y, p0=coeff_guess)
    p1 = np.poly1d(popt)
    plt.plot(x_points, p1(x_points),
             label=str(len(coeff_guess) - 1) + "th deg pol fit")

    coeff_guess = np.asarray((1, 1, 1, 1, 1, 1))
    popt, pcov = curve_fit(vector_polynomial_sum, X, Y, p0=coeff_guess)
    p1 = np.poly1d(popt)
    plt.plot(x_points, p1(x_points),
             label=str(len(coeff_guess) - 1) + "th deg pol fit")

    coeff_guess = np.asarray((1, 1, 1))
    popt, pcov = curve_fit(vector_polynomial_sum, X, Y, p0=coeff_guess)
    p1 = np.poly1d(popt)
    plt.plot(x_points, p1(x_points),
             label=str(len(coeff_guess) - 1) + "th deg pol fit")

    plt.legend(loc=0)
    plt.xlabel("Per pixel value offset from Colony Growth in 'Kodak Space'")
    plt.ylabel("Cell Estimate per pixel")
    plt.title("Conversion formula independent measures")
    plt.show()

    print "Doing random test removing 10 data points for verification from",
    print len(Y), "total"

    #
    # Initialize the arrays of right shape and right dtype
    omitted = 10
    X1 = np.empty((measures - omitted,), dtype=object)
    Y1 = np.zeros((measures - omitted,), dtype=np.float64)
    X2 = np.empty((omitted,), dtype=object)
    Y2 = np.zeros((omitted,), dtype=np.float64)

    pos_list = range(measures)
    omissionList = random.sample(pos_list, omitted)
    omissionList.sort()
    pos_list = list(set(pos_list).difference(omissionList))

    print "Simulation set", pos_list
    print "Test set", omissionList

    # Populating from the first label, expanding the compressed vectors

    in_pos = 0
    out_pos = 0
    for pos in range(measures):

        if pos in pos_list:

            X1[in_pos] = np.asarray(expand_compressed_vector(
                data_store[labelSourceValues][pos],
                data_store[labelSourceCounts][pos]),
                dtype=np.float64)

            Y1[in_pos] = data_store[labelTargetValue][pos]

            in_pos += 1

        else:

            X2[out_pos] = np.asarray(expand_compressed_vector(
                data_store[labelSourceValues][pos],
                data_store[labelSourceCounts][pos]),
                dtype=np.float64)

            Y2[out_pos] = data_store[labelTargetValue][pos]

            out_pos += 1

    coeff_guess = np.asarray((1, 1))
    popt, pcov = curve_fit(vector_polynomial_sum_dropped_coeffs2, X1, Y1,
                           p0=coeff_guess)

    popt = [popt[0], 0, 0, 0, popt[1], 0]
    ps = np.poly1d(popt)
    plt.clf()
    plt.plot(x_points, pa(x_points), label="Full set")
    plt.plot(x_points, ps(x_points), label="Set minus {0} random".format(
        omitted))

    plt.legend(loc=0)
    plt.xlabel("Per pixel value offset from Colony Growth in 'Kodak Space'")
    plt.ylabel("Cell Estimate per pixel")
    plt.title("Conversion polynomial stability for withdrawing 10 points")
    plt.show()

    CY2 = vector_polynomial_sum_dropped_coeffs2(X2, popt[0], popt[4])
    print CY2
    print Y2
    line_max = np.asarray(CY2 + Y2).max()

    plt.clf()
    plt.plot(Y2, CY2, 'r*')
    plt.plot([0, line_max], [0, line_max], 'b-')
    plt.xlabel("Lasse's independent cell count measure")
    plt.ylabel("Cell Estimate calculation from fitted curve ax**5 + bx = y")
    plt.title("Evaluation of polynomial")
    plt.show()
