#!/usr/bin/env python
"""
Part of analysis work-flow that produces a grid-array from an image secion.
"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Mats Kvarnstroem", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.996"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import numpy as np
from math import ceil
#import logging
import matplotlib.pyplot as plt
from scipy import signal as scisg

#
# SCANNOMATIC LIBRARIES
#

import resource_logger as logger
import resource_histogram as hist
import resource_signal as r_signal

#
# FUNCTIONS
#


def simulate(measurements, segments):
    segments_left = segments
    true_segment_length = 28
    error_fraction = 30
    noise_max_fraction = 0.7

    error_function = np.random.rand(measurements) * \
        true_segment_length / error_fraction

    signal_start_pos = int(ceil(np.random.rand(1)[0] * \
        (measurements - segments)))

    measures = []
    noise_max_length = noise_max_fraction * true_segment_length

    for i in xrange(measurements):

        if i < signal_start_pos or segments_left == 0:
            measures.append(
                np.random.rand(1)[0] * noise_max_length)
        else:
            measures.append(true_segment_length)

            segments_left -= 1

    measures = np.array(measures) + error_function

    return signal_start_pos, measures


#
# CLASS Grid_Analysis
#

class Grid_Analysis():

    def __init__(self, parent, pinning_matrix, verbose=False, visual=False,
            dim_order={0: 0, 1: 1}, gridding_settings=None):

        self._parent = parent

        if parent is None:

            self.logger = logger.Log_Garbage_Collector()

        else:

            self.logger = self._parent.logger

        default_settings = {'median_coeff': 0.99,
                'use_otsu': False, 'manual_threshold': '0.05'}

        if gridding_settings is None:

            gridding_settings = default_settings

        for k in default_settings.keys():

            if k in gridding_settings.keys():

                setattr(self, k, gridding_settings[k])

            else:

                setattr(self, k, default_settings[k])

        self.im = None
        self.histogram = hist.Histogram(self.im, run_at_init=False)
        self.threshold = None
        #self.best_fit_start_pos = None
        self.best_fit_frequency = None
        self.best_fit_positions = None
        self.dim_order = dim_order
        self.R = 0

        self.visual = visual
        self.verbose = verbose

        self.pinning_matrix = pinning_matrix

    def set_dim_order(self, dim_order):

        self.dim_order = dim_order

    #
    # GET functions
    #

    def get_analysis(self, im, history=[]):
        """
        get_analysis is a convenience function for get_spikes and
        get_signal_position_and_frequency functions run on both
        dimensions of the image.

        (This function sets the self.im for get_spikes.)

        The function takes the following arguments:

        @im         An array / the image

        @pinning_matrix  A list/tuple/array where first position is
                        the number of rows to be detected and second
                        is the number of columns to be detected.

        @use_otsu   Causes thresholding to be done by Otsu
                    algorithm (Default)

        @median_coeff       Coefficient to threshold from the
                            median when not using Otsu.

        @verbose   If a lot of things should be printed out

        @visual     If visual information should be presented.

        @history    A history of the top-left positions selected
                    for the particular format for the particular plate

        The function returns two arrays, one per dimension, of the
        positions of the spikes and a quality index
        """


        self.im = im
        positions = [None, None]
        measures = [None, None]
        best_fit_frequency = [None, None]
        best_fit_positions = [None, None]
        adjusted_by_history = False

        R = 0

        if history is not None and len(history) > 0:

            history_rc = (np.array([h[1][0] for h in history]).mean(),
                np.array([h[1][1] for h in history]).mean())

            history_f = (np.array([h[2][0] for h in history]).mean(),
                np.array([h[2][1] for h in history]).mean())

        else:

            history_rc = None
            history_f = None

        """
        fig = plt.figure(10)
        fig.add_subplot(2,2,1).imshow(im, cmap=plt.cm.Greys)
        fig.add_subplot(2,2,2).plot(im.mean(0))
        t = hist.otsu(hist.Histogram(im.mean(0)))
        fig.gca().plot((0, im.shape[1]),(t, t))
        fig.gca().set_title("Dim 0 ({0})".format(pinning_matrix[dim_order[0]]))
        fig.add_subplot(2,2,3).plot(im.mean(1))
        t = hist.otsu(hist.Histogram(im.mean(1)))
        fig.gca().plot((0, im.shape[0]),(t, t))
        fig.gca().set_title("Dim 1 ({0})".format(pinning_matrix[dim_order[1]]))
        fig.show() 
        raw_input("Im {0}, Pin {1}, Order {2}".format(im.shape,
                pinning_matrix, dim_order))
        """

        #Obtaining current values by pinning grid dimension order

        np.save("_grid_im.npy", im)

        for dim in xrange(2):

            positions[dim], measures[dim] = \
                    self._get_new_spikes(
                    self.dim_order[dim]-1,
                    im=im)

            """
            positions[dim], measures[dim] = \
                    self.get_spikes(
                    self.dim_order[dim]-1,
                    im=im, visual=self.visual, verbose=self.verbose,
                    use_otsu=self.use_otsu, median_coeff=self.median_coeff,
                    manual_threshold=self.manual_threshold)
            """

            self.logger.info(
                "GRID ARRAY, Peak positions %sth dimension:\n%s" %\
                (str(dim), str(positions[dim])))

            best_fit_frequency[dim] = r_signal.get_signal_frequency(
                positions[dim])

            if best_fit_frequency[dim] is not None \
                                and history_f is not None:

                if abs(best_fit_frequency[dim] /\
                            float(history_f[dim]) - 1) > 0.1:

                    self.logger.warning(
                            ('GRID ARRAY, frequency abnormality for ' +\
                            'dimension {0} (Current {1}, Expected {2}'.format(
                            dim, best_fit_frequency[dim],
                            history_f)))

                    adjusted_by_history = True
                    best_fit_frequency[dim] = history_f[dim]

            best_fit_positions[dim] = r_signal.get_true_signal(
                im.shape[self.dim_order[dim]], self.pinning_matrix[dim],
                positions[dim],
                frequency=best_fit_frequency[dim],
                offset_buffer_fraction=0.5)

            if best_fit_positions[dim] is not None and \
                                        history_rc is not None:

                goodness_of_signal = r_signal.get_position_of_spike(
                    best_fit_positions[dim][0], history_rc[dim],
                    history_f[dim])

                if abs(goodness_of_signal) > 0.2:

                    self.logger.warning(("GRID ARRAY, dubious pinning " + \
                        "position for  dimension {0} (Current signal " + \
                        "start {1}, Expected {2} (error: {3}).".format(
                        dim, best_fit_positions[dim][0],
                        history_rc[dim], goodness_of_signal)))

                    adjusted_by_history = True

                    new_fit = r_signal.move_signal(
                        list(best_fit_positions[dim]),
                        [-1 * round(goodness_of_signal)], freq_offset=0)

                    if new_fit is not None:

                        self.logger.warning(
                            "GRID ARRAY, remapped signal for " +\
                            "dimension {0} , new signal:\n{1}".format(
                            dim, list(new_fit)))

                        best_fit_positions[dim] = new_fit[0]

            self.logger.info("GRID ARRAY, Best fit:\n" +\
                "* Elements: " + str(self.pinning_matrix[dim]) +\
                "\n* Positions:\n" + str(best_fit_positions[dim]))

            #DEBUGHACK
            #visual = True
            #DEBUGHACK - END

            if self.visual and best_fit_positions[dim] is not None:

                Y = np.ones(len(best_fit_positions[dim])) * 50
                Y2 = np.ones(positions[dim].shape) * 100
                plt.clf()

                if self.dim_order[dim] == 1:

                    plt.imshow(im[:, 900: 1200].T, cmap=plt.cm.gray)

                else:

                    plt.imshow(im[300: 600, :], cmap=plt.cm.gray)

                plt.plot(positions[dim], Y2, 'r*',
                    label='Detected spikes', lw=3, markersize=10)

                plt.plot(np.array(best_fit_positions[dim]),\
                    Y, 'g*', label='Selected positions', lw=3, markersize=10)

                plt.legend(loc=0)
                plt.ylim(ymin=0, ymax=150)
                plt.show()

            if best_fit_positions[dim] != None:

                #Comparing to previous
                if self.best_fit_positions != None:

                    if self.best_fit_positions[dim] != None:

                        R += ((best_fit_positions[dim] - \
                            self.best_fit_positions[dim]) ** 2).sum() / \
                            float(pinning_matrix[dim])

                        #Updating previous
                        self.logger.info(
                            "GRID ARRAY, Got a grid R at, {0}".format(R))

        if R < 20 and best_fit_positions[0] != None and \
                             best_fit_positions[1] != None:

            #self.best_fit_start_pos = best_fit_start_pos
            self.best_fit_frequency = best_fit_frequency
            self.best_fit_positions = best_fit_positions
            self.R = R

        else:

            self.R = -1

        if self.best_fit_positions == None:

            return None, None, None, adjusted_by_history

        else:

            ret_tuple = (self.best_fit_positions[0],
                    self.best_fit_positions[1], self.R,
                    adjusted_by_history)

            return ret_tuple

    def _get_new_spikes(self, dim, im, threshold_coeff=0.75):

        im_1D = im.mean(axis=dim)

        #DE-NOISE
        im_1D = scisg.medfilt(im_1D)

        #CONVOLUTION
        d_im_1D = scisg.convolve(im_1D, np.array((-1, 0, 1)), mode='same')

        #FIND MAX AND MIN FOR CENTER 50% (SAFE SECTION)
        low_bound = d_im_1D.shape[0] / 4
        high_bound = d_im_1D.shape[0] * 3 / 4

        low_threshold = threshold_coeff * d_im_1D[low_bound: high_bound].min()
        high_threshold = threshold_coeff * d_im_1D[low_bound: high_bound].max()

        spikes = list()
        slope_direction = None

        transitions = np.where(np.logical_or(d_im_1D < low_threshold,
                                             d_im_1D > high_threshold))[0]

        best_up_pos = None
        best_down_pos = None
                
        #DEMAND FLIPPING UP DOWN TO REPORT SPIKE
        for pos in transitions:

            if d_im_1D[pos] > 0:

                if slope_direction != 1:

                    if best_up_pos is not None and best_down_pos is not None:

                        spikes.append(sum((best_up_pos, best_down_pos)) / 2.0)

                    slope_direction = 1
                    best_up_pos = pos

                elif d_im_1D[best_up_pos] < d_im_1D[pos]:

                    best_up_pos = pos

            else:

                if slope_direction != -1:

                    slope_direction = -1
                    best_down_pos = pos

                elif d_im_1D[best_down_pos] > d_im_1D[pos]:

                    best_down_pos = pos

        spikes = np.array(spikes)
        spike_f = spikes[1:] - spikes[: -1]

        return spikes, spike_f

    def get_spikes(self, dimension, im=None, visual=False, verbose=False,
             use_otsu=True, median_coeff=0.99, manual_threshold=None):
        """
        get_spikes returns a spike list for a dimension of an image array

        The function takes the following arguments:

        @dimension  The dimension to be analysed (0 or 1)

        @im         An image numpy array, if left out previously loaded
                    image will be used

        @visual     Plot the results (only possible when running
                    the script from prompt)

        @verbose   Do a whole lot of print out of everything to
                    debug what goes wrong.

        @use_otsu   Using the Otsu algorithm to set the threshold used in
                    spike detection (default True). If Otsu is not used,
                    the median coefficient is used.

        @median_coeff   A float that is multiplied to the median of the
                        1D flattned image to get a threshold if otsu is not
                        used.
        """

        if im == None:

            im = self.im

        im_1D = im.mean(axis=dimension)

        if use_otsu:

            self.histogram.re_hist(im_1D)
            self.threshold = hist.otsu(histogram = self.histogram)

        elif manual_threshold:

            self.threshold = manual_threshold

        else:

            if median_coeff is None:

                median_coeff = 0.99

            self.threshold = np.median(im_1D) * median_coeff

        im_1D2 = (im_1D < self.threshold).astype(int)
        if visual:
            Y = im_1D2 * 100
            plt.plot(np.arange(len(im_1D)), im_1D, 'b-')
            plt.plot(np.arange(len(im_1D2)), Y, 'g-')
            #print self.threshold, median_coeff
            plt.axhline(y=self.threshold, color='r')
            plt.axhline(y=np.median(im_1D), color='g')

        #kernel = [-1,1]
        #spikes = np.convolve(im_1D, kernel, 'same')

        spikes_toggle_up = []
        spikes_toggle_down = []

        spikes_toggle = False

        for i in xrange(len(im_1D2)):

            if im_1D2[i] and not spikes_toggle:

                spikes_toggle = True
                spikes_toggle_up.append(i)

            elif not im_1D2[i]:

                if spikes_toggle == True:
                    spikes_toggle_down.append(i)

                spikes_toggle = False

        if len(spikes_toggle_down) != len(spikes_toggle_up):

            spikes_toggle_up = spikes_toggle_up[: len(spikes_toggle_down)]

        self.logger.debug("GRID CELL get_spikes, %d long %d downs %d ups." % \
            (len(im_1D2), len(spikes_toggle_down), len(spikes_toggle_up)))

        stt = (np.array(spikes_toggle_up) + np.array(spikes_toggle_down)) / 2

        if visual:

            Y = np.ones(len(stt)) * 80

            plt.plot(stt, Y, 'b.')
            plt.show()

        spike_f = stt[1:] - stt[: -1]

        return stt[1:], spike_f


#
# COMMAND PROMPT BEHAVOUR
#

#This is just bebugging stuff at present
if __name__ == "__main__":
    measurements = 50
    segments = 12

    tests = 1  # 0000
    corrects = 0
    test = 0
    correct_pos = -1
    im = plt.imread("section.tiff")
    verbose = False
    visual = True

    while test < tests:

        #correct_pos, measures = simulate(measurements, segments)
        positions, measures = get_spikes(im, 1)
        if verbose:
            pass
            #print len(measures)

        est_pos, frequency = get_signal_position_and_frequency(
                                            measures, segments)

        if verbose:
            #print correct_pos, est_pos
            print list(measures)
            print list(positions)

        if correct_pos == est_pos:
            corrects += 1
        else:
            break

        test += 1

    if visual:

        Y = np.ones(positions.shape) * 40

        if verbose:

            print len(Y), len(positions[est_pos: est_pos + segments + 2])

        plt.plot(positions, Y, 'ko', lw=2)
        pfound = positions[est_pos: est_pos + segments]
        Y = np.ones(pfound.shape) * 80

        plt.plot(positions[est_pos: est_pos + segments], Y, 'ro')
        plt.show()

    if verbose:

        print "*** ", est_pos, segments, len(positions)
        print "*** Got", corrects, "out of", tests, "right"
