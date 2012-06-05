#!/usr/bin/env python
"""
Part of analysis work-flow that produces a grid-array from an image secion.
"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson","Mats Kvarnstroem", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.993"
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

#
# SCANNOMATIC LIBRARIES
#

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
                np.random.rand(1)[0]* noise_max_length )
        else:
            measures.append(
                true_segment_length
            )

            segments_left -= 1

    measures = np.array(measures) + error_function

    return signal_start_pos, measures


#
# CLASS Grid_Analysis
#

class Grid_Analysis():

    def __init__(self, parent):

        self._parent = parent
        self.logger = self._parent.logger

        self.im = None
        self.histogram = hist.Histogram(self.im, run_at_init = False)        
        self.threshold = None
        #self.best_fit_start_pos = None
        self.best_fit_frequency = None
        self.best_fit_positions = None
        self.R = 0

    #
    # GET functions
    #

    def get_analysis(self, im, pinning_matrix, use_otsu = True, 
        median_coeff=None, verboise=False, visual=False,
        history=[]):
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

            @verboise   If a lot of things should be printed out

            @visual     If visual information should be presented.

            @history    A history of the top-left positions selected
                        for the particular format for the particular plate

            The function returns two arrays, one per dimension, of the
            positions of the spikes and a quality index

        """

        self.im = im
        positions = [None, None]
        measures = [None, None]
        #best_fit_start_pos = [None, None]
        best_fit_frequency = [None, None]
        best_fit_positions = [None, None]
        R = 0
        if history is not None and len(history) > 0:
            history_rc = (np.array([h[1][0] for h in history]).mean(), 
                np.array([h[1][1] for h in history]).mean())

            history_f = (np.array([h[2][0] for h in history]).mean(),
                np.array([h[2][1] for h in history]).mean())
        else:
            history_rc = None
            history_f = None
 
        #Obtaining current values
        for dimension in xrange(2):
            if median_coeff:
                positions[dimension], measures[dimension] = self.get_spikes(
                    dimension, im, visual, verboise, use_otsu, median_coeff)
            else:
                positions[dimension], measures[dimension] = self.get_spikes(
                    dimension, im, visual, verboise, use_otsu)

            #DEBUG ROBUSTNESS TEST
            #from random import randint
            #print "Positions before test:", len(positions[dimension])
            #pos_range = range(len(positions[dimension]))
            #for del_count in xrange(randint(1,5)+1):
                #del_pos = randint(0,len(pos_range)-1)
                #del pos_range[del_pos]
            #positions[dimension] = positions[dimension][pos_range]
            #measures[dimension] = measures[dimension][pos_range]
            #print "Deleted", del_count, "positions"
            #DEBUG END

            self.logger.info("GRID ARRAY, Peak positions %sth dimension:\n%s" %\
                (str(dimension), str(positions[dimension])))

            best_fit_frequency[dimension] = r_signal.get_signal_frequency(\
                positions[dimension])

            if best_fit_frequency[dimension] is not None and history_f is not None:
                if abs(best_fit_frequency[dimension]/float(history_f[dimension]) - 1) > 0.1:
                    self.logger.warning('GRID ARRAY, frequency abnormality for dimension {0} (Current {1}, Expected {2}'.format(dimension, best_fit_frequency[dimension], history_f))

            best_fit_positions[dimension] = r_signal.get_true_signal(\
                im.shape[int(dimension==0)], pinning_matrix[dimension], 
                positions[dimension], \
                frequency=best_fit_frequency[dimension], 
                offset_buffer_fraction=0.5)

            if best_fit_positions[dimension] is not None and history_rc is not None:
                goodness_of_signal = r_signal.get_position_of_spike(\
                    best_fit_positions[dimension][0], history_rc[dimension], 
                    history_f[dimension])
                if abs(goodness_of_signal) > 0.2:
                    self.logger.warning("GRID ARRAY, dubious pinning position for\
 dimension {0} (Current signal start {1}, Expected {2}).".format(\
                        dimension, best_fit_positions[dimension][0],
                        history_rc[dimension]))
                    
            ###START HERE MARKING OUT ALL OLD STUFF...
            #best_fit_start_pos[dimension], best_fit_frequency[dimension] = \
                #self.get_signal_position_and_frequency( measures[dimension],
                    #pinning_matrix[dimension], verboise )            
 
            self.logger.info("GRID ARRAY, Best fit:\n" + \
                "* Elements" + str(pinning_matrix[dimension]) +\
                "\n* Positions" + str(best_fit_positions[dimension]))

            #DEBUGHACK
            #visual = True
            #DEBUGHACK - END

            if visual:
                Y = np.ones(pinning_matrix[dimension]) * 50
                Y2 = np.ones(positions[dimension].shape) * 100
                plt.clf()
                if dimension == 1:
                    plt.imshow(im[:,900:1200].T, cmap=plt.cm.gray)
                else:
                    plt.imshow(im[300:600,:], cmap=plt.cm.gray)
                plt.plot(positions[dimension], Y2, 'r*', 
                    label='Detected spikes', lw=3, markersize=10)
                plt.plot(np.array(best_fit_positions[dimension]),\
                    Y ,'g*', label='Selected positions', lw=3, markersize=10)
                plt.legend(loc=0)
                plt.ylim(ymin=0, ymax=150)
                plt.show()
                #plt.savefig('signal_fit.png')
                #DEBUG HACK
                #visual = False
                #DEBUG HACK
            #if best_fit_start_pos[dimension] != None:
 
                #best_fit_positions[dimension] = \
                    #positions[dimension][best_fit_start_pos[dimension] : \
                        #best_fit_start_pos[dimension] + \
                        #pinning_matrix[dimension] ]

                #if visual:
                   
                    #import matplotlib.pyplot as plt
                    #m_im = im.mean(axis=dimension)
                    #plt.plot(np.arange(len(m_im)), m_im, 'b-')
                    #Y = np.ones(len(best_fit_positions[dimension])) * 150
                    #plt.plot(np.array(best_fit_positions[dimension]),\
                        #Y ,'r*')

                #best_fit_positions[dimension] = \
                    #self.get_inserts_discards_extrapolations(\
                        #best_fit_positions[dimension],\
                        #best_fit_frequency[dimension],\
                        #pinning_matrix[dimension])

                #if visual:
                    #Y = np.ones(len(positions[dimension])) * 140
                    #plt.plot(np.array(positions[dimension]),\
                        #Y ,'g*') * 50
                    #Y = np.ones(len(best_fit_positions[dimension])) * 160
                    #plt.plot(np.array(best_fit_positions[dimension]),\
                        #Y ,'b*')
                    #plt.get_axes().set_ylim(ymin=-1,ymax=3)
                    #plt.show()

            if best_fit_positions[dimension] != None:

                #Comparing to previous
                if self.best_fit_positions != None:
                    if self.best_fit_positions[dimension] != None:
                        R += ((best_fit_positions[dimension] - \
                            self.best_fit_positions[dimension])**2).sum() / \
                            float(pinning_matrix[dimension])



                        #Updating previous
                        self.logger.info("GRID ARRAY, Got a grid R at, %s" % str(R))

        #DEBUG R
        #fs = open('debug_R.log','a')
        #if self.best_fit_positions is None:
            #fs.write(str([best_fit_positions[0][0], best_fit_positions[1][0]]) + "\n")
        #else:
            #fs.write(str([R, (best_fit_positions[0][0], best_fit_positions[1][0]),
                #(self.best_fit_positions[0][0], self.best_fit_positions[1][0]) ]) + "\n")
        #fs.close()
        #DEBUG END

        if R < 20 and best_fit_positions[0] != None and best_fit_positions[1] != None:
            #self.best_fit_start_pos = best_fit_start_pos
            self.best_fit_frequency = best_fit_frequency
            self.best_fit_positions = best_fit_positions
            self.R = R
        else:            
            self.R = -1

        if self.best_fit_positions == None:
            return None, None, None
        else:
            return self.best_fit_positions[0], self.best_fit_positions[1], self.R


    def get_spikes(self, dimension, im=None, visual = False, verboise = False,\
             use_otsu=True, median_coeff=0.99):
        """
            get_spikes returns a spike list for a dimension of an image array

            The function takes the following arguments:

            @dimension  The dimension to be analysed (0 or 1)

            @im         An image numpy array, if left out previously loaded
                        image will be used
            
            @visual     Plot the results (only possible when running
                        the script from prompt)

            @verboise   Do a whole lot of print out of everything to
                        debug what goes wrong.

            @use_otsu   Using the Otsu algorithm to set the threshold used in
                        spike detection (default True). If Otsu is not used,
                        the median coefficient is used.

            @median_coeff   A float that is multiplied to the median of the 
                            1D flattned image to get a threshold if otsu is not
                            used.
        """
        if im == None:
            im  = self.im

        im_1D = im.mean(axis=dimension)

        if use_otsu:

            self.threshold = hist.otsu(
                self.histogram.re_hist(im_1D))
        else:
            self.threshold = np.median(im_1D)*median_coeff

        im_1D2 = (im_1D < self.threshold).astype(int)
        if visual:
            Y = im_1D2 * 100
            plt.plot(np.arange(len(im_1D)),im_1D,'b-')
            plt.plot(np.arange(len(im_1D2)), Y, 'g-')
            #print self.threshold, median_coeff
            plt.axhline(y=self.threshold, color = 'r')
            plt.axhline(y=np.median(im_1D), color = 'g')

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
           spikes_toggle_up = spikes_toggle_up[:len(spikes_toggle_down)]

        self.logger.debug("GRID CELL get_spikes, %d long %d downs %d ups." % \
            ( len(im_1D2), len(spikes_toggle_down), len(spikes_toggle_up)))
        stt = (np.array(spikes_toggle_up) + np.array(spikes_toggle_down)) / 2
            
        if visual:
            Y = np.ones(len(stt)) * 80
            plt.plot(stt,Y,'b.')
            plt.show()

        spike_f = stt[1:] - \
             stt[:-1]

        return stt[1:], spike_f


#
# COMMAND PROMPT BEHAVOUR
#

#This is just bebugging stuff at present
if __name__ == "__main__":
    measurements = 50
    segments = 12

    tests = 1#0000
    corrects = 0
    test = 0
    correct_pos = -1
    im = plt.imread("section.tiff")
    verboise = False
    visual = True

    while test < tests:

        #correct_pos, measures = simulate(measurements, segments)
        positions, measures = get_spikes(im, 1)
        if verboise:
            pass
            #print len(measures)
        est_pos, frequency = get_signal_position_and_frequency(measures, segments)
        if verboise:
            #print correct_pos, est_pos
            print list(measures)
            print list(positions)
        if correct_pos == est_pos:
            corrects += 1
        else:
            break
        test += 1

    if visual:
        Y = np.ones(positions.shape)*40
        if verboise:
            print len(Y), len(positions[est_pos:est_pos+segments + 2])
        plt.plot(positions, Y, 'ko', lw=2)
        pfound = positions[est_pos:est_pos+segments]
        Y = np.ones(pfound.shape)*80

        plt.plot(positions[est_pos:est_pos+segments], Y, 'ro')
        plt.show()

    if verboise:
        print "*** ", est_pos, segments, len(positions)
        print "*** Got", corrects, "out of", tests, "right"
