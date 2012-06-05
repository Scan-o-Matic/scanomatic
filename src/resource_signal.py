#!/usr/bin/env python
"""
The resource signal-module deals with detecting repeating patterns such as 
gray-scales and grids.
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
import logging

#
# SCANNOMATIC LIBRARIES
#


#
# FUNCTIONS
#

def get_perfect_frequency(best_measures, guess_frequency, tollerance=0.15):

    dists = get_spike_distances(best_measures)

    good_measures = []
    tollerance = (1-tollerance, 1+tollerance)
    guess_frequency = float(guess_frequency)

    for d in dists:
        if tollerance[0] < d/guess_frequency < tollerance[1]:
            good_measures.append(d)
        elif tollerance[0] < d/(2*guess_frequency) < tollerance[1]:
            good_measures.append(d/2.0)
    return np.mean(good_measures)

def get_perfect_frequency2(best_measures, guess_frequency, tollerance=0.15):

    where_measure = np.where(best_measures==True)[0]
    if where_measure.size < 1:
        return guess_frequency

    toll = (1-tollerance, 1+tollerance)
    guess_frequency = float(guess_frequency)
    f = where_measure[-1] - where_measure[0]

    f /= (np.round(f/guess_frequency))

    if toll[1] > f/guess_frequency > toll[0]:
        return f

    return get_perfect_frequency(best_measures, guess_frequency, tollerance)

def get_signal_frequency(measures):
    """
        get_signal_frequency returns the median distance between two
        consecutive measures.

        The function takes the following arguments:

        @measures       An array of spikes as returned from get_spikes

    """

    tmp_array = np.asarray(measures)
    #print "F", tmp_array
    return np.median( tmp_array[1:] - tmp_array[:-1] ) 


def get_best_offset(n, measures, frequency=None):
    """
        get_best_offset returns a optimal starting-offset for a hypthetical
        signal with frequency as specified by frequency-variable
        and returns a distance-value for each measure in measures to this 
        signal at the optimal over-all offset.

        The function takes the following arguments:

        @n              The number of peaks expected

        @measures       An array of spikes as returned from get_spikes

        @frequency      The frequency of the signal, if not submitted
                        it is derived as the median inter-measure
                        distance in measures.

    """


    dist_results = []

    if sum(measures.shape) == 0:
        logging.warning("RESOURCE SIGNAL: No spikes where passed, so best offset can't be found.")    
        return None

    if n > measures.size:
        n = measures.size

    if measures.max() == 1:
        m_where = np.where(measures==True)[0]
    else:
        m_where = measures

    if frequency is None:
        frequency = get_signal_frequency(measures)
   
    if np.isnan(frequency):
        return None

    for offset in xrange(int(np.ceil(frequency))):

        quality = [] 

        for m in m_where:

            #IMPROVE THIS ONE...
            #n_signal_dist is peak index of the closest signal peak
            n_signal_dist = np.round((m - offset) / float(frequency))

            signal_diff = offset + frequency * n_signal_dist - m
            if abs(signal_diff) > 0:
                quality.append(signal_diff**2)
            else:
                quality.append(0)
        dist_results.append(np.sum(np.sort(np.asarray(quality))[:n]))

    #print np.argsort(np.asarray(dist_results))
    #print np.sort(np.asarray(dist_results))
    return np.asarray(dist_results).argmin()


def get_spike_quality(measures, n=None, offset=None, frequency=None):
    """
        get_spike_quality returns a quality-index for each spike
        as to how well it fits the signal.

        If no offset is supplied, it is derived from measures.

        Equally so for the frequency.

        The function takes the following arguments:

        @measures       An array of spikes as returned from get_spikes

        @n              The number of peaks expected (needed if offset
                        is not given)

        @offset         Optional. Sets the offset of signal start

        @frequency      The frequency of the signal, if not submitted
                        it is derived as the median inter-measure
                        distance in measures.

    """

    if frequency is None:
        frequency = get_signal_frequency(measures)

    if offset is None and n != None:
        offset = get_best_offset(n, measures, frequency)

    if offset is None:
        print "*** ERROR: You must provide n if you don't provide offset"
        return None

    quality_results = []

    for m in measures:

        #n_signal_dist is peak number of the closest signal peak
        n_signal_dist = np.round((m - offset) / frequency)

        quality_results.append( ( m - offset + frequency * n_signal_dist)**2 )


    return quality_results 

def get_true_signal(max_value, n, measures, measures_qualities= None,
    offset=None, frequency=None, offset_buffer_fraction=0):

    """
        get_true_signal returns the best spike pattern n peaks that 
        describes the signal (described by offset and frequency).

        The function takes the following arguments:

        @max_value      The number of pixel in the current dimension

        @n              The number of peaks expected

        @measures       An array of spikes as returned from get_spikes

        @measures_qualities
                        Optional. A quality-index for each measure,
                        high values representing bad quality. If not
                        set, it will be derived from signal.

        @offset         Optional. Sets the offset of signal start

        @frequency      The frequency of the signal, if not submitted
                        it is derived as the median inter-measure
                        distance in measures.    
        @offset_buffer_fraction     Default 0, buffer to edge on
                        both sides in which signal is not allowed

    """ 


    if frequency is None:
        frequency = get_signal_frequency(measures)

    if frequency == 0:
        return None

    if offset is None:
        offset = get_best_offset(n, measures, frequency)


    if measures.max() == 1:
        m_array = np.where(np.asarray(measures)==True)[0]
    else:
        m_array = np.asarray(measures)

    if measures_qualities is None:
        measures_qualities = get_spike_quality(m_array, n, offset, frequency)

    mq_array = np.asarray(measures_qualities)

    if offset is None:
        return None

    start_peak = 0
    start_position_qualities = []
    frequency = float(frequency)
    while offset_buffer_fraction*frequency >= offset + frequency * \
        ((n-1) + start_peak):
        start_peak += 1
        start_position_qualities.append(0)
    #print "---Best signal---"
    #print offset, frequency, n, start_peak, max_value
    #print "peaks", m_array
    while offset_buffer_fraction*frequency <offset + frequency * \
        ((n-1) + start_peak) < max_value - offset_buffer_fraction*frequency:

        covered_peaks = 0
        quality = 0
        ideal_peaks = (np.arange(n) + start_peak) * frequency + offset

        for pos in xrange(n):
           
            distances = (m_array - float(ideal_peaks[pos]))**2
            closest = distances.argmin()
            #print closest, ((ideal_peaks - float(m_array[closest]))**2).argmin(), pos, np.round((m_array[closest]-offset) / frequency), pos + start_peak
            if np.round((m_array[closest]-offset) / frequency) == pos + start_peak:
                #Most difference with small errors... should work ok. 
                quality += distances[closest]
                #if distances[closest] >= 1:
                #    quality += np.log2(distances[closest])
                #quality += ((m_array - (offset + frequency * (n + pos + start_peak))).min())**2
                #quality += np.log2(((m_array - (offset + frequency * (n + pos + start_peak)))**2).min())
                covered_peaks += 1

        if covered_peaks > 0:
            start_position_qualities.append(covered_peaks + 1 / ((quality+1) / covered_peaks))
        else:
            start_position_qualities.append(0)
        start_peak += 1

    #If there simply isn't anything that looks good, the we need to stop here.
    if len(start_position_qualities) == 0:
        return None

    best_start_pos = int(np.asarray(start_position_qualities).argmax())

    logging.info("SIGNAL: Quality at start indices {0}".format(\
        start_position_qualities))
 
    quality_threshold = np.mean(mq_array) + np.std(mq_array) * 3

    ideal_signal = np.arange(n)*frequency + offset + best_start_pos * frequency

    best_fit = []

    for pos in xrange(len(ideal_signal)):

        best_measure = float( m_array[((m_array - float(ideal_signal[pos]))**2).argmin()] )
        if (ideal_signal - best_measure).argmin() == pos:
            if (ideal_signal[pos] - best_measure)**2 < quality_threshold:
                best_fit.append(best_measure)
            else:
                best_fit.append(ideal_signal[pos])
        else:
            best_fit.append(ideal_signal[pos])


    return ideal_signal

def get_center_of_spikes(spikes):
    """
        The function returns the an array matching the input-array but
        for each stretch of consequtive truth-values, only the center
        is kept true.

        @args : signal (numpy, 1D boolean array)

    """    

    up_spikes = spikes.copy()
    t_zone = False
    t_low = None

    for pos in xrange(up_spikes.size):
        if t_zone:
            if up_spikes[pos] == False or pos == up_spikes.size-1:
                if pos == up_spikes.size-1:
                    pos+=1
                up_spikes[t_low:pos] = False
                up_spikes[t_low + (t_low - pos)/2] = True
                t_zone = False

        else:
            if up_spikes[pos] == True:
                t_zone = True
                t_low = pos

    return up_spikes

def get_spike_distances(spikes):

    spikes_where = np.where(spikes == True)[0]
    if spikes_where.size == 0:
        return ()

    return np.append(spikes_where[0], spikes_where[1:] - spikes_where[:-1])

def get_best_spikes(spikes, frequency, tollerance=0.05, 
    require_both_sides=False):
    """
        Looks through a spikes-array for spikes with expected distance to
        their neighbours (with a tollerance) and returns these

        @args: spikes (numpy 1D boolean array of spikes)

        @args: frequency (expected frequency (float))

        @args: tollerance (error tollerance (float))

        @args: require_both_sides (boolean)

    """
    best_spikes = spikes.copy()
    spikes_dist = get_spike_distances(spikes)

    frequency = float(frequency)    
    accumulated_pos = 0  
    tollerance = (1-tollerance, 1+tollerance)

    for pos in xrange(spikes_dist.size):
        
        accumulated_pos += spikes_dist[pos]
        good_sides = tollerance[0] < spikes_dist[pos]/frequency < tollerance[1]
        good_sides += tollerance[0] < spikes_dist[pos]/(2*frequency) < tollerance[1]

        if pos + 1 < spikes_dist.size:
            good_sides += tollerance[0] < spikes_dist[pos+1]/frequency < tollerance[1]
        if good_sides >= require_both_sides + 1 - \
            (require_both_sides == True and pos + 1 == spikes_dist.size):
            pass
        else:
            best_spikes[accumulated_pos] = False

    return best_spikes


def get_position_of_spike(spike, signal_start, frequency):
    """
        Gives the spike position as a float point indicating which signal it
        is relative the signal start.

        @args: spike: The point where the spike is detected.

        @args: signal_start: The known or guessed start of the signal

        @args: frequency: The frequency of the signal

        @returns: Float point value for the closest position in the signal.
    """

    return (spike - signal_start) / float(frequency)
