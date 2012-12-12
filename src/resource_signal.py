#!/usr/bin/env python
"""
The resource signal-module deals with detecting repeating patterns such as 
gray-scales and grids.
"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson","Mats Kvarnstroem", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.997"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import numpy as np
from math import ceil
import logging
from scipy import signal, ndimage

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

def move_signal(signals, shifts, frequencies=None, freq_offset=1):

    if len(shifts) != len(signals):
        logging.error("1st Dimension missmatch between signal and shift-list")
        return None

    else:
        if frequencies is None:
            frequencies = [None] * len(shifts)
            for i in xrange(len(shifts)):
                frequencies[i] = (np.array(signals[i][1:]) - np.array(signals[i][:-1])).mean()

        
        for i,s in enumerate(map(int, shifts)):
            if s != 0:
                
                f = frequencies[(i + freq_offset) % len(signals)]
                if s > 0:
                    signal = list(signals[i][s:])
                    for i in xrange(s):
                        signal.append(signal[-1] + f)
                else:
                    signal = signals[i][:s]
                    for i in xrange(-s):
                        signal.insert(0, signal[0] - f)

                signals[i][:] = np.array(signal)

        for i,s in enumerate([s - int(s) for s in shifts]):

            if s != 0:

                signals[i][:] = np.array([sign + s*frequencies[i] for sign in signals[i]])

        return signals

def get_continious_slopes(s, min_slope_length=20, noise_reduction=4):
    """Function takes a 1D noisy signal, e.g. from taking mean of image slice
    in one dimension and gets regions of continious slopes.

    Returns two arrays, first with all continious up hits
    Second with all continious down hits"""

    #Get derivative of signal without catching high freq noise
    ds = signal.fftconvolve(s, np.array([-1, -1, 1, 1]), mode="full")

    #Look for positive slopes
    s_up = ds > 0
    continious_s_up = signal.fftconvolve(s_up, 
        np.ones((min_slope_length,)), mode='full') == min_slope_length

    #Look for negative slopes
    s_down = ds < 0
    continious_s_down = signal.fftconvolve(s_down,
        np.ones((min_slope_length,)), mode='full') == min_slope_length

    #Reduce noise 2
    for i in range(noise_reduction):
        continious_s_up = ndimage.binary_dilation(continious_s_up)
        continious_s_down = ndimage.binary_dilation(continious_s_down)
    for i in range(noise_reduction):
        continious_s_up = ndimage.binary_erosion(continious_s_up)
        continious_s_down = ndimage.binary_erosion(continious_s_down)


    return continious_s_up, continious_s_down


def get_closest_signal_pair(s1, s2, s1_value=-1, s2_value=1):
    """The function returns the positions in s1 and s2 for where pairs of
    patterns s1-value -> s2-value are found (s1-value is assumed to preceed
    s2-value)."""

    s1_positions = np.where(s1==s1_value)[0]
    s2_positions = np.where(s2==s2_value)[0]

    #Match all 
    signals = list()
    for p in s1_positions:
        tmp_diff = s2_positions - p
        tmp_diff = tmp_diff[tmp_diff > 0]
        if tmp_diff.size > 0:
            p2 = tmp_diff.min() + p
            if len(signals) > 0 and p2 == signals[-1][1]:
                if p2 - p < signals[-1][1] - signals[-1][0]: 
                    del signals[-1]
                    signals.append((p, p2))
            else:
                signals.append((p, p2))
                
    S = np.array(signals)
    if S.size == 0:
        return None, None

    return S[:, 0], S[:, 1]


def get_signal_spikes(down_slopes, up_slopes):
    """Returns where valleys are in a signal based on down and up slopes"""


    #combined_signal = down_slopes.astype(np.int) * -1 + up_slopes.astype(np.int)

    #Edge-detect so that signal start is >0 and signal end <0
    kernel = np.array([-1, 1])
    d_down = np.round(signal.fftconvolve(down_slopes, kernel, mode='full')).astype(np.int)
    d_up = np.round(signal.fftconvolve(up_slopes, kernel, mode='full')).astype(np.int)

    s1, s2 = get_closest_signal_pair(d_down, d_up, s1_value=-1, s2_value=1)
    return s2


def _get_closest(X, Y):

    new_list = list()
    for i in ideal_signal:
        delta_i = np.abs(X - i)
        delta_reciprocal = np.abs(Y - X[delta_i.argmin()])
        if delta_i.min() == delta_reciprocal.min():
            new_list += ([X[delta_i.argmin()],
                Y[delta_reciprocal.argmin()]])

    return np.array(new_list)


def _get_alt_closest(X, Y):

    dist = np.abs(np.subtract.outer(X, Y))
    idx1 = np.argmin(dist, axis = 0)
    idx2 = np.argmin(dist, axis = 1)
    Z = np.c_[X[idx1[idx2] == np.arange(len(X))], 
        Y[idx2[idx1] == np.arange(len(Y))]].ravel()

    return Z


def _get_orphans(X, shortX):

    return X[np.abs(np.subtract.outer(X, shortX)).min(axis=1).astype(bool)]


def get_offset_quality(s, offset, expected_spikes, wl, raw_signal):

    #Get the ideal signal from parameters
    ideal_signal = np.arange(expected_spikes) * wl + offset    

    Z = _get_alt_closest(s, ideal_signal)

    #Making arrays
    #X  is s positions
    #Y  is ideal_signal positions
    X = Z[0::2]
    Y = Z[1::2]

    new_signal = np.r_[X, _get_orphans(ideal_signal, Y)]
    new_signal_val = raw_signal[new_signal.astype(int)]

    dXY = np.abs(np.r_[(Y - X),
        np.ones(ideal_signal.size - X.size) * (0.5 * wl)])

    X_val = raw_signal[X.astype(np.int)]
    dV = 3 * np.abs(new_signal_val - np.median(X_val))
    dV[new_signal_val > np.median(X_val)*0.8] *= 0.25
    q = -0.1 * dXY * dV

    return q.sum()


def get_grid_signal(raw_signal, expected_spikes):
    """Gives grid signals according to number of expected spikes (rows or columns)
    on 1D raw signal"""

    #Get slopes
    up_slopes, down_slopes = get_continious_slopes(
            raw_signal, min_slope_length=10, noise_reduction=4)

    #Get signal from slopes
    s = get_signal_spikes(down_slopes, up_slopes)
    print s

    #Wave-length 'frequency'

    wl = get_perfect_frequency2(s, get_signal_frequency(s))

    #Signal length is wave-length * expected signals
    l = wl * expected_spikes

    #Evaluating all allowed offsets:
    grid_scores = list()
    for offset in range(int(raw_signal.size - l)):
        grid_scores.append(
            get_offset_quality(s, offset, expected_spikes, 
            wl, raw_signal))

    GS = np.array(grid_scores)
    offset = GS.argmax()

    #Make signal here

    return GS, wl, offset, s
