#!/usr/bin/env python
"""
This module reads xml-files as produced by scannomatic and returns numpy-arrays.
"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import os
import numpy as np
import re
import time

#
# INTERNAL DEPENDENCIES
#

import scanomatic.logger as logging

#
# GLOBALS
#

_logger = logging.Logger("Resource XML Reader")

#
# SCANNOMATIC LIBRARIES
#


#
# FUNCTIONS
#

#
# CLASSES
#

class XML_Reader():

    def __init__(self, file_path=None, data=None, meta_data=None,
                 scan_times=None):

        self._file_path = file_path
        self._loaded = (data is not None or meta_data is not None)
        self._data = data
        self._meta_data = meta_data
        self._scan_times = scan_times

        if file_path:
            self._logger = logging.getLogger(
                "XML-reader '{0}'".format(os.path.basename(file_path)))
            if not self.read():
                self._logger.error("XML Reader not fully initialized!")
            else:
                self._loaded = True
        else:
            self._logger = logging.getLogger('XML-reader')

    def __getitem__(self, position):

        if isinstance(position, int):
            return self._data[position]
        else:
            return self._data[position[0]][position[1:]]

    def read(self, file_path=None):
        """Reads the file_path file using short-format xml"""

        if file_path is not None:
            self._file_path = file_path

        try:
            fs = open(self._file_path, 'r')
        except:
            self._logger.error("XML-file '{0}' not found".format(self._file_path))
            return False

        print (
            time.strftime("%Y-%m-%d %H:%M:%S\t", time.localtime()) +
            "Started Processing\n")

        f = fs.read()
        fs.close()
        self._data = {}
        self._meta_data = {}

        XML_TAG_CONT = "<{0}>([^<]*)</{0}>"
        XML_TAG_INDEX_VALUE_CONT = "<{0} {1}..(\d).>([^<]*)</{0}>"
        XML_TAG_INDEX_VALUE = "<{0} {1}..(\d*).>"
        #XML_TAG_2_INDEX_VALUE = "<{0} {1}..(\d*). {2}..(\d*).>"
        XML_TAG_2_INDEX_FULL_NONGREEDY = "<{0} {1}..{3}. {2}..{4}.>(.*?)</{0}>"
        XML_ANY_CONT_NONGREEDY = ">([^<>]+?)<"
        #XML_TAG_2_INDEX_VALUE_CONT = "<{0} {1}..{3}. {2}..{4}.>[^\d]*([0-9.]*)<"
        XML_BAD_SCANS = "<s i..(\d*).><ok>0"

        #METADATA
        tags = ['start-t', 'desc', 'n-plates']
        for t in tags:
            self._meta_data[t] = re.findall(XML_TAG_CONT.format(t), f)

        #DATA
        bad_scans = map(int, re.findall(XML_BAD_SCANS, f))
        nscans = len(re.findall(XML_TAG_INDEX_VALUE.format('s', 'i'), f))
        pms = re.findall(XML_TAG_INDEX_VALUE_CONT.format('p-m', 'i'), f)
        pms = map(lambda x: map(eval, x), pms)
        print "Pinning matrices: {0}".format(pms)
        colonies = sum([p[1][0] * p[1][1] for p in pms if p[1] is not None])
        #measures = colonies * nscans
        max_pm = np.max(np.array([p[1] for p in pms]), 0)

        #GET NUMBER OF MEASURES
        v = re.findall(XML_TAG_2_INDEX_FULL_NONGREEDY.format(
            'gc', 'x', 'y', 0, 0), f)[0]

        m_types = len(re.findall(XML_ANY_CONT_NONGREEDY, v))

        for pm in pms:
            if pm[1] is not None:
                self._data[pm[0]] = np.zeros((pm[1] + (nscans, m_types)),
                                             dtype=np.float64)

        #SCAN TIMES
        self._scan_times = np.array(map(float, re.findall(XML_TAG_CONT.format('t'),
                                                          f)))
        self._scan_times.sort()  # Scans come in chronological order
        self._scan_times -= self._scan_times[0]  # Make it relative
        self._scan_times /= 3600  # Make it in hours

        print (
            "Ready for {0} plates ({1} scans, {2} measures per colony)".format(
                len(self._data), nscans, m_types))

        colonies_done = 0

        for x in xrange(max_pm[0]):
            for y in xrange(max_pm[1]):

                #print XML_TAG_2_INDEX_FULL_NONGREEDY.format('gc', 'x', 'y', x, y)

                v = re.findall(XML_TAG_2_INDEX_FULL_NONGREEDY.format(
                    'gc', 'x', 'y', x, y), f)

                v = [re.findall(XML_ANY_CONT_NONGREEDY, i) for i in v]

                for pos, vals in enumerate(v):

                    try:

                        v[pos] = map(lambda x: (x == 'nan' or x == 'None') and
                                     np.nan or np.float64(x), vals)

                    except ValueError:

                        v_list = []
                        for i in vals:

                            try:

                                v_list.append(np.float64(i))

                            except:

                                v_list.append(np.nan)

                        v[pos] = v_list

                    if len(v[pos]) != m_types:

                        v[pos] = [np.nan] * m_types

                """
                try:

                    v = map(lambda x: (x == 'nan' or x=='') and
                        np.nan or np.float64(x),
                        re.findall(
                                XML_TAG_2_INDEX_VALUE_CONT.format(
                                'gc','x','y',x,y), f))

                except ValueError:
                    print XML_TAG_2_INDEX_VALUE_CONT.format('gc','x','y',x,y)
                    print re.findall(
                                XML_TAG_2_INDEX_VALUE_CONT.format(
                                'gc','x','y',x,y), f)
                    #self._logger.error(re.findall(
                                XML_TAG_2_INDEX_VALUE_CONT.format(
                                'gc','x','y',x,y), f))
                """
                slicers = [False] * len(pms)
                for i, pm in enumerate(pms):
                    if pm[1] is not None:
                        if x < pm[1][0] and y < pm[1][1]:
                            slicers[i] = True
                #print "Data should go into Plates {0}".format(slicers)
                slice_start = 0
                for i, pm in enumerate(slicers):
                    if pm:
                        well_as_list = list(
                            (np.array(v)[range(slice_start,
                             len(v), sum(slicers))])[-1::-1])

                        for bs in bad_scans:
                            well_as_list.insert(bs - 1, np.nan)

                        d_arr = np.array(well_as_list)

                        if d_arr.ndim == 1:

                            self._data[i][x, y, :, 0] = d_arr

                        else:

                            self._data[i][x, y, :] = d_arr

                        slice_start += 1
                        colonies_done += 1

            print "Completed {0}%\r".format(
                100 * colonies_done / float(colonies))

        print (
            time.strftime("%Y-%m-%d %H:%M:%S\t", time.localtime()) +
            "Started Processing\n")

        return True

    def get_scan_times(self):

        return self._scan_times

    def set_file_path(self, file_path):
        """Sets file-path"""
        self._file_path = file_path

    def set_data_value(self, plateIndex, x, y, timeIndex, values):

        self._data[self._data.keys()[plateIndex]][x, y, timeIndex] = values

    def get_meta_data(self):
        """Returns meta-data dictionary"""
        return self._meta_data

    def get_data(self):

        return self._data

    def get_file_path(self):
        """Returns the currently set file-path"""
        return self._file_path

    def get_colony(self, plate, x, y):
        """Returns the data array for a specific colony"""
        try:
            return self._data[plate][x, y, :]
        except:
            return None

    def get_plate(self, plate):
        """Returns the data for the plate specified"""
        try:
            return self._data[plate]
        except:
            return None

    def get_shapes(self):
        """Gives the shape for each plate in data"""
        try:
            return [(p, self._data[p].shape) for p in self._data.keys()]
        except:
            return None

    def get_loaded(self):
        return self._loaded
