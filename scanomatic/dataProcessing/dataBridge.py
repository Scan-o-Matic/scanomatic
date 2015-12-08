"""Contains basic aspects of numpy interface such that it in
basic aspect can be used in the same way while derived classes
can implement specific behaviours.
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
#   DEPENDENCIES
#

import numpy as np

#
#   INTERNAL DEPENDENCIES
#

import _mockNumpyInterface
import scanomatic.io.xml.reader as xmlReader


class Data_Bridge(_mockNumpyInterface.NumpyArrayInterface):

    def __init__(self, source, **kwargs):
        """The data structure is expected to be convertable into
        a four dimensional array where dimensions are as follows:

            1:   Plate
            2,3: Positional Coordinates
            4:   Data Measure

        For importing primary analysis data, this means that measures
        for the different compartments will be enumerated after each other
        along the 4th dimension as they come.

        In the case of working directly on an XML file such as the
        analysis.xml or the analysis_slim.xml a keyword argument 'time'
        is required:

            time:   The time index  for which to bridge the data

        """

        self._source = source
        super(Data_Bridge, self).__init__(None)

        #This method is assigned dynamically based on
        #type of data imported
        self.updateSource = None

        self._createArrayRepresentation(**kwargs)

    def _createArrayRepresentation(self, **kwargs):

        if isinstance(self._source, np.ndarray):
            self._smooth_growth_data = self._source.copy()
            self.updateSource = self._updateToArray

        elif isinstance(self._source, dict):

            plates = [[]] * len(self._source)  # Creates plates and 1D pos

            for p in self._source.values():

                for d1 in p:
                    plates[-1].append([])  # Vector for 2D pos

                    for cell in d1:

                        plates[-1][-1].append([])

                        for compartment in cell.values():

                            for value in compartment.values():

                                plates[-1][-1][-1].append(value)

            self._smooth_growth_data = np.array(plates)
            self.updateSource = self._updateToFeatureDict

        elif isinstance(self._source, xmlReader.XML_Reader):

            if "time" not in kwargs:
                raise Exception(
                    "XML Reader objects can only be bridged for a time index"
                    " at a time, you must supply keyword argument 'time'")

            else:

                self._timeIndex = kwargs["time"]
                tmpD = []
                for p in self._source.get_data().values():
                    tmpD.append(p[..., self._timeIndex, :].copy())
                self._smooth_growth_data = np.array(tmpD)

        else:

            raise Exception(
                "Unknown data format {0}".format(type(self._source)))

    def _updateToFeatureDict(self):
        """Updates the source inplace"""

        plateIndex = 0
        for p in self._source:

            for d1Index, d1 in enumerate(p):

                for d2Index, cell in enumerate(d1):

                    measureIndex = 0

                    for compartment in cell.values():

                        for valueKey in compartment:

                            compartment[valueKey] = self._smooth_growth_data[
                                plateIndex, d1Index, d2Index, measureIndex]

                            measureIndex += 1
            plateIndex += 1

    def _updateToArray(self):
        """Updates the source inplace"""

        for i, p in enumerate(self._source):

            p[...] = self._smooth_growth_data[i]

    def _updateToXMLreader(self):
        """Updates the source inplace"""

        for plateIndex in self._smooth_growth_data.shape[0]:

            for d1 in self._smooth_growth_data[plateIndex].shape[0]:

                for d2 in self._smooth_growth_data[plateIndex].shape[1]:

                    self._source.set_data_value(
                        plateIndex, d1, d2, self._timeIndex,
                        self._smooth_growth_data[plateIndex][d1, d2])

    def getSource(self):
        """Returns a reference to the source"""

        return self._source

    def getAsArray(self):
        """Returns the data as a normalisations compatible array"""

        return self._smooth_growth_data

    def setArrayRepresentation(self, array):
        """Method for overwriting the array representation of the data"""

        if (array.shape == self._smooth_growth_data.shape):
            self._smooth_growth_data = array
        else:
            raise Exception(
                "New representation must match current shape: {0}".format(
                    self._smooth_growth_data.shape))
