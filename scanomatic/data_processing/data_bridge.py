import numpy as np

#
#   INTERNAL DEPENDENCIES
#

import mock_numpy_interface
import scanomatic.io.xml.reader as xmlReader


class Data_Bridge(mock_numpy_interface.NumpyArrayInterface):

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

        self._time_index = None
        self._source = source
        super(Data_Bridge, self).__init__(None)

        # This method is assigned dynamically based on
        # type of data imported
        self.update_source = None

        self._create_array_representation(**kwargs)

    def _create_array_representation(self, **kwargs):

        if isinstance(self._source, np.ndarray):
            self._data = self._source.copy()
            self.update_source = self._update_to_array

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

            self._data = np.array(plates)
            self.update_source = self._update_to_feature_dict

        elif isinstance(self._source, xmlReader.XML_Reader):

            if "time" not in kwargs:
                raise Exception(
                    "XML Reader objects can only be bridged for a time index"
                    " at a time, you must supply keyword argument 'time'")

            else:

                self._time_index = kwargs["time"]
                tmp_d = []
                for p in self._source.get_data().values():
                    tmp_d.append(p[..., self._time_index, :].copy())
                self._data = np.array(tmp_d)

        else:

            raise Exception(
                "Unknown data format {0}".format(type(self._source)))

    def _update_to_feature_dict(self):
        """Updates the source inplace"""

        id_plate = 0
        for p in self._source:

            for id_d1, d1 in enumerate(p):

                for id_d2, cell in enumerate(d1):

                    id_measure = 0

                    for compartment in cell.values():

                        for key in compartment:

                            compartment[key] = self._data[
                                id_plate, id_d1, id_d2, id_measure]

                            id_measure += 1
            id_plate += 1

    def _update_to_array(self):
        """Updates the source inplace"""

        for i, p in enumerate(self._source):

            p[...] = self._data[i]

    def _update_to_xml_reader(self):
        """Updates the source inplace"""

        for id_plate in self._data.shape[0]:

            for d1 in self._data[id_plate].shape[0]:

                for d2 in self._data[id_plate].shape[1]:

                    self._source.set_data_value(
                        id_plate, d1, d2, self._time_index,
                        self._data[id_plate][d1, d2])

    def get_source(self):
        """Returns a reference to the source"""

        return self._source

    def get_as_array(self):
        """Returns the data as a normalisations compatible array"""

        return self._data

    def set_array_representation(self, array):
        """Method for overwriting the array representation of the data"""

        if array.shape == self._data.shape:
            self._data = array
        else:
            raise Exception(
                "New representation must match current shape: {0}".format(
                    self._data.shape))
