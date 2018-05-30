import numpy as np
import os
from collections import namedtuple

from scanomatic.ui_server.general import convert_url_to_path


PlateData = namedtuple('PlateData', ['raw', 'smooth'])


class CorruptAnalysisError(Exception):
    pass


class PlateNotFoundError(Exception):
    pass


class AnalysisLoader(object):
    def __init__(self, project):
        self.project = project
        self.path = convert_url_to_path(project)

    @property
    def times(self):
        try:
            return np.load(os.path.join(self.path, 'phenotype_times.npy'))
        except (IOError, ValueError):
            raise CorruptAnalysisError()

    @property
    def raw_growth_data(self):
        try:
            return np.load(os.path.join(self.path, 'curves_raw.npy'))
        except (IOError, ValueError):
            raise CorruptAnalysisError()

    @property
    def smooth_growth_data(self):
        try:
            return np.load(os.path.join(self.path, 'curves_smooth.npy'))
        except (IOError, ValueError):
            raise CorruptAnalysisError()

    def get_plate_data(self, plate):
        raw = self.raw_growth_data
        smooth = self.smooth_growth_data
        try:
            return PlateData(raw[plate], smooth[plate])
        except IndexError:
            raise PlateNotFoundError('Plate {} not in project'.format(plate))
