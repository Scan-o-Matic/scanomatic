import numpy as np
import os
from scanomatic.ui_server.general import convert_url_to_path


class AnalysisLoader(object):
    def __init__(self, project):
        self.project = project
        self.path = convert_url_to_path(project)

    @property
    def times(self):
        return np.load(os.path.join(self.path, 'phenotype_times.npy'))

    @property
    def raw_growth_data(self):
        return np.load(os.path.join(self.path, 'curves_raw.npy'))

    @property
    def smooth_growth_data(self):
        return np.load(os.path.join(self.path, 'curves_smooth.npy'))
