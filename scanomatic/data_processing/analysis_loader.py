import numpy as np
import os
from scanomatic.ui_server.general import convert_url_to_path


class AnalysisLoader(object):
    def __init__(self, project):
        self.project = project
        self.path = convert_url_to_path(project)

    @property
    def times(self):
        try:
            return np.load(os.path.join(self.path, '.npy'))
