#!/usr/bin/env python
"""Resource Fixture"""
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
import copy

#
# INTERNAL DEPENDENCIES
#

from paths import Paths
import app_config
import grid_history
from scanomatic.models.factories.fixture_factories import FixtureFactory

#
# CLASSES
#


class Fixture_Settings(object):

    def __init__(self, dir_path, name):

        path_name = Paths().get_fixture_path(name, only_name=True)

        conf_rel_path = Paths().fixture_conf_file_rel_pattern.format(path_name)

        self._conf_path = os.path.join(dir_path, conf_rel_path)
        self.history = grid_history.GriddingHistory(self)
        self.model = FixtureFactory.serializer.load(self._conf_path)
        """:type : scanomatic.models.fixture_models.FixtureModel"""

    def get_marker_position(self, index):

        try:
            return self.model.orientation_marks_x[index], self.model.orientation_marks_y[index]
        except (IndexError, TypeError):
            return None

    @property
    def path(self):

        return self._conf_path

    def get_marker_path(self):

        #Evaluate math from settings-file
        good_path = True
        try:

            fs = open(self.marker_path, 'rb')
            fs.close()
        except:
            good_path = False

        if good_path:
            return self.marker_path

        #Evaluate name of file from settings-file with default path
        good_name = True
        try:
            fs = open(self._paths.images + os.sep + self.marker_name, 'rb')
            fs.close()
        except:
            good_name = False

        if good_name:
            return self._paths.images + os.sep + self.marker_name

        #Evalutate default marker path
        good_default = True
        try:

            fs = open(self._paths.marker, 'rb')
            fs.close()

        except:
            good_default = False

        if good_default:

            return self._paths.marker

        return None

    def save(self):
        pass

    def set_experiment_model(self, model, default_pinning=None):

        model['fixture'] = self.name
        model['plate-areas'] = copy.copy(self.plate_areas)
        model['pinnings-list'] = [default_pinning] * len(self.plate_areas)
        model['marker-count'] = self.marker_count
        model['grayscale'] = self.grayscale
        model['grayscale-area'] = copy.copy(self.grayscale_area)
        model['ref-marker-positions'] = copy.copy(self.marker_positions)
        model['marker-path'] = self.get_marker_path()


class Fixtures(object):

    def __init__(self):

        self._paths = paths.Paths()
        self._app_config = app_config.Config()
        self._fixtures = None
        self.update()

    def __getitem__(self, fixture):

        if fixture in self:
            return self._fixtures[fixture]

        return None

    def __contains__(self, name):

        return self._fixtures is not None and name in self._fixtures

    def update(self):

        directory = self._paths.fixtures
        extension = ".config"

        list_fixtures = map(lambda x: x.split(extension, 1)[0],
                            [file for file in os.listdir(directory)
                                if file.lower().endswith(extension)])

        self._fixtures = dict()

        for f in list_fixtures:
            if f.lower() != "fixture":
                fixture = Fixture_Settings(directory, f)

                if (float(fixture.version) >=
                        self._app_config.version_oldest_allow_fixture):

                    self._fixtures[fixture.name] = fixture

    def get_names(self):

        if self._fixtures is None:
            return tuple()

        return tuple(sorted(self._fixtures.keys()))

    def fill_model(self, model):

        fixture = model['fixture']
        if fixture in self._fixtures.keys():

            model['im-original-scale'] = self._fixtures[fixture].scale
            model['im-scale'] = self._fixtures[fixture].scale
            model['im-path'] = self._fixtures[fixture].im_path
            model['fixture-file'] = self._fixtures[fixture].conf_rel_path

        else:

            model['im-original-scale'] = 1.0
            model['im-scale'] = 1.0
            model['im-path'] = None
            model['fixture-file'] = self._paths.get_fixture_path(model['fixture'], only_name=True)
