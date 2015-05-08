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

#
# INTERNAL DEPENDENCIES
#

from paths import Paths
import app_config
import grid_history
from scanomatic.models.factories.fixture_factories import FixtureFactory
from scanomatic.io.logger import Logger
import ConfigParser


#
# CLASSES
#


class FixtureSettings(object):

    def __init__(self, dir_path, name):

        self._logger = Logger("Fixture {0}".format(name))

        path_name = Paths().get_fixture_path(name, only_name=True)

        conf_rel_path = Paths().fixture_conf_file_rel_pattern.format(path_name)

        self._conf_path = os.path.join(dir_path, conf_rel_path)
        self.model = self._load_model(name)
        self.history = grid_history.GriddingHistory(self)

    def _load_model(self, name):
        """:rtype : scanomatic.models.fixture_models.FixtureModel"""

        try:
            return tuple(FixtureFactory.serializer.load(self._conf_path))[0]
        except (IndexError, ConfigParser.Error), e:
            if isinstance(e, ConfigParser.Error):
                self._logger.error("Trying to load an outdated fixture at {0}, this won't work".format(self._conf_path))
            return FixtureFactory.create(path=self._conf_path, name=name)

    def get_marker_position(self, index):

        try:
            return self.get_marker_positions()[index]
        except (IndexError, TypeError):
            return None

    def get_marker_positions(self):

        return zip(self.model.orientation_marks_x, self.model.orientation_marks_y)

    @property
    def path(self):

        return self._conf_path

    def get_marker_path(self):

        paths = Paths()

        for path in (self.model.orentation_mark_path,
                     os.path.join(paths.images, os.path.basename(self.model.orentation_mark_path)),
                     paths.marker):
            try:

                with open(path, 'rb') as _:
                    return path
            except IOError:
                self._logger.error("The designated orientation marker file does not exist ({0})".format(path))

        return None

    def save(self):

        FixtureFactory.serializer.dump(self.model, self.path)


class Fixtures(object):

    def __init__(self):

        self._app_config = app_config.Config()
        self._fixtures = None
        self.update()

    def __getitem__(self, fixture):
        """:rtype : FixtureSettings"""
        if fixture in self:
            return self._fixtures[fixture]

        return None

    def __contains__(self, name):

        return self._fixtures is not None and name in self._fixtures

    def update(self):

        directory = Paths().fixtures
        extension = ".config"

        list_fixtures = map(lambda x: x.split(extension, 1)[0],
                            [fixture for fixture in os.listdir(directory)
                                if fixture.lower().endswith(extension)])

        self._fixtures = dict()

        for f in list_fixtures:
            if f.lower() != "fixture":
                fixture = FixtureSettings(directory, f)
                self._fixtures[fixture.model.name] = fixture

    def get_names(self):

        if self._fixtures is None:
            return tuple()

        return tuple(sorted(self._fixtures.keys()))

    def fill_model(self, model):

        fixture_name = model['fixture']
        if fixture_name in self:
            fixture = self[fixture_name]
            model['im-original-scale'] = fixture.model.scale
            model['fixture-file'] = fixture.path

        else:

            model['im-original-scale'] = 1.0
            model['im-scale'] = 1.0
            model['fixture-file'] = Paths().get_fixture_path(model['fixture'], only_name=True)
