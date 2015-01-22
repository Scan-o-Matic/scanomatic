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

import config_file

#
# CLASSES
#


class Fixture_Settings(object):

    def __init__(self, dir_path, name, paths):

        name = paths.get_fixture_path(name, only_name=True)
        self._paths = paths
        self.dir_path = dir_path
        self.conf_rel_path = paths.fixture_conf_file_rel_pattern.format(name)
        self.conf_path = os.sep.join((dir_path, self.conf_rel_path))
        self.im_path = os.sep.join(
            (dir_path, paths.fixture_image_file_rel_pattern.format(name)))
        self.scale = 0.25

        #THIS SHOULD BE DONE ELSEWHERE
        self.name = name.replace("_", " ").capitalize()

        self.marker_name = None

        for attrib in ('marker_path', 'marker_count', 'grayscale',
                       'marker_positions', 'plate_areas', 'grayscale_area'):

            self.__setattr__(attrib, None)

        self.load_from_file()

    def get_location(self):

        return self.cont_path

    def load_from_file(self):

        f = config_file.Config_File(self.conf_path)

        #Version
        self.version = f['version']
        if self.version is None:
            self.version = 0

        #Marker path and name
        self.marker_path = f['marker_path']
        if self.marker_path is not None:
            self.marker_name = self.marker_path.split(os.sep)[-1]
        else:
            self.marker_name = None

        #Marker count
        try:
            self.marker_count = int(f['marker_count'])
        except:
            self.marker_count = 0

        #Marker positions
        if self.marker_count is None:
            self.marker_positions = None
        else:
            m_str = 'marking_{0}'
            markings = list()
            for m in range(self.marker_count):
                markings.append(f[m_str.format(m)])
            self.marker_positions = markings

        #Plate areas
        p_str = 'plate_{0}_area'
        p = 0
        plates = list()
        while f[p_str.format(p)] is not None:
            plates.append(f[p_str.format(p)])
            p += 1
        self.plate_areas = plates

        #Grayscale
        self.grayscale = f['grayscale']
        self.grayscale_area = f['grayscale_area']

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

    def __init__(self, paths, app_config):

        self._paths = paths
        self._app_config = app_config
        self._fixtures = None
        self.update()

    def __getitem__(self, fixture):

        if exists(fixture):
            return self._fixtures[fixture]

        return None

    def exists(self, name):

        return self._fixtures is not None and name in self._fixtures

    def update(self):

        directory = self._paths.fixtures
        extension = ".config"

        list_fixtures = map(lambda x: x.split(extension, 1)[0],
                            [file for file in os.listdir(directory)
                                if file.lower().endswith(extension)])

        self._fixtures = dict()
        paths = self._paths

        for f in list_fixtures:
            if f.lower() != "fixture":
                fixture = Fixture_Settings(directory, f, paths)

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
