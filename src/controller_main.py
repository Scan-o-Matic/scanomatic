#!/usr/bin/env python
"""The Main Controller"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.997"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os
import gtk
import copy

#
# INTERNAL DEPENDENCIES
#

#Own Model and View
import src.model_main as model_main
import src.view_main as view_main
#Controllers
import src.controller_generic as controller_generic
import src.controller_subprocs as controller_subprocs
import src.controller_analysis as controller_analysis
import src.controller_experiment as controller_experiment
#Resources
import src.controller_calibration as controller_calibration
import src.resource_os as resource_os
import src.resource_config as resource_config

#
# EXCEPTIONS
#

class UnknownContent(Exception): pass

#
# CLASSES
#


class Paths(object):

    def __init__(self, program_path, config_file=None):

        self.root = program_path
        self.scanomatic = self.root + os.sep + "run_analysis.py"
        self.src = self.root + os.sep + "src"
        self.analysis = self.src + os.sep + "analysis.py"
        self.config = self.src + os.sep + "config"
        self.fixtures = self.config + os.sep + "fixtures"
        self.images = self.src + os.sep + "images"
        self.marker = self.images + os.sep + "orientation_marker_150dpi.png" 
        self.log = self.root + os.sep + "log"
        self.experiment_root = os.path.expanduser("~") + os.sep + "Documents"
        self.experiment_analysis_relative = "analysis"
        self.experiment_analysis_file_name = "analysis.log"
        self.experiment_first_pass_analysis_relative = "{0}.1_pass.analysis"
        self.experiment_first_pass_log_relative = ".1_pass.log"

class Fixture_Settings(object):

    def __init__(self, dir_path, name, paths):

        self._paths = paths
        self.dir_path = dir_path
        self.file_name = name + ".config"
        self.im_path = dir_path + os.sep + name + ".tiff"
        self.scale = 0.25
        self.name = name.replace("_", " ").capitalize()
        self.marker_name = None

        for attrib in ('marker_path', 'marker_count', 'grayscale',
            'marker_positions', 'plate_areas', 'grayscale_area'):

            self.__setattr__(attrib, None)
             
        self.load_from_file()

    def get_location(self):

        return self.dir_path + os.sep + self.file_name

    def load_from_file(self):

        f = resource_config.Config_File(self.get_location())

        #Version
        self.version = f['version']

        #Marker path and name
        self.marker_path = f['marker_path']
        if self.marker_path is not None:
            self.marker_name = self.marker_path.split(os.sep)[-1]
        else:
            self.marker_name = None

        #Marker count
        self.marker_count = f['marker_count']

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

        print "DP: '{0}'".format(default_pinning)

        model['fixture'] = self.name
        model['plate-areas'] = copy.copy(self.plate_areas)
        model['pinnings-list'] = [default_pinning] * len(self.plate_areas)
        model['marker-count'] = self.marker_count
        model['grayscale'] = self.grayscale
        model['grayscale-area'] = copy.copy(self.grayscale_area)
        model['ref-marker-positions'] = copy.copy(self.marker_positions)
        model['marker-path'] = self.get_marker_path()

class Fixtures(object):

    def __init__(self, paths):

        self._paths = paths
        self._fixtures = None
        self.update()

    def __getitem__(self, fixture):

        if self._fixtures is not None and fixture in self._fixtures:

            return self._fixtures[fixture]

        return None

    def update(self):

        directory = self._paths.fixtures
        extension = ".config"

        list_fixtures = map(lambda x: x.split(extension,1)[0], 
            [file for file in os.listdir(directory) 
            if file.lower().endswith(extension)])

        self._fixtures = dict()
        paths = self._paths

        for f in list_fixtures:
            fixture = Fixture_Settings(directory, f, paths)
            self._fixtures[fixture.name] = fixture

    def names(self):

        if self._fixtures is None:
            return list()

        return self._fixtures.keys()

    def fill_model(self, model):

        fixture=model['fixture']
        if fixture in self._fixtures.keys():

            model['im-original-scale'] = self._fixtures[fixture].scale
            model['im-scale'] = self._fixtures[fixture].scale
            model['im-path'] = self._fixtures[fixture].im_path
            model['fixture-file'] = self._fixtures[fixture].file_name

        else:

            model['im-original-scale'] = 1.0
            model['im-scale'] = 1.0
            model['im-path'] = None
            model['fixture-file'] = model['fixture'].lower().replace(" ","_")


class Scanner(object):

    def __init__(self, paths, config, name):

        self._paths = paths
        self._config = config
        self._name = name
        self._claimed = False
        self._master_process = None

    def _write_to_lock_file(self):

        pass

    def get_claimed(self):

        return self._claimed

    def get_name(self):

        return self._name

    def claim(self):

        if self.get_claimed():
            return False
        else:
            self._set_claimed(True)
            return True

    def free(self):

        self._set_claimed(False)
        return True

    def _set_claimed(self, val, master_proc=None):

        self._claimed = val
        if val == True and master_proc is not None:
            self._master_process = master_proc
        elif val == False:
            self._master_process = None
        
        self._write_to_lock_file()


class Scanners(object):

    def __init__(self, paths, config):

        self._paths = paths
        self._config = config
        self._scanners = dict()
        self._generic_naming = True

    def update(self):

        scanner_count = self._config.number_of_scanners
        if self._generic_naming:
            scanner_name_pattern = "Scanner {0}"
            scanners = [scanner_name_pattern.format(s + 1) for s in xrange(scanner_count)]
        else:
            scanners = self._config.scanner_names

        for s in scanners:

            if s not in self._scanners.keys():
                self._scanners[s] = Scanner(self._paths, self._config, s)

        for s in self._scanners.keys():

            if s not in scanners:

                del self._scanners[s]

    def names(self, available=True):

        self.update()

        scanners = [s_name for s_name, s in self._scanners.items() if available and \
            (s.get_claimed() == False) or True]

        return sorted(scanners)

    def claim(self, scanner_name):

        if scanner_name in self._scanners:
            return self._scanners[scanner_name].claim()
        else:
            print "***WARNING:\tUnknown scanner requested"
            return False

    def free(self, scanner_name):
        if scanner_name in self._scanners:
            return self._scanners[scanner_name].free()
        else:
            print "***WARNING:\tUnknown scanner requested"
            return False


class Config(object):

    def __init__(self, paths):

        self._paths = paths

        #TMP SOLUTION TO BIGGER PROBLEMS
        self.number_of_scanners = 3
        self.scanner_names = list()

    def get_default_analysis_query(self):


        analysis_query =  {
            "-i": "",  # No default input file
            "-o", self._paths.experiment_analysis_relative_path,  # Default subdir
            "-t" : 100,  # Time to set grid
            '--xml-short': 'True',  # Short output format
            '--xml-omit-compartments': 'background,cell',  # Only look at blob
            '--xml-omit-measures':
            'mean,median,IQR,IQR_mean,centroid,perimeter,area',  # only get pixelsum
            '--debug', 'info'  # Report everything that is info and above in seriousness
            }

        return analysis_query

class Controller(controller_generic.Controller):

    def __init__(self, model, view, program_path):

        super(Controller, self).__init__(view, None)
        self._model = model
        self._view = view

        self.paths = Paths(program_path)
        self.fixtures = Fixtures(self.paths)
        self.config = Config(self.paths)
        self.scanners = Scanners(self.paths, self.config)
        #Subprocs
        self.subprocs = controller_subprocs.Subprocs_Controller(self._view, self)
        self.add_subprocess = self.subprocs.add_subprocess
        self.add_subcontroller(self.subprocs)
        self._view.populate_stats_area(self.subprocs.get_view())

        if view is not None:
            view.set_controller(self)

    def add_contents_by_controller(self, c):

        page = c.get_view()
        title = c.get_page_title()
        self._view.add_notebook_page(page, title, c)
        self.add_subcontroller(c)

    def add_contents(self, widget, content_name):

        m = self._model
        if content_name in ('analysis', 'experiment', 'calibration'):
            title = m['content-page-title-{0}'.format(content_name)]
        else:
            err = UnknownContent("{0}".format(content_name))
            raise err

        if content_name == 'analysis':
            c = controller_analysis.Analysis_Controller(self._view, self)
        elif content_name == 'experiment':
            c = controller_experiment.Experiment_Controller(self._view, self)
        elif content_name == 'calibration':
            c = controller_calibration.Calibration_Controller(self._view, self)
        else:
            err = UnknownContent("{0}".format(content_name))
            raise err

        page = c.get_view()
        self._view.add_notebook_page(page, title, c)
        self.add_subcontroller(c)

    def remove_contents(self, widget, page_controller):

        view = self._view
        if page_controller.ask_destroy():
            view.remove_notebook_page(page_controller.get_view())
            for i, c in enumerate(self._controllers):
                if c == page_controller:
                    del self._controllers[i]
                    break

    def ask_quit(self, *args):

        #INTERIM SOLUTION, SHOULD HANDLE SUBPROCS
        if self.ask_destroy():

            gtk.main_quit()
