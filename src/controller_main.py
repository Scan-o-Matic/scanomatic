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
import sys

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
import src.controller_config as controller_config
#Resources
import src.controller_calibration as controller_calibration
import src.resource_os as resource_os
import src.resource_scanner as resource_scanner
import src.resource_fixture as resource_fixture
import src.resource_path as resource_path
import src.resource_app_config as resource_app_config

#
# EXCEPTIONS
#

class UnknownContent(Exception): pass

#
# CLASSES
#


class Unbuffered:
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)


class Controller(controller_generic.Controller):

    def __init__(self, model, view, program_path, logger=None):

        super(Controller, self).__init__(None, view=view, model=model,
            logger=logger)
        """
        self._model = model
        self._view = view
        """

        self.paths = resource_path.Paths(root=program_path)
        self.set_simple_logger()
        self.fixtures = resource_fixture.Fixtures(self.paths)
        self.config = resource_app_config.Config(self.paths)
        self.scanners = resource_scanner.Scanners(self.paths, self.config)
        #Subprocs
        self.subprocs = controller_subprocs.Subprocs_Controller(self)
        self.add_subprocess = self.subprocs.add_subprocess
        self.add_subcontroller(self.subprocs)
        self._view.populate_stats_area(self.subprocs.get_view())
        self._view.show_notebook_or_logo()

        if view is not None:
            view.set_controller(self)

    def set_simple_logger(self):

        stdout = open(self.paths.log_main_out, 'w', 0)
        sys.stdout = Unbuffered(stdout)
        stderr = open(self.paths.log_main_err, 'w', 0)
        sys.stderr = Unbuffered(stderr)

    def close_simple_logger(self):

        sys.stdout.close()
        sys.stderr.close()

    def add_contents_by_controller(self, c):

        page = c.get_view()
        title = c.get_page_title()
        self._view.add_notebook_page(page, title, c)
        self.add_subcontroller(c)

    def add_contents_from_controller(self, page, title, c):

        self._view.add_notebook_page(page, title, c)

    def add_contents(self, widget, content_name):

        m = self._model
        if content_name in ('analysis', 'experiment', 'calibration', 'config'):
            title = m['content-page-title-{0}'.format(content_name)]
        else:
            err = UnknownContent("{0}".format(content_name))
            raise err

        if content_name == 'analysis':
            c = controller_analysis.Analysis_Controller(self,
                    logger=self._logger)
        elif content_name == 'experiment':
            c = controller_experiment.Experiment_Controller(self,
                    logger=self._logger)
        elif content_name == 'calibration':
            c = controller_calibration.Calibration_Controller(
                    self, logger=self._logger)
        elif content_name == 'config':
            c = controller_config.Config_Controller(
                    self, logger=self._logger)
        else:
            err = UnknownContent("{0}".format(content_name))
            raise err

        page = c.get_view()
        self._view.add_notebook_page(page, title, c)
        self.add_subcontroller(c)

    def remove_contents(self, widget, page_controller):

        view = self._view
        if page_controller.ask_destroy():
            page_controller.destroy()
            view.remove_notebook_page(widget)
            for i, c in enumerate(self._controllers):
                if c == page_controller:
                    del self._controllers[i]
                    break

    def ask_quit(self, *args):

        #INTERIM SOLUTION, SHOULD HANDLE SUBPROCS
        if self.ask_destroy():

            for c in self._controllers:
                c.destroy()

            self.get_view().destroy()
            gtk.main_quit()
            return False

        return True
