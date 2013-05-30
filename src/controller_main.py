#!/usr/bin/env python
"""The Main Controller"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import gtk
import sys
import gobject

#
# INTERNAL DEPENDENCIES
#

#Own Model and View
import src.model_main as model_main
import src.view_main as view_main
#Controllers
import src.controller_generic as controller_generic
import src.gui.subprocs.controller_subprocs as controller_subprocs
import src.controller_analysis as controller_analysis
import src.gui.experiment.controller_experiment as controller_experiment
import src.controller_config as controller_config
#Resources
import src.controller_calibration as controller_calibration
import src.resource_scanner as resource_scanner
import src.resource_fixture as resource_fixture
import src.resource_path as resource_path
import src.resource_app_config as resource_app_config

#
# EXCEPTIONS
#


class UnknownContent(Exception):
    pass

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

    def __init__(self, program_path, logger=None, debug_mode=False):

        #PATHS NEED TO INIT BEFORE GUI
        self.paths = resource_path.Paths(root=program_path)

        model = model_main.load_app_model()
        view = view_main.Main_Window(controller=self, model=model)

        super(Controller, self).__init__(None, view=view, model=model,
                                         logger=logger)
        """
        self._model = model
        self._view = view
        """

        if debug_mode is False:
            self.set_simple_logger()

        self.fixtures = resource_fixture.Fixtures(self.paths)
        self.config = resource_app_config.Config(self.paths)
        self.scanners = resource_scanner.Scanners(self.paths, self.config)

        self._view.show_notebook_or_logo()

        view.show_all()
        gobject.timeout_add(71, self._second_init_step)

    def _second_init_step(self):

        #Subprocs
        self.subprocs = controller_subprocs.Subprocs_Controller(self)
        self.add_subprocess = self.subprocs.add_subprocess
        #self.add_subcontroller(self.subprocs)
        view = self._view
        view.populate_stats_area(self.subprocs.get_view())
        view.populate_panel()

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
        self._view.set_current_page(-1)
        self.add_subcontroller(c)

    def add_contents_from_controller(self, page, title, c):

        self._view.add_notebook_page(page, title, c)

    def add_contents(self, widget, content_name, **kwargs):
        """Adds a content to the notebook, and allows for passing of
        content specific keyword kwargs to the content controller such that
        specific view-modes and can be inited (and with specific variables"""

        m = self._model
        if content_name in ('analysis', 'experiment', 'calibration', 'config'):
            title = m['content-page-title-{0}'.format(content_name)]
        else:
            err = UnknownContent("{0}".format(content_name))
            raise err

        if content_name == 'analysis':
            c = controller_analysis.Analysis_Controller(
                self, logger=self._logger, **kwargs)

        elif content_name == 'experiment':
            c = controller_experiment.Experiment_Controller(
                self, logger=self._logger, **kwargs)

        elif content_name == 'calibration':
            c = controller_calibration.Calibration_Controller(
                self, logger=self._logger, **kwargs)

        elif content_name == 'config':
            c = controller_config.Config_Controller(
                self, logger=self._logger, **kwargs)

        else:
            err = UnknownContent("{0}".format(content_name))
            raise err

        page = c.get_view()
        self._view.add_notebook_page(page, title, c)
        self._view.set_current_page(-1)
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

        #CHECK SO NOT ORPHANING SUBPROCS
        if self.subprocs.get_saved() is False:

            if view_main.dialog(
                    self.get_window(),
                    self._model['content-app-close-orphan-warning'],
                    'warning',
                    yn_buttons=True) != 1:

                return True

        #THEN IF UNSAVED EXISTS
        if self.ask_destroy():

            for c in self._controllers:
                c.destroy()

            self.get_view().destroy()
            gtk.main_quit()
            return False

        return True
