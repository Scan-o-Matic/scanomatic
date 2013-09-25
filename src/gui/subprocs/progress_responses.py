#!/usr/bin/env python
"""Progress_Responses is intended and only works by extending a controller class."""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# INTERNAL DEPENDENCIES
#

import view_subprocs as view_subprocs

#
# CLASSES
#


class Progress_Responses(object):

    def produce_live_projects(self, widget):

        self.get_top_controller().add_contents_from_controller(
            view_subprocs.Live_Projects(self, self._model,
                                        self._specific_model),
            self._model['live-projects'],
            self)

    def produce_free_scanners(self, widget):

        self.get_top_controller().add_contents_from_controller(
            view_subprocs.Free_Scanners(self, self._model,
                                        self._specific_model),
            self._model['free-scanners'],
            self)

    def produce_running_analysis(self, widget):

        self.get_top_controller().add_contents_from_controller(
            view_subprocs.Running_Analysis(self, self._model,
                                           self._specific_model),
            self._model['running-analysis'],
            self)

    def produce_errors_and_warnings(self, widget):

        self.get_top_controller().add_contents_from_controller(
            view_subprocs.Errors_And_Warnings(self, self._model,
                                              self._specific_model),
            self._model['collected-messages'],
            self)

    def produce_inspect_gridding(self, widget, prefix, data={}):

        a_file = self._project_progress.get_analysis_path(prefix)

        data['stage'] = 'inspect'
        data['analysis-run-file'] = a_file
        data['project-name'] = prefix

        tc = self.get_top_controller()
        tc.add_contents(widget, 'analysis', **data)

    def produce_upload(self, widget, prefix):

        data = {'launch-filezilla': True}
        self.produce_inspect_gridding(widget, prefix, data=data)

    def produce_launch_analysis(self, widget, prefix):
        """produce_launch_analysis, short-cuts to displaying a
        view for analysing a specific project as defined in prefix
        """

        proj_dir = self._project_progress.get_path(prefix)
        data = {
            'stage': 'project',
            'analysis-project-log_file_dir': proj_dir,
            'analysis-project-log_file':
            self._project_progress.get_first_pass_file(prefix)}

        tc = self.get_top_controller()
        tc.add_contents(widget, 'analysis', **data)

    def produce_running_experiments(self, widget):

        self.get_top_controller().add_contents_from_controller(
            view_subprocs.Running_Experiments(
                self, self._model,
                self._specific_model),
            self._model['running-experiments'],
            self)
