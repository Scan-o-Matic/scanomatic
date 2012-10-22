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

#
# INTERNAL DEPENDENCIES
#

import src.model_main as model_main
import src.view_main as view_main
import src.controller_generic as controller_generic
import src.controller_analysis as controller_analysis
import src.resource_os as resource_os

#
# CLASSES
#

class Controller(controller_generic.Controller):

    def __init__(self, model, view):

        super(Controller, self).__init__(view, None)

        self._model = model
        self._view = view
        self._controllers = list()

        if view is not None:
            view.set_controller(self)

    def add_contents(self, widget, content_name):

        m = self._model
        title = m['content-page-title-{0}'.format(content_name)]
        c = controller_analysis.Analysis_Controller(self._view, self)
        page = c.get_view()
        self._view.add_notebook_page(page, title, c)
        self._controllers.append(c)

    def remove_contents(self, widget, page_controller):

        view = self._view
        if page_controller.ask_destroy():
            view.remove_notebook_page(page_controller.get_view())
            for i, c in enumerate(self._controllers):
                if c == page_controller:
                    del self._controllers[i]
                    break
