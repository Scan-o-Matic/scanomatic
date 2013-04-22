#!/usr/bin/env python
"""The Generic Controller"""
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

#
# INTERNAL DEPENDENCIES
#

import src.view_generic as view_generic
import src.resource_logger as resource_logger

#
# CLASSES
#


class Controller(object):

    ALLOW_SEVERAL_ACTIVE = 0
    DISALLOW_SEVERAL_ACTIVE_REPLACE = 1
    DISALLOW_SEVERAL_ACTIVE_IGNORE = 2

    def __init__(self, parent_controller,
                 model=None, view=None,
                 specific_model=None, logger=None,
                 controller_name="U"):

        if logger is not None:
            self._logger = logger
        else:
            self._logger = resource_logger.Fallback_Logger()

        self._parent = parent_controller

        #MODEL SHOULD BE SET BEFORE VIEW!
        self.set_model(model)
        self.set_specific_model(specific_model)
        self.set_view(view)

        self._allow_friendly_remove = True
        self._controller_name = controller_name
        self._controllers = list()

        self._allow_several_active = self.ALLOW_SEVERAL_ACTIVE

    def get_name(self):

        return self._controller_name

    def get_is_same_type(self, other_name):

        if (other_name is None or len(other_name) == 0 or
                self._controller_name is None or
                len(self._controller_name) == 0 or
                other_name[0].lower() != self._controller_name[0].lower()):

            return False

        return True

    def get_allow_several_active(self):

        return self._allow_several_active == self.ALLOW_SEVERAL_ACTIVE

    def get_do_several_active_replace(self):

        return self._allow_several_active == self.DISALLOW_SEVERAL_ACTIVE_REPLACE

    def set_unsaved(self):

        self._allow_friendly_remove = False

    def set_saved(self):

        self._allow_friendly_remove = True

    def get_top_controller(self):

        if self._parent is None:

            return self

        else:

            return self._parent.get_top_controller()

    def add_subcontroller(self, controller):

        self._controllers.append(controller)

    def get_saved(self):

        if self._allow_friendly_remove is False:

            return False

        for c in self._controllers:

            if c.get_saved() is False:

                return False

        return True

    def ask_destroy(self):

        top_controller = self.get_top_controller()
        view = top_controller.get_view()
        m = top_controller.get_model()

        if view is not None and self.get_saved() is False:

            d_text = m['content-page-close']

            return view_generic.dialog(
                view, d_text,
                d_type='question', yn_buttons=True)

        return self._allow_friendly_remove

    def destroy(self):

        pass

    def set_view(self, view=None):

        if view is None:

            view = self._get_default_view()

        self._view = view

    def get_view(self):

        return self._view

    def _get_default_view(self):

        return None

    def set_model(self, model=None):

        if model is None:

            model = self._get_default_model()

        self._model = model

    def _get_default_model(self):

        return None

    def get_model(self):

        return self._model

    def get_window(self):

        tc = self.get_top_controller()
        return tc.get_view()

    def set_specific_model(self, specific_model):

        self._specific_model = specific_model

    def get_specific_model(self):

        if self._specific_model is None:

            self.set_specific_model(dict())

        return self._specific_model
