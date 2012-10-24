#!/usr/bin/env python
"""The Calibration Controller"""
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

import src.model_calibration as model_calibration
import src.view_calibration as view_calibration
import src.controller_generic as controller_generic

#
# EXCEPTIONS
#

class Bad_Stage_Call(Exception): pass
class No_View_Loaded(Exception): pass
class Not_Yet_Implemented(Exception): pass
class UnDocumented_Error(Exception): pass

#
# CLASSES
#

class Calibration_Controller(controller_generic.Controller):

    def __init__(self, window, main_controller):

        super(Calibration_Controller, self).__init__(window, main_controller)

        self._specific_controller = None

    def _get_default_view(self):

        return view_calibration.Calibration_View(self, self._model)

    def _get_default_model(self):

        return model_calibration.model

    def set_mode(self, widget, calibration_mode):

        if calibration_mode == 'fixture':

            self._specific_controller = Fixture_Controller(self)

        elif calibration_mode == "poly":

            err = Not_Yet_Implemented("Mode 'Cell Count Calibration'")

            raise err

        else:

            raise Bad_Stage_Call(calibration_mode)

class Fixture_Controller(object):

    def __init__(self, parent):

        pass
