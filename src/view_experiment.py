#!/usr/bin/env python
"""The GTK-GUI view for experiments"""
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

import pygtk
pygtk.require('2.0')
import gtk
"""
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
import matplotlib.image as plt_img
import matplotlib.pyplot as plt
import matplotlib.text as plt_text
import matplotlib.patches as plt_patches
"""

#
# INTERNAL DEPENDENCIES
#

from src.view_generic import *

#
# STATIC GLOBALS
#

"""Gotten from view_generic instead
PADDING_LARGE = 10
PADDING_MEDIUM = 4
PADDING_SMALL = 2
"""

#
# CLASSES
#

class Experiment_View(Page):

    def __init__(self, controller, model, top=None, stage=None):

        super(Experiment_View, self).__init__(controller, model,
            top=top, stage=stage)

    def _default_stage(self):

        return Stage_About(self._controller, self._model)

    def _default_top(self):

        return Top_Root(self._controller, self._model)


class Top_Root(Top):

    def __init__(self, controller, model):

        super(Top_Root, self).__init__(controller, model)

        button = gtk.Button()
        button.set_label(model['mode-selection-top-project'])
        button.connect("clicked", controller.set_mode, 'project')
        self.pack_start(button, False, False, PADDING_MEDIUM)

        button = gtk.Button()
        button.set_label(model['mode-selection-top-gray'])
        button.connect("clicked", controller.set_mode, 'gray')
        self.pack_start(button, False, False, PADDING_MEDIUM)

        button = gtk.Button()
        button.set_label(model['mode-selection-top-color'])
        button.connect("clicked", controller.set_mode, 'color')
        self.pack_start(button, False, False, PADDING_MEDIUM)

class Stage_About(gtk.Label):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model

        super(Stage_About, self).__init__()

        self.set_justify(gtk.JUSTIFY_LEFT)
        self.set_markup(model['project-stage-about-text'])

