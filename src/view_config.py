#!/usr/bin/env python
"""The GTK-GUI view for config"""
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

class Config_View(Page):

    def __init__(self, controller, model, top=None, stage=None):

        super(Config_View, self).__init__(controller, model,
            top=top, stage=stage)

    def _default_stage(self):

        return Settings_Cont(self._controller, self._model)

    def _default_top(self):

        return Top_Title(self._controller, self._model)


class Top_Title(Top):

    def __init__(self, controller, model):

        super(Top_Title, self).__init__(controller, model)

        label = gtk.Label()
        label.set_markup(model['config-title'])
        self.pack_start(label, True, True, PADDING_MEDIUM)

        self.show_all()


class Settings_Cont(gtk.VBox):

    def __init__(self, controller, model):

        super(Settings_Cont, self).__init__(False, 0)

        hbox = gtk.HBox(False, 0)
        label = gtk.Label(model['config-desktop-short_cut'])
        hbox.pack_start(label, False, False, PADDING_SMALL)
        button = gtk.Button()
        button.set_label(model['config-desktop-short_cut-make'])
        button.connect("clicked", controller.set_desktop_shortcut)
        hbox.pack_end(button, False, False, PADDING_SMALL)
        self.pack_start(hbox, False, False, PADDING_LARGE)

        hbox = gtk.HBox(False, 0)
        label = gtk.Label(model['config-log-save'])
        button = gtk.Button(label=model['config-log-save-button'])
        button.connect("clicked", controller.make_state_backup)
        hbox.pack_start(label, False, False, PADDING_SMALL)
        hbox.pack_end(button, False, False, PADDING_SMALL)
        self.pack_start(hbox, False, False, PADDING_LARGE)

        self.show_all()
