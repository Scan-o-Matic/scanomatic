#!/usr/bin/env python
"""The GTK-GUI view for the general layout"""
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

class Calibration_View(Page):

    def __init__(self, controller, model, top=None, stage=None):

        super(Calibration_View, self).__init__(controller, model,
            top=top, stage=stage)

    def _default_stage(self):

        return Stage_About(self._controller, self._model)

    def _default_top(self):

        return Top_Root(self._controller, self._model)


class Top_Root(Top):

    def __init__(self, controller, model):

        super(Top_Root, self).__init__(controller, model)

        button = gtk.Button()
        button.set_label(model['mode-selection-top-fixture'])
        button.connect("clicked", controller.set_mode, 'fixture')
        self.pack_start(button, False, False, PADDING_MEDIUM)

        button = gtk.Button()
        button.set_label(model['mode-selection-top-poly'])
        button.connect("clicked", controller.set_mode, 'poly')
        self.pack_start(button, False, False, PADDING_MEDIUM)

class Stage_About(gtk.Label):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model

        super(Stage_About, self).__init__()

        self.set_justify(gtk.JUSTIFY_LEFT)
        self.set_markup(model['calibration-stage-about-text'])

        self.show()

class Fixture_Select_Top(Top):

    def __init__(self, controller, model, specific_model):

        super(Fixture_Select_Top, self).__init__(controller, model)
        self._specific_model = specific_model

class Fixture_Select_Stage(gtk.VBox):

    def __init__(self, controller, model, specific_model):

        super(Fixture_Select_Stage, self).__init__(0, False)
        self._controller = controller
        self._model = model
        self._specific_model = specific_model

        #EDIT BUTTON
        self.edit_fixture = gtk.RadioButton(group=None, 
            label=model['fixture-select-radio-edit'])
        self.edit_fixture.connect("clicked",
            self.toggle_new_fixture, False)
        self.pack_start(self.edit_fixture, False, False, PADDING_SMALL)

        #CURRENT FIXTURES
        self.fixtures = gtk.ListStore(str)
        self.treeview = gtk.TreeView(self.fixtures)
        tv_cell = gtk.CellRendererText()
        tv_column = gtk.TreeViewColumn(
            model['fixture-select-column-header'],
            tv_cell, text=0)

        self.treeview.append_column(tv_column)
        
        fixtures = controller.get_top_controller().fixtures.names()
        for f in sorted(fixtures):
            self.fixtures.append([f])

        self.pack_start(self.treeview, False, False, PADDING_LARGE)

        #NEW BUTTON
        self.new_fixture = gtk.RadioButton(group=self.edit_fixture,
            label=model['fixture-select-radio-new'])
        self.new_fixture.connect("clicked",
            self.toggle_new_fixture, True)
        self.pack_start(self.new_fixture, False, False, PADDING_SMALL)

        #NEW NAME
        hbox = gtk.HBox(False, 0)
        self.new_name = gtk.Entry()
        self.new_name.set_sensitive(False)
        hbox.pack_start(self.new_name, True, True, PADDING_MEDIUM)
        self.name_warning = gtk.Image()
        hbox.pack_end(self.name_warning, False, False, PADDING_SMALL)

        self.pack_start(hbox, False, False, PADDING_SMALL)        

        self.show_all()

    def toggle_new_fixture(self, widget, is_new):

        self.treeview.set_sensitive(is_new==False)
        self.new_name.set_sensitive(is_new)
