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

        #INSTALL
        frame = gtk.Frame(model['config-install'])
        self.pack_start(frame, False, False, PADDING_LARGE)
        hbox = gtk.HBox(False, 0)
        frame.add(hbox)

        label = gtk.Label(model['config-desktop-short_cut'])
        hbox.pack_start(label, False, False, PADDING_SMALL)
        button = gtk.Button()
        button.set_label(model['config-desktop-short_cut-make'])
        button.connect("clicked", controller.set_desktop_shortcut)
        hbox.pack_end(button, False, False, PADDING_SMALL)

        #BACKUP
        frame = gtk.Frame(model['config-backup'])
        self.pack_start(frame, False, False, PADDING_LARGE)
        hbox = gtk.HBox(False, 0)
        frame.add(hbox)

        label = gtk.Label(model['config-log-save'])
        button = gtk.Button(label=model['config-log-save-button'])
        button.connect("clicked", controller.make_state_backup)
        hbox.pack_start(label, False, False, PADDING_SMALL)
        hbox.pack_end(button, False, False, PADDING_SMALL)

        #SETTINGS
        frame = gtk.Frame(model['config-settings'])
        self.pack_start(frame, False, False, PADDING_LARGE)
        table = gtk.Table(rows=4, columns=3)
        frame.add(table)

        ##POWER MANAGER
        label = gtk.Label(model['config-pm'])
        table.attach(label, 0, 1, 0, 1)
        
        self._pm_usb = gtk.RadioButton(label=model['config-pm-usb'])
        self._pm_usb_signal = self._pm_usb.connect('toggled',
            controller.set_pm_type, 'usb')
        table.attach(self._pm_usb, 1, 2, 0, 1)
        self._pm_lan = gtk.RadioButton(group=self._pm_usb,
            label=model['config-pm-lan'])
        self._pm_lan_signal = self._pm_lan.connect('toggled',
            controller.set_pm_type, 'lan')
        table.attach(self._pm_lan, 2, 3, 0, 1)

        ##SCANNERS
        label = gtk.Label(model['config-scanners'])
        table.attach(label, 0, 1, 1, 2)
        self._scanners = gtk.Entry(1)
        self._scanners_signal = self._scanners.connect("changed",
            controller.set_scanners)
        table.attach(self._scanners, 1, 2, 1, 2)

        ##SAVE
        button = gtk.Button(label=model['config-settings-save'])
        table.attach(button, 0, 1, 2, 3)
        self.show_all()

    def set_signal_block(self, s_type):

        if s_type in ('pm', 'usb', 'all'):
            self._pm_usb.handler_block(self._pm_usb_signal)

        if s_type in ('pm', 'lan', 'all'):
            self._pm_lan.handler_block(self._pm_lan_signal)

        if s_type in ('scanners', 'all'):
            self._scanners.handler_block(self._scanners_signal)

    def set_signal_unblock(self, s_type):

        if s_type in ('pm', 'usb', 'all'):
            self._pm_usb.handler_unblock(self._pm_usb_signal)

        if s_type in ('pm', 'lan', 'all'):
            self._pm_lan.handler_unblock(self._pm_lan_signal)

        if s_type in ('scanners', 'all'):
            self._scanners.handler_unblock(self._scanners_signal)
