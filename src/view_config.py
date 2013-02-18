#!/usr/bin/env python
"""The GTK-GUI view for config"""
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

        self._controller = controller
        self._model = model
        tc = controller.get_top_controller()
        config = tc.config
        paths = tc.paths

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

        #UPDATE
        frame = gtk.Frame(model['config-update'])
        self.pack_start(frame, False, False, PADDING_LARGE)
        hbox = gtk.HBox(False, 0)
        frame.add(hbox)

        button = gtk.Button()
        button.set_label(model['config-update-button'])
        button.connect("clicked", controller.run_update)
        hbox.pack_start(button, False, False, PADDING_SMALL)

        self._restart = gtk.Button()
        self._restart.set_label(model['config-update-button'])
        self._restart.connect("clicked", controller.run_restart)
        self._restart.set_sensitive(False)
        hbox.pack_start(self._restart, False, False, PADDING_SMALL)
        
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

        ##EXPERIMENT ROOT
        button = gtk.Button(
            label=model['config-settings-experiments-root'])
        table.attach(button, 0, 1, 2, 3)
        self._experiment_root = gtk.Label(
            "{0}".format(paths.experiment_root))
        button.connect('clicked', self.set_new_experiments_root,
            self._experiment_root)
        table.attach(self._experiment_root, 1, 2, 2, 3)

        ##SAVE
        button = gtk.Button(label=model['config-settings-save'])
        button.connect('clicked', controller.save_current_config)
        table.attach(button, 0, 1, 3, 4)
        self.show_all()

    def set_activate_restart(self):

        self._restart.set_sensitive(True)

    def set_new_experiments_root(self, widget, path_widget):

        file_list = select_dir(path_widget.get_text())

        if file_list is not None:

            self._controller.set_new_experiments_root(file_list)

    def update_experiments_root(self, p):

        self._experiment_root.set_text(p)

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

    def update_scanners(self, scanners):

        self.set_signal_block('scanners')
        self._scanners.set_text(str(scanners))
        self.set_signal_unblock('scanners')

    def update_pm(self, pm_type):

        self.set_signal_block('pm')
        if pm_type == 'lan':
            self._pm_lan.set_active(True)
        elif pm_type == 'usb':
            self._pm_usb.set_active(True)
        self.set_signal_unblock('pm')
