#!/usr/bin/env python
"""GTK-GUI for configuring application settings."""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.0995"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import pygtk
pygtk.require('2.0')

import gtk, pango
import os, os.path, sys
import types

#
# SCANNOMATIC LIBRARIES
#

import src.resource_config as conf

class Config_GUI(gtk.Frame):
    def __init__(self, owner, conf_file=None, title=None, conf_root=None, conf_path=None):

        self.owner = owner
        self.DMS = self.owner.DMS

        if conf_file is None:
            if conf_root is None:
                self.conf_location =  self.owner._program_config_root + os.sep + conf_path 
            else:
                self.conf_location = conf_root + os.sep + conf_path 

            self.config_file = conf.Config_File(self.conf_location)
        else:
            self.config_file = conf_file
            self.conf_location = self.config_file.get_location()

        title == "APPLICATION CONFIGURATION"
        
        gtk.Frame.__init__(self, title)

        self._GUI_updating = False
        self._changed = False

        vbox = gtk.VBox(False, 0)
        self.add(vbox)

        #NUMBER OF SCANNERS
        hbox = gtk.HBox(False, 0)
        vbox.pack_start(hbox)
        label = gtk.Label("Number of scanners connected:")
        hbox.pack_start(label, False, False, 2)
        self.scanner_count = gtk.Entry(1)
        self.scanner_count.connect("focus-out-event", self.validate_input, 
            (int, "0","number_of_scanners"))
        hbox.pack_end(self.scanner_count, False, False, 10)

        #PROJECT ROOT PATH
        hbox = gtk.HBox(False, 0)
        vbox.pack_start(hbox)
        label = gtk.Label("Projects location:")
        hbox.pack_start(label, False, False, 2)
        self.data_root = gtk.Label("")
        hbox.pack_start(self.data_root, False, False, 2)
        button = gtk.Button("Select")
        button.connect("clicked", self.select_dialog, ("Data root", self.data_root,"data_root",""))
        hbox.pack_end(button, False, False, 2)

        #AUTO-UPDATE
        hbox = gtk.HBox(False, 0)
        vbox.pack_start(hbox)
        self.auto_dev_update = gtk.CheckButton("Automatically pull the newest dev-version from repo.")
        self.auto_dev_update.connect("clicked", self.set_update_pattern)
        self.auto_ver_update = gtk.CheckButton("Automatically pull the newest version from repo.")
        self.auto_ver_update.connect("clicked", self.set_update_pattern)
        hbox.pack_start(self.auto_dev_update, False, False, 2)
        hbox = gtk.HBox(False, 0)
        vbox.pack_start(hbox)
        hbox.pack_start(self.auto_ver_update, False, False, 2)
        

        #MESSAGE LOG FILE PATH
        hbox = gtk.HBox(False, 0)
        vbox.pack_start(hbox)
        label = gtk.Label("Log-files path:")
        hbox.pack_start(label, False, False, 2)
        self.logs_root = gtk.Label("")
        hbox.pack_start(self.logs_root, False, False, 2)
        button = gtk.Button("Select")
        button.connect("clicked", self.select_dialog, 
            ("Log-files root", self.logs_root,"log_file", os.sep + "runtime_{0}.log"))
        hbox.pack_end(button, False, False, 2)

        #MESSAGE LEVELS
        hbox = gtk.HBox(False, 0)
        vbox.pack_start(hbox)

        label = gtk.Label("Default message logging:")
        hbox.pack_start(label, False, False, 2)
        self.loglvl_A = gtk.CheckButton(label="Application")
        self.loglvl_A.connect("clicked", self.set_log_level, "A")
        self.loglvl_L = gtk.CheckButton(label="Log-file")
        self.loglvl_L.connect("clicked", self.set_log_level, "L")
        self.loglvl_D = gtk.CheckButton(label="Dialog")
        self.loglvl_D.connect("clicked", self.set_log_level, "D")
        hbox.pack_start(self.loglvl_A, False, False, 2)
        hbox.pack_start(self.loglvl_L, False, False, 2)
        hbox.pack_start(self.loglvl_D, False, False, 2)

        #SAVE/RESET
        hbox = gtk.HBox(False, 0)
        vbox.pack_start(hbox)

        self.save_button = gtk.Button(label="Save")
        self.save_button.connect("clicked", self.save_data_file)
        self.save_button.set_sensitive(False)
        hbox.pack_end(self.save_button, False, False, 10)

        button = gtk.Button(label="Reset")
        button.connect("clicked", self.reload_data_file)
        hbox.pack_end(button, False, False, 10)

        self._GUI_updating = True
        self.set_values_from_file()
        self._GUI_updating = False
        vbox.show_all()

    def set_log_level(self, widget=None, data=None):

        if not self._GUI_updating:
            log_level = self.config_file.get("log_level","")
            tf = widget.get_active()
            if data in log_level and tf == False:
                log_level_new = ""
                for i,l in enumerate(log_level):
                    if l != data:
                        log_level_new += l
                log_level = log_level_new

            elif data not in log_level and tf == True:

                log_level += data

            self.config_file.set("log_level", log_level)
            self.set_changed(True)

    def validate_input(self, widget=None, event=None, data=None):

        if not self._GUI_updating:
            try:
                data[0](widget.get_text())
            except:
                widget.set_text(data[1])

            self.config_file.set(data[2], widget.get_text())
            self.save_button.set_sensitive(True)

    def select_dialog(self, widget=None, data=None):

        new_path = gtk.FileChooserDialog(title=data[0], 
            action=gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER, 
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_APPLY, gtk.RESPONSE_APPLY))

        result = new_path.run()
        
        if result == gtk.RESPONSE_APPLY:
            data[1].set_text(newroot.get_filename()+data[3])
            self.config_file.set(data[2], data[1].get_text()+data[3])

        new_path.destroy()

    def set_values_from_file(self):
        self.scanner_count.set_text(self.config_file.get("number_of_scanners","0"))
        self.data_root.set_text(self.config_file.get("data_root",""))
        self.logs_root.set_text(self.config_file.get("log_path","runtime_{0}.log"))
        log_level = self.config_file.get("log_level","")
        self.loglvl_A.set_active("A"  in log_level)
        self.loglvl_L.set_active("L" in log_level)
        self.loglvl_D.set_active("D" in log_level)
        try:
            dev_update = bool( int( self.config_file.get("dev_update","0")))
        except:
            dev_update = False

        self.auto_dev_update.set_active(dev_update)

        if dev_update:
            ver_update = False
        else:
            try:
                ver_update = bool( int( self.config_file.get("ver_update","0")))
            except:
                ver_update = False

            self.auto_ver_update.set_active(ver_update)
        self.set_update_pattern(force_update=True)

    def set_update_pattern(self, widget=None, data=None, force_update=False):

        if not self._GUI_updating or force_update:
            if not force_update:
                self._GUI_updating = True

            if self.auto_dev_update.get_active():
                self.auto_ver_update.set_sensitive(False)
                self.auto_ver_update.set_active(False)
            else:
                self.auto_ver_update.set_sensitive(True)

            self.config_file.set("dev_update", int( self.auto_dev_update.get_active()))
            self.config_file.set("ver_update", int( self.auto_ver_update.get_active()))

            self.set_changed(True)

            if not force_update:
                self._GUI_updating = False

    def set_changed(self, value):
        self._changed = value
        self.save_button.set_sensitive(value)

    def reload_data_file(self, widget=None, event=None, data=None):
        self._GUI_updating = True
        if self.config_file.reload():
            self.set_changed(False)
            self.set_values_from_file()
        else:
            self.DMS("Config","Could not reload config file")
        self._GUI_updating = False


    def save_data_file(self, widget=None, event=None, data=None):
        self.config_file.save()
        self.owner.set_installed_scanners()
        self.DMS("CONFIGURATION", "Changes have been saved","LA",debug_level="info")
        self.set_changed(False)

class Config_TreeView(gtk.TreeView):
    def __init__(self, owner):

        self.owner = owner

        self._settings_list = None

        self.build_settings_list()

        self._treestore = gtk.TreeStore(int,str, str)
        
        for f, root in enumerate(self._settings_list):
            root_iter = self._treestore.append(None, [f,None, root[0]])
            for key, item in self._settings_list[f][1].items():
                self._treestore.append(root_iter, [f, key, item])
                                      
        gtk.TreeView.__init__(self, self._treestore)

        self._columns = []
        self._columns.append(gtk.TreeViewColumn('Setting'))
        self._columns.append(gtk.TreeViewColumn('Value'))
                                      
        for i, c in enumerate(self._columns):
            self.append_column(c)
            cell = gtk.CellRendererText()
            cell.connect("edited", self.verify_input)
            #cell.connect("editing-started", self.guide_input)
            self._columns[i].pack_start(cell, True)
            self._columns[i].add_attribute(cell, 'text', i + 1) #Not showing column 0...
            cell.set_property('editable', i)

        self.show_all()

    def build_settings_list(self):
        self._settings_list = []
        for f, c_file in enumerate(self.owner.config_files):
            self._settings_list.append([c_file.get_location(),{}])
            k, v = c_file.items()

            for i, key in enumerate(k):
                if key[0] != "#":
                    self._settings_list[f][1][key] = v[i]

    def add_key(self, key):
        if type(key) == types.StringType:
            treeview_selection = self.get_selection()
            rows = treeview_selection.get_selected_rows()
            treeview_reference = gtk.TreeRowReference(rows[0], rows[1][0])

    def guide_input(self, widget=None, event=None, data=None):
        row = self._treestore.get_iter(data)
        try:
            datatype = self._settings_list[self._treestore.get_value(row, 0)][2]
        except:
            datatype = None

        if datatype == "options":
            options = []
            combobox = gtk.combo_box_new_text()
            for i, val in enumerate(self._settings_list[self._treestore.get_value(row, 0)]):
                if i > 2:
                    if i % 2 == 1:
                        combobox.append_text(val)
            combobox.show()
            combobox.set_active(0)
            dialog = gtk.Dialog(title="Select an option", parent=None, flags=0, buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_APPLY, gtk.RESPONSE_APPLY))
            dialog.vbox.pack_start(combobox, True, True, 2)
            result = dialog.run()
            if result == gtk.RESPONSE_APPLY:
                active = combobox.get_active()
                self._treestore[data][1] = self._settings_list[self._treestore.get_value(row,0)][3+2*int(active)]
            dialog.destroy()
            
    def verify_input(self, widget=None, event=None, data=None):
        row = self._treestore.get_iter(event)
        try:
            conf_file = self._settings_list[self._treestore.get_value(row, 0)]
        except:
            conf_file = None

        try:
            conf_key = self._settings_list[self._treestore.get_value(row, 1)]
        except:
            conf_key = None

        try:
            conf_value = self._settings_list[self._treestore.get_value(row, 2)]
        except:
            conf_value = None

        if conf_file and conf_key:
            self.owner.config_files[int(conf_file)].set(conf_key, conf_value)
            
