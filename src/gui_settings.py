#!/usr/bin/env python
"""GTK-GUI for configuring application settings."""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.993"
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
    def __init__(self, owner, conf_paths, title=None, conf_root=None):

        self.owner = owner
        self.DMS = self.owner.DMS

        if type(conf_paths) != types.ListType:
            conf_paths = [conf_paths]

        self.conf_locations = []
        for c_path in conf_paths:
            if conf_root == None:
                self.conf_locations.append( self.owner._program_config_root + os.sep + c_path )
            else:
                self.conf_locations.append( conf_root + os.sep + c_path )

        self.config_files = []
        for conf_location in self.conf_locations:
            self.config_files.append( conf.Config_File(conf_location) )
            if title == None:
                title = self.config_files[-1].get("GUI_title")

        special_message = ""

        if title == None:
            title == "CONFIGURATION OF SOMETHING"
            special_message = "If you create a string with the key 'GUI_title' this will" +\
                " show up as the frame - text for this configuration..."
        
        gtk.Frame.__init__(self, title)

        self._GUI_updating = False

        vbox = gtk.VBox(False, 0)
        vbox.show()
        self.add(vbox)

        self.special_message = gtk.Label(special_message)
        self.special_message.show()
        vbox.pack_start(self.special_message, False, False, 10)
        
        self.treeview = Config_TreeView(self)
        self.treeview.show()
        vbox.pack_start(self.treeview, False, False, 2)

        hbox = gtk.HBox()
        hbox.show()
        vbox.pack_start(hbox, False, False, 2)

        label = gtk.Label("Insert key:")
        label.show()
        hbox.pack_start(label, False, False, 2)

        self.new_key = gtk.Entry()
        self.new_key.show()
        hbox.pack_start(self.new_key, False, False, 2)

        button = gtk.Button(label='Put it there')
        button.show()
        button.connect("clicked", self.add_key)
        hbox.pack_start(button, False, False, 2)

        hbox = gtk.HBox()
        hbox.show()
        vbox.pack_start(hbox, False, False, 10)

        button = gtk.Button(label="Save")
        button.connect("clicked", self.save_data_file)
        button.show()
        hbox.pack_end(button, False, False, 10)

        button = gtk.Button(label="Reset")
        button.connect("clicked", self.reload_data_file)
        button.show()
        hbox.pack_end(button, False, False, 10)

    def add_key(self, widget=None, event=None, data=None):
        self.treeview.add_key(self.new_key.get_text())
        self.new_key.set_text("")

    def reload_data_file(self, widget=None, event=None, data=None):
        pass

    def save_data_file(self, widget=None, event=None, data=None):
        for c_file in self.config_files:
            c_file.save()
            self.DMS("CONFIGURATION", "Changes have been saved","LA",debug_level="info")

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
            
