#!/usr/bin/env python
"""GTK-GUI for manual correction of grid"""

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
import gobject
import os, os.path, sys
import re
import time
import types

#
# SCANNOMATIC LIBRARIES
#

import resource_config as conf
import resource_fixture as r_fixture

#
# SCANNING RE GRIDDING
#


class Grid(gtk.Frame):
    def __init__(self, owner):


        self.owner = owner
        self.DMS = self.owner.DMS
        self._program_config_root = self.owner._program_config_root

        self._log = None
        self._gui_updating = False
        self._log_file_path = None
        self._run_file_path = None
        self._matrices = None
        self.pinnings =  {}
        self._diagnostic_images = []
        self._fixture_name = "fixture_a"
        self.fixture = None
        self._pinning_matrices = None
        self._p_uuid = None 
        self._grid_adjustments = None 
        self._grid_image = None
        self._re_pattern_diag_imgs = "grid-image as file '(.*)' for plate (.)"
        self._re_pattern_imgs = ".*Running analysis on '[^']*'".format(os.sep)
        self._re_pattern_pinnings = "Positions([^\]]*])"
        self.pinning_matrices = {'A: 8 x 12 (96)':(8,12), 
            'B: 16 x 24 (384)': (16,24), 
            'C: 32 x 48 (1536)': (32,48),
            'D: 64 x 96 (6144)': (64,96),
            '--Empty--': None}

        #Make GTK-stuff
        gtk.Frame.__init__(self, "REMAPPING OF GRIDS")

        vbox = gtk.VBox()
        self.add(vbox)


        #Log file
        hbox = gtk.HBox()
        label = gtk.Label("Log file:")
        hbox.pack_start(label, False, False, 2)

        self._gui_log_file = gtk.Label("")
        hbox.pack_start(self._gui_log_file, False, False, 20)

        button = gtk.Button(label = 'Open')
        button.connect("clicked", self._select_file,'log')
        hbox.pack_end(button, False, False, 2)
        vbox.pack_start(hbox, False, False, 2)

        #Run file
        hbox = gtk.HBox()
        label = gtk.Label("If analysis has been started, please supply the"+\
            " 'analysis.run' file normally in the 'analysis' sub-directory")
        vbox.pack_start(label, False, False, 20)

        label = gtk.Label("Run file:")
        hbox.pack_start(label, False, False, 2)

        self._gui_run_file = gtk.Label("")
        hbox.pack_start(self._gui_run_file, False, False, 20)

        button = gtk.Button(label = 'Open')
        button.connect("clicked", self._select_file,'run')
        hbox.pack_end(button, False, False, 2)
        

        vbox.pack_start(hbox, False, False, 2)

        
        #Matrices-override
        hbox = gtk.HBox()
        self.plates_label = gtk.Label("Plates")
        checkbox = gtk.CheckButton(label="Override pinning settings", use_underline=False)
        checkbox.connect("clicked", self._set_override_toggle)
        self.plate_pinnings = gtk.HBox()
        self.plates_entry = gtk.Entry(max=1)
        self.plates_entry.set_size_request(20,-1)
        self.plates_entry.connect("focus-out-event", self._set_plates)
        self.plates_entry.set_text(str(len(self._matrices or 4*[None])))
        hbox.pack_start(checkbox, False, False, 2)
        hbox.pack_end(self.plate_pinnings, False, False, 2)
        hbox.pack_end(self.plates_entry, False, False, 2)
        hbox.pack_end(self.plates_label, False, False, 2)
        vbox.pack_start(hbox, False, False, 2)


        hbox = gtk.HBox()
        label = gtk.Label("Select plate:")
        hbox.pack_start(label, False, False, 2)
        self.plate_selector = gtk.combo_box_new_text()                   
        self.plate_selector.connect("changed", self._set_active_plate)
        hbox.pack_start(self.plate_selector, False, False, 2)
        vbox.pack_start(hbox, False, False, 2)
   
        self.review_hbox = gtk.HBox()
        vbox2 = gtk.VBox()
        label = gtk.Label("Gridding as it was done automatically:")
        self.pin_img = gtk.Image()
        vbox2.pack_start(label, False, False, 2)
        vbox2.pack_start(self.pin_img, False, False, 2)
        self.review_hbox.pack_start(vbox2, False, False, 2)

        vbox2 = gtk.VBox()
        label = gtk.Label("Remapping of grid:")
        vbox2.pack_start(label, False, False, 2)
        label = gtk.Label("For now only done in simple shifting in discrete steps")
        vbox2.pack_start(label, False, False, 20)
        hbox = gtk.HBox()
        label = gtk.Label("How many steps should the columns be moved (positive numbers = to the right)?")
        hbox.pack_start(label, False, False, 2)
        self._gui_column_shift = gtk.Entry(2)
        self._gui_column_shift.set_text("0")
        hbox.pack_end(self._gui_column_shift, False, False, 2)
        vbox2.pack_start(hbox, False, False, 2)
        hbox = gtk.HBox()
        label = gtk.Label("How many steps should the rows be moved (positive numbers = down)?")
        hbox.pack_start(label, False, False, 2)
        self._gui_row_shift = gtk.Entry(2)
        self._gui_row_shift.set_text("0")
        hbox.pack_end(self._gui_row_shift, False, False, 2)
        vbox2.pack_start(hbox, False, False, 2)

        self.review_hbox.pack_end(vbox2, False, False, 2)
        vbox.pack_start(self.review_hbox,False, False, 2)

        vbox.show_all()
        self.review_hbox.hide()

    def _set_active_plate(self, widget=None, event=None, data=None):

        active = str(widget.get_active())
        if active >= 0:

            dbg_img = [i[0] for i in self._diagnostic_images if i[1] == active]

            if len(dbg_img) > 0:
                self.pin_img.set_from_file(dbg_img[0])

            try:
                print self.pinnings[int(active)]
            except:
                pass

            self.review_hbox.show()
        else: 
            self.review_hbox.hide()

    def set_diagnostic_image_resources(self):

        if self._run_file_path:
            try:
                fs = open(self._run_file_path,'r')
            except:
                fs = None

            if fs is not None:

                fs_lines = fs.read()

                fs.close()

                self._diagnostic_images = re.findall(self._re_pattern_diag_imgs, fs_lines)
                try:
                    self._grid_image = [i for i in re.findall(self._re_pattern_imgs, 
                        fs_lines) if self._diagnostic_images[0][0][-16:-12] in i][0]
                except:
                    self._grid_image = []
                    self.DMS('Regridding', "Error parsing diagnostic images {0}"\
                        .format(self._diagnostic_images), 110, debug_level='warning')

                pinnings = re.findall(self._re_pattern_pinnings, fs_lines)
                pinnings = [re.sub(" +",",", p.replace("\n","")).replace(",","",1) for p in pinnings]
                try:
                    pinnings = map(eval, pinnings)
                except:
                    pinnings = None

                true_plate = 0
                self.pinnings = {}

                if pinnings is not None:
                    for plate in xrange(len(pinnings)/2):
                        if self._pinning_matrices:
            
                            while true_plate < len(self._pinning_matrices) and\
                                self._pinning_matrices[true_plate] is None:

                                true_plate += 1
                            if true_plate > len(self._pinnings_matrices):
                                break
                        else:
                            true_plate = plate

                        self.pinnings[true_plate] = (pinnings[plate], pinnings[plate+1])

        if not self._run_file_path or fs is None and self._log is not None:

            self._diagnostic_images = []
            image_dicts = self._log.get_all("%n")
            if len(image_dicts) > 100:
                self._grid_image = image_dicts[100]['File']
            else:
                self._grid_image = image_dicts[-1]['File']

    def set_log_file(self):

        if self._log_file_path:

            self._log = conf.Config_File(self._log_file_path)
        else:
            self._log = None

        if self._log is not None:

            image_dicts = self._log.get_all("%n")

            if 'Fixture' in image_dicts[0].keys():
                self._fixture_name = image_dicts[0]['Fixture']

            if 'Pinning Matrices' in image_dicts[0].keys():
                self._pinning_matrices = image_dicts[0]['Pinning Matrices']

            if 'UUID' in image_dicts[0].keys():
                self._p_uuid =  image_dicts[0]['UUID']
                
            if 'Grid Adjustments' in image_dicts[0].keys():
                self._grid_adjustments =  image_dicts[0]['Grid Adjustments']

            if self._fixture_name is not None:
                self.fixture = r_fixture.Fixture_Settings(\
                    self._program_config_root + os.sep + "fixtures", 
                    fixture = self._fixture_name)

                while len(self.plate_selector.get_model()) > 0:
                    self.plate_selector.remove_text(0)

    
                if self._pinning_matrices is not None:
                    self._set_number_of_plates(len(self._pinning_matrices))
                else:
                    self._set_number_of_plates(len(self.fixture.get_plates_list()))

    def _set_number_of_plates(self, n):
        for i in xrange(n):

            self.plate_selector.append_text("Plate {0}".format(i))

        self.plate_selector.set_sensitive(True)

    def _set_override_toggle(self, widget=None, event=None, data=None):

        if widget.get_active():
            self.plates_entry.show()
            self.plate_pinnings.show()
            self._set_plates(widget=self.plates_entry)
            self.plates_label.set_text("Plates:")
        else:
            self.plates_entry.hide()
            self.plate_pinnings.hide()
            self.plates_label.set_text("(Using the pinning matrices specified in the log-file)")
            self.pinning_string = None

    def _set_plates(self, widget=None, event=None, data=None):

        pass

    def _select_file(self, widget=None, event=None, data=None):

        newlog = gtk.FileChooserDialog(title="Select {0} file".format(event), 
            action=gtk.FILE_CHOOSER_ACTION_OPEN, 
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, 
            gtk.STOCK_APPLY, gtk.RESPONSE_APPLY))

        f = gtk.FileFilter()
        f.set_name("Valid {0} files".format(event))
        f.add_mime_type("text")
        f.add_pattern("*.{0}".format(event))
        newlog.add_filter(f)


        result = newlog.run()
        
        if result == gtk.RESPONSE_APPLY:

            if event == 'run':
                self._run_file_path = newlog.get_filename()
                self._gui_run_file.set_text("Run file: %s" % \
                    str(self._run_file_path))
                self.set_diagnostic_image_resources()
            elif event == 'log':
                self._log_file_path = newlog.get_filename()
                self._gui_log_file.set_text("Log file: %s" % \
                    str(self._log_file_path))
                self.set_log_file()

        newlog.destroy()

