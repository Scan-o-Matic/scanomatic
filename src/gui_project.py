#!/usr/bin/env python
"""GTK-GUI for running analysis on a project"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.992"
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
import os, os.path, sys#, shutil
import re
import time
import types
from subprocess import call, Popen

#
# SCANNOMATIC LIBRARIES
#

import src.resource_log_maker as log_maker
import src.resource_log_reader as log_reader
import src.resource_power_manager as power_manager
import src.resource_image as img_base
import src.resource_fixture as fixture_settings
import src.resource_os as os_tools


#
# SCANNING EXPERIMENT GUI
#


class Project_Analysis_Running(gtk.Frame):
    def __init__(self, owner, gtk_target, log_file, matrices, 
        watch_colony = None, supress_other = False, watch_time = 1, 
        analysis_output='analysis'):

        self.USE_CALLBACK = owner.USE_CALLBACK

        self.owner = owner
        self.DMS = self.owner.DMS

        self._gtk_target = gtk_target


        self._analyis_script_path = self.owner._program_config_root + os.sep + "analysis.py"

        self._analysis_running = False

        self._matrices = matrices
        self._watch_colony = watch_colony
        self._supress_other = supress_other
        self._watch_time = watch_time

        self._analysis_log_dir = os.sep.join(log_file.split(os.sep)[:-1]) + os.sep 

        self._analysis_output = analysis_output
        self._analysis_log_file_path = log_file

        self._start_time = time.time()

        #Make GTK-stuff
        gtk.Frame.__init__(self, "Running Analysis On: %s" % log_file)

        vbox = gtk.VBox()
        self.add(vbox)


        #Time status
        hbox = gtk.HBox()
        self._gui_analysis_start = gtk.Label("Start time: %s" % str(\
            time.strftime("%Y-%m-%d %H:%M",
            time.localtime(time.time()))))
        self._gui_timer = gtk.Label("Run-time: %d" % int((time.time() - float(self._start_time)) / 60))
        hbox.pack_start(self._gui_analysis_start, False, False, 2)
        hbox.pack_start(self._gui_timer, False, False, 20)
        vbox.pack_start(hbox, False, False, 2)

        #Run status
        self._gui_status_text = gtk.Label("")
        vbox.pack_start(self._gui_status_text, False, False, 2)

            
        self._gtk_target.pack_start(self, False, False, 20)
        self.show_all()

        gobject.timeout_add(1000*5, self._run)

    def _run(self):

        if self._analysis_running:

            if self._analysis_sub_proc.poll() != None:
                self._analysis_log.close()
                self._gui_status_text.set_text("Analysis complete")
                gobject.timeout_add(1000*60*3, self.destroy)          
            else:
                
                self._gui_timer = gtk.Label("Run-time: %d" % int((time.time() \
                    - float(self._start_time)) / 60))
                gobject.timeout_add(1000*60*2, self._run)
        else:
            self._gui_status_text.set_text("Analysis is running! (This may take several hours)")
            self._analysis_running = True
            self._analysis_log = open(self._analysis_log_dir +  ".analysis.log", 'w')
            analysis_query = [self.owner._program_code_root + os.sep + \
                "analysis.py","-i", self._analysis_log_file_path, 
                "-o", self._analysis_output, "-t", 
                self._watch_time, '--xml-short', 'True', 
                '--xml-omit-compartments', 'background,cell',
                '--xml-omit-measures','mean,median,IQR,IQR_mean,centroid,perimeter,area']

            if self._matrices is not None:
                analysis_query += ["-m", self._matrices]
            if self._watch_colony is not None:
                analysis_query += ["-w", self._watch_colony]
            if self._supress_other is True: 
                analysis_query += ["-s", "True"]

            self.DMS("Executing", str(analysis_query), level=110)

            self._analysis_sub_proc = Popen(map(str, analysis_query), 
                stdout=self._analysis_log, shell=False)
            gobject.timeout_add(1000*60*10, self._run)

          
class Project_Analysis_Setup(gtk.Frame):
    def __init__(self, owner):

        self._gui_updating = False
        self._owner = owner
        self.DMS = owner.DMS
        
        self._matrices = None
        self._watch_colony = None
        self._supress_other = False
        self._watch_time = '-1'

        self._analysis_output = 'analysis'
        self._analysis_log_file_path = None

        #GTK - stuff

        gtk.Frame.__init__(self, "(RE)-START ANALYSIS OF A PROJECT")
        vbox = gtk.VBox()
        self.add(vbox)

        #Log-file selection
        hbox = gtk.HBox()
        self._gui_analysis_log_file = gtk.Label("Log file: %s" % \
            str(self._analysis_log_file_path))
        #label.set_max_width_chars(110)
        #label.set_ellipsize(pango.ELLIPSIZE_MIDDLE)

        button = gtk.Button(label = 'Select Log File')
        button.connect("clicked", self._select_log_file)
        hbox.pack_start(self._gui_analysis_log_file, False, False, 2)
        hbox.pack_end(button, False, False, 2)
        vbox.pack_start(hbox, False, False, 2)

        #Output directory name
        hbox = gtk.HBox()
        label = gtk.Label("Analysis output relative directory:")
        entry = gtk.Entry()
        entry.set_text(str(self._analysis_output))
        entry.connect("focus-out-event", self._eval_input, "analysis_output")
        hbox.pack_start(label, False, False, 2)
        hbox.pack_end(entry, False, False, 2)
        vbox.pack_start(hbox, False, False, 2)

        #Output directory overwrite warning
        self._gui_warning = gtk.Label("")
        vbox.pack_start(self._gui_warning, False, False, 2)

        #Matrices-override
        hbox = gtk.HBox()
        label = gtk.Label("Override matrices")
        entry = gtk.Entry()
        entry.set_text(str(self._matrices))
        entry.connect("focus-out-event", self._eval_input, "override_matrices")
        hbox.pack_start(label, False, False, 2)
        hbox.pack_end(entry, False, False, 2)
        vbox.pack_start(hbox, False, False, 2)

        #Watch-colony
        hbox = gtk.HBox()
        label = gtk.Label("Watch colony:")
        entry = gtk.Entry()
        entry.set_text(str(self._watch_colony))
        entry.connect("focus-out-event",self._eval_input, "watch_colony")
        hbox.pack_start(label, False, False, 2)
        hbox.pack_end(entry, False, False, 2)
        vbox.pack_start(hbox, False, False, 2)
       
        #Watch-time 
        hbox = gtk.HBox()
        label = gtk.Label("Watch time:")
        entry = gtk.Entry()
        entry.set_text(str(self._watch_time))
        entry.connect("focus-out-event",self._eval_input, "watch_time")
        hbox.pack_start(label, False, False, 2)
        hbox.pack_end(entry, False, False, 2)
        vbox.pack_start(hbox, False, False, 2)

        #Supress other
        hbox = gtk.HBox()
        label = gtk.Label("Supress analysis of the non-watched colonies:")
        entry = gtk.Entry()
        entry.set_text(str(self._supress_other))
        entry.connect("focus-out-event",self._eval_input, "supress_other")
        hbox.pack_start(label, False, False, 2)
        hbox.pack_end(entry, False, False, 2)
        vbox.pack_start(hbox, False, False, 20)
        
        #Start button
        hbox = gtk.HBox()
        self._gui_start_button = gtk.Button("Start")
        self._gui_start_button.connect("clicked", self._start)
        self._gui_start_button.set_sensitive(False)
        hbox.pack_start(self._gui_start_button, False, False, 2)
        vbox.pack_start(hbox, False, False, 2)
        
        vbox.show_all() 

    def _eval_input(self, widget=None, event=None, widget_name=None):

        if not self._gui_updating:

            self._gui_updating = True

            data = widget.get_text()

            if widget_name == "supress_other":
                try:
                    data = bool(eval(data))
                except:
                    data = False

                self._supress_other = data

            elif widget_name == "watch_time":
                try:
                    data = data.split(",")
                    data = map(int, data)
                    data = str(data)[1:-1].replace(" ","")
                except:
                    data = "-1"
                
                self._watch_time = data

            elif widget_name == "watch_colony":
                if data == "None":
                    data = None
                else:
                    try:
                        data = data.split(":")
                        data = map(int, data)
                        data = str(data)[1:-1].replace(" ","").replace(",",":")
                    except: 
                        data = None

                self._watch_colony = data

            elif widget_name == "override_matrices":
                if data == "None":
                    data = None
                else:

                    try:
                        data = data.split(":")
                        data = map(eval, data)
                        for i, d in enumerate(data):
                            if d is not None:
                                d = tuple(map(int, d))
                                data[i] = d[:2]
                        data = str(data)[1:-1].replace(" ","")\
                            .replace("),","):").replace(",(",":(")\
                            .replace("e,N","e:N")
                    except:
                        data = None

                self._matrices = data

            elif widget_name == "analysis_output":

                
                if data == "":
                    data = "analysis"

                self._path_warning(output=data)

                self._analysis_output = data

            widget.set_text(str(data))
            self._gui_updating = False

    def _path_warning(self, output=None):

        if output is None:
            output = self._analysis_output

        abs_path = os.sep.join(str(self._analysis_log_file_path).split(os.sep)[:-1]) + \
            os.sep + output 

        if os.path.exists(abs_path):
            self._gui_warning.set_text("Warning: There is already a"\
                +" directory with that name.")
        else:
            self._gui_warning.set_text("") 

    def _select_log_file(self, widget=None, event=None, data=None):
        newlog = gtk.FileChooserDialog(title="Select log file", 
            action=gtk.FILE_CHOOSER_ACTION_OPEN, 
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, 
            gtk.STOCK_APPLY, gtk.RESPONSE_APPLY))

        f = gtk.FileFilter()
        f.set_name("Valid log files")
        f.add_mime_type("text")
        f.add_pattern("*.log")
        newlog.add_filter(f)


        result = newlog.run()
        
        if result == gtk.RESPONSE_APPLY:

            self._analysis_log_file_path = newlog.get_filename()
            self._gui_analysis_log_file.set_text("Log file: %s" % \
                str(self._analysis_log_file_path))

            self._path_warning()

            self._gui_start_button.set_sensitive(True)
            
        newlog.destroy()

    def _start(self, widget=None, event=None, data=None):

        self.hide()
        self._owner.analysis_Start_New(widget=self)
