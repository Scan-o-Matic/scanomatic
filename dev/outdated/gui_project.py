#!/usr/bin/env python
"""GTK-GUI for running analysis on a project"""

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

import gtk, pango
import gobject
import os, os.path, sys#, shutil
import re
import time
import types
from subprocess import call, Popen
import signal

#
# SCANNOMATIC LIBRARIES
#

import src.resource_project_log as rpl
import src.resource_power_manager as power_manager
import src.resource_os as os_tools


#
# SCANNING EXPERIMENT GUI
#


class Project_Analysis_Running(gtk.Frame):
    def __init__(self, owner, gtk_target, log_file, matrices, 
        watch_colony = None, supress_other = False, watch_time = "1", 
        analysis_output='analysis', manual_grid=False):

        self.USE_CALLBACK = owner.USE_CALLBACK

        self.owner = owner
        self.DMS = self.owner.DMS

        self._gtk_target = gtk_target

        self._subprocess_pid = None

        self._analyis_script_path = self.owner._program_config_root + os.sep + "analysis.py"

        self._analysis_running = False

        if matrices != None and matrices != "":
            self._matrices = matrices
        else:
            self._matrices = None

        self._watch_colony = watch_colony
        self._supress_other = supress_other
        self._watch_time = watch_time
        self._manual_grid = manual_grid

        self._analysis_log_dir = os.sep.join(log_file.split(os.sep)[:-1]) + os.sep 
        self._analysis_sub_proc = None
        self._analysis_output = analysis_output
        self._analysis_log_file_path = log_file
        self._analysis_run_file = self._analysis_log_dir + analysis_output + os.sep + 'analysis.run'
        self._analysis_re_pattern = "Running analysis on '([^']*)'".format(os.sep)
        self._start_time = time.time()

        #Make GTK-stuff
        gtk.Frame.__init__(self, "Running Analysis On: {0}".format(log_file))

        vbox = gtk.VBox()
        self.add(vbox)

        #Time status
        hbox = gtk.HBox()
        self._gui_analysis_start = gtk.Label("Start time: %s" % str(\
            time.strftime("%Y-%m-%d %H:%M",
            time.localtime(time.time()))))
        self._gui_timer = gtk.Label("In about 2 min you will get feedback on ETA etc.")
        button = gtk.Button('Teminate Analysis')
        button.connect('clicked', self._terminate)

        hbox.pack_start(self._gui_analysis_start, False, False, 2)
        hbox.pack_start(self._gui_timer, False, False, 20)
        hbox.pack_end(button, False, False, 2)        
        vbox.pack_start(hbox, False, False, 2)

        #Run status
        self._gui_status_text = gtk.Label("")
        vbox.pack_start(self._gui_status_text, False, False, 2)

        self._gtk_target.pack_start(self, False, False, 20)
        self.show_all()

        gobject.timeout_add(1000*5, self._run)

    def get_pid(self):

        return self._subprocess_pid

    def get_run_file_contents(self):

        fs_lines = ""
        try:
            fs = open(self._analysis_run_file,'r')

        except:
            fs = None

        if fs is not None:
            fs_lines = fs.read()
            fs.close()

        return fs_lines

    def _run(self):

        if self._analysis_running:

            if self._analysis_sub_proc.poll() != None:
                self._analysis_log.close()
                self._gui_status_text.set_text("Analysis done...")
                fs_lines = self.get_run_file_contents()
                error_pattern = re.compile(r"^ +ERROR: (.+)\n", re.MULTILINE)
                warning_pattern = re.compile(r"^ +WARNING: (.+)\n", re.MULTILINE)
                critical_pattern = re.compile(r"^ +CRITICAL: (.+)\n", re.MULTILINE)
                crash_pattern = re.compile(r"^ +CRITICAL: Uncaught exception([^:]+)", 
                    re.MULTILINE)

                crashed = re.findall(crash_pattern, fs_lines)
                if len(crashed) > 0:

                    crash_text = crashed[0].split("Traceback")[0].strip()[3:]

                    self.DMS("Analysis Project", crash_text, 'LD', debug_level='critical')

                else:

                    criticals = re.findall(critical_pattern, fs_lines)
                    errors = re.findall(error_pattern, fs_lines)
                    warnings = re.findall(warning_pattern, fs_lines)

                    if len(criticals) > 0:


                        self.DMS("Analyse Project", "The analysis produced " +\
                        "{0} critical exception{1}:\n\n".format(len(criticals),
                        ['','s'][len(criticals)>1]) +\
                        "\n".join(criticals) +\
                        "\n\n Further it produced {0} error{1}.".format(len(errors),
                        ['s',''][len(errors) == 1]) +\
                        " and {0} warning{1}.".format(len(warnings),
                        ['s',''][len(warnings) == 1]),
                        'DL', debug_level='critical')
                    elif len(errors) > 0:
                        self.DMS("Analyse Project", "The analysis was completed," +\
                        " but with {0} error{1}.".format(len(errors),
                        ['','s'][len(errors) > 1]) +\
                        "and {0} warning{1}.".format(len(warnings),
                        ['s',''][len(warnings) == 1]), level="DL", debug_level="warning")

                    elif len(warnings) > 0:
                        self.DMS("Analyse Project", "The analysis was completed," +\
                        " but with {0} warning{1}.".format(len(warnings),
                        ['','s'][len(warnings) > 1]), level="LA", debug_level="warning")
 
                gobject.timeout_add(1000*60*1, self.destroy)          
            else:

                timer_text = "Run-time: %d min " % int((time.time() \
                    - float(self._start_time)) / 60)

                try:                
                    fs = open(self._analysis_log_dir +  ".analysis.log", 'r')
                    timer_text += fs.readlines()[-1].split("\r")[-1].split("]")[-1]
                    fs.close()

                except:

                    timer_text += "No info yet on progress and ETA"

                self._gui_timer.set_text(timer_text)
                fs_lines = self.get_run_file_contents()

                if fs_lines != "":
                    re_hits = re.findall(self._analysis_re_pattern, fs_lines)
                    status_text = "Currently anlysing image: {0}".format(\
                        re_hits[-1].split(os.sep)[-1])
                    self._gui_status_text.set_text(status_text)
                else:
                    self._gui_status_text.set_text("Can't find analysis"\
                        + " run-log, no status will be produced")

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
                '--xml-omit-measures','mean,median,IQR,IQR_mean,centroid,perimeter,area',
                '--debug', 'info', '--manual-grid', str(self._manual_grid)]

            if self._matrices is not None:
                analysis_query += ["-m", self._matrices]
            if self._watch_colony is not None:
                analysis_query += ["-w", self._watch_colony]
            if self._supress_other is True: 
                analysis_query += ["-s", "True"]

            self.DMS("ANALYSE PROJECT", "Executing {0}".format(analysis_query), level="L")

            self._analysis_sub_proc = Popen(map(str, analysis_query), 
                stdout=self._analysis_log, shell=False)
            self._subprocess_pid = self._analysis_sub_proc.pid

            gobject.timeout_add(1000*30, self._run)

    def _terminate(self, widget=None, event=None, data=None, ask=True):

        if ask:
            dialog = gtk.MessageDialog(self.owner.window, gtk.DIALOG_DESTROY_WITH_PARENT,
               gtk.MESSAGE_WARNING, gtk.BUTTONS_NONE,
               "This will terminate the ongoing analyis prematurely.")

            dialog.add_button(gtk.STOCK_YES, gtk.RESPONSE_YES)
            dialog.add_button(gtk.STOCK_NO, gtk.RESPONSE_NO)
            resp = dialog.run()
            dialog.destroy()


        if not ask or resp == gtk.RESPONSE_YES:
            if self._analysis_sub_proc != None and self._analysis_sub_proc.poll() == None:
                term_success = False
                if os_tools.OS().name == 'linux':

                    term_success = True
                    os.kill(self._analysis_sub_proc.pid, signal.SIGTERM)

                elif os_tools.OS().name == 'windows':

                    term_success = True
                    import ctypes
                    PROCESS_TERMINATE = 1 
                    handle = ctypes.windll.kernel32.OpenProcess(\
                        PROCESS_TERMINATE, False, self._analysis_sub_proc.pid)
                    ctypes.windll.kernel32.TerminateProcess(handle, -1)
                    ctypes.windll.kernel32.CloseHandle(handle)

                else:
                    self.DMS("ANALYSE PROJECT", 
                        "OS not supported for manual termination of subprocess"+\
                        " {0}".format(self._analysis_sub_proc.pid), 
                        level="DL", debug_level="error")
                if term_success:
                    self._subprocess_pid = None
                    self._gui_status_text.set_text("Analysis termitating manually")
                    self.DMS("ANALYSE PROJECT", "Analysis termitating manually", level="LA")
            else:
                self.destroy()

class Project_Analysis_Setup(gtk.Frame):
    def __init__(self, owner):

        self._gui_updating = False
        self._owner = owner
        self.DMS = owner.DMS
        
        self._matrices = None
        self._watch_colony = None
        self._supress_other = False
        self._watch_time = '-1'
        self._manual_grid = False
        self._analysis_output = 'analysis'
        self._analysis_log_file_path = None

        self.pinning_matrices = {'A: 8 x 12 (96)':(8,12), 
            'B: 16 x 24 (384)': (16,24), 
            'C: 32 x 48 (1536)': (32,48),
            'D: 64 x 96 (6144)': (64,96),
            '--Empty--': None}

        self.pinning_string = None
 
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
        self.output_entry = gtk.Entry()
        self.output_entry.set_text(str(self._analysis_output))
        self.output_entry.connect("focus-out-event", self._eval_input, "analysis_output")
        hbox.pack_start(label, False, False, 2)
        hbox.pack_end(self.output_entry, False, False, 2)
        vbox.pack_start(hbox, False, False, 2)

        #Output directory overwrite warning
        self._gui_warning = gtk.Label("")
        vbox.pack_start(self._gui_warning, False, False, 2)

        #Matrices-override
        hbox = gtk.HBox()
        self.plates_label = gtk.Label("Plates")
        self.override_checkbox = gtk.CheckButton(label="Override pinning settings", use_underline=False)
        self.override_checkbox.connect("clicked", self._set_override_toggle)
        self.plate_pinnings = gtk.HBox()
        self.plates_entry = gtk.Entry(max=1)
        self.plates_entry.set_size_request(20,-1)
        self.plates_entry.connect("focus-out-event", self._set_plates)
        self.plates_entry.set_text(str(len(self._matrices or 4*[None])))
        hbox.pack_start(self.override_checkbox, False, False, 2)
        hbox.pack_end(self.plate_pinnings, False, False, 2)
        hbox.pack_end(self.plates_entry, False, False, 2)
        hbox.pack_end(self.plates_label, False, False, 2)
        vbox.pack_start(hbox, False, False, 2)
        self._set_plates(self.plates_entry)

        #Manual gridding (if exists)
        hbox = gtk.HBox()
        self.manual_gridding_cb = gtk.CheckButton(\
            label="Use manual gridding (if exists in log-file)",
            use_underline=False)
        self.manual_gridding_cb.connect("clicked", self._set_manual_grid_toggle)
        hbox.pack_start(self.manual_gridding_cb, False, False, 2)
        vbox.pack_start(hbox, False, False, 2)

        #Watch-colony
        hbox = gtk.HBox()
        label = gtk.Label("Watch colony (to track and give extremely rich data on):")
        self.watch_entry = gtk.Entry()
        self.watch_entry.set_text(str(self._watch_colony))
        self.watch_entry.connect("focus-out-event",self._eval_input, "watch_colony")
        hbox.pack_start(label, False, False, 2)
        hbox.pack_end(self.watch_entry, False, False, 2)
        vbox.pack_start(hbox, False, False, 2)
       
        #Watch-time 
        hbox = gtk.HBox()
        label = gtk.Label("Watch time (index of when to save images of how gridding worked):")
        self.gui_watch_time_entry = gtk.Entry()
        self.gui_watch_time_entry.set_text(str(self._watch_time))
        self.gui_watch_time_entry.connect("focus-out-event",self._eval_input, "watch_time")
        hbox.pack_start(label, False, False, 2)
        hbox.pack_end(self.gui_watch_time_entry, False, False, 2)
        vbox.pack_start(hbox, False, False, 2)

        #Supress other
        hbox = gtk.HBox()
        label = gtk.Label("Supress analysis of the non-watched colonies:")
        self.supress_entry = gtk.Entry()
        self.supress_entry.set_text(str(self._supress_other))
        self.supress_entry.connect("focus-out-event",self._eval_input, "supress_other")
        hbox.pack_start(label, False, False, 2)
        hbox.pack_end(self.supress_entry, False, False, 2)
        vbox.pack_start(hbox, False, False, 20)
        
        #Start button
        hbox = gtk.HBox()
        self._gui_start_button = gtk.Button("Start")
        self._gui_start_button.connect("clicked", self._start)
        self._gui_start_button.set_sensitive(False)
        hbox.pack_start(self._gui_start_button, False, False, 2)
        vbox.pack_start(hbox, False, False, 2)
        
        vbox.show_all() 
        self._set_override_toggle(widget=self.override_checkbox)

    def _set_manual_grid_toggle(self, widget=None, event=None, data=None):
    
        self._manual_grid = widget.get_active()

    def _set_override_toggle(self, widget=None, event=None, data=None):

        if widget.get_active():
            self.plates_entry.show()
            self.plate_pinnings.show()
            self._set_plates(widget=self.plates_entry)
            self.plates_label.set_text("Plates:")
            self. _build_pinning_string()
        else:
            self.plates_entry.hide()
            self.plate_pinnings.hide()
            self.plates_label.set_text("(Using the pinning matrices specified in the log-file)")
            self.pinning_string = None

    def _set_plates(self, widget=None, event=None, data=None):

        try:
            slots = int(widget.get_text())
        except:
            slots = 0
            widget.set_text(str(slots))
        if slots < 0:
            slots = 0
            widget.set_text(str(slots))
 
        cur_len = len(self.plate_pinnings.get_children()) / 2

        if cur_len < slots:
            for pos in xrange(cur_len, slots):

                label = gtk.Label('#%d' % pos)
                self.plate_pinnings.pack_start(label, False, False, 2)

                dropbox = gtk.combo_box_new_text()                   
                def_key_text = '1536'
                def_key = 0
                for i, m in enumerate(sorted(self.pinning_matrices.keys())):
                    dropbox.append_text(m)
                    if def_key_text in m:
                        def_key = i
                dropbox.connect("changed", self._build_pinning_string)
                self.plate_pinnings.pack_start(dropbox, False, False, 2)
                dropbox.set_active(def_key)
                
            self.plate_pinnings.show_all()
        elif cur_len > slots:
            children = self.plate_pinnings.get_children()
            for i, c in enumerate(children):
                if i >= slots*2:
                    c.destroy()

    def set_defaults(self, log_file=None, manual_gridding=False):

        self.supress_entry.set_text("False")
        self.gui_watch_time_entry.set_text("-1")
        self.watch_entry.set_text("None")
        self.output_entry.set_text("analysis")


        if log_file is not None:
            self._select_log_file(data=log_file)
        else:
            self._gui_analysis_log_file.set_text("")      
            self._gui_start_button.set_sensitive(False)

        self.override_checkbox.set_active(False)
        self.manual_gridding_cb.set_active(True)


    def _build_pinning_string(self, widget=None):

        children = self.plate_pinnings.get_children()
        self.pinning_string = ""
        sep = ":"
        for i in xrange(1, len(children), 2):
            c_active = children[i].get_active()
            c_text = children[i].get_model()[c_active][0]
            self.pinning_string += str(self.pinning_matrices[c_text])
            if i < len(children)-1:
                self.pinning_string += sep

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


            elif widget_name == "analysis_output":

                
                if data == "":
                    data = "analysis"

                self._path_warning(output=data)

                self._analysis_output = data

            widget.set_text(str(data))
            self._gui_updating = False

    def _set_watch_time(self):

        try:
            current_values = map(int, self.gui_watch_time_entry.get_text().split(","))
        except:
            current_values = [-1]

        images_in_run = -1
        if self._analysis_log_file_path is not None:

            try:
                fs = open(self._analysis_log_file_path,'r')
            except:
                fs = None

            if fs is not None:
                fs_lines = fs.read()
                fs.close()
                re_pattern = 'File'
                images_in_run = len(re.findall(re_pattern, fs_lines))
        
        if images_in_run > 0:

            #clean out indices that are too high
            current_values = [x for x in current_values if x < images_in_run]
            
            if current_values == [-1] or current_values ==[]:
                current_values = [images_in_run-1]

        self.gui_watch_time_entry.set_text(",".join(map(str,current_values)))
        self._watch_time = self.gui_watch_time_entry.get_text() 


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
        if data is None:
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
        
        if data is not None or result == gtk.RESPONSE_APPLY:

            if data is not None:
                self._analysis_log_file_path = data
            else:
                self._analysis_log_file_path = newlog.get_filename()

            self._gui_analysis_log_file.set_text("Log file: %s" % \
                str(self._analysis_log_file_path))

            self._path_warning()
            self._set_watch_time()

            self._gui_start_button.set_sensitive(True)
            
        if data is None:
            newlog.destroy()


    def _start(self, widget=None, event=None, data=None):

        self.hide()
        self._owner.analysis_Start_New(widget=self)
