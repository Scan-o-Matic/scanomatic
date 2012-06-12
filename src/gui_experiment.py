#!/usr/bin/env python
"""GTK-GUI for running a data-collection experiment"""

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
import os, os.path, sys, shutil
import re
import time
import types
import uuid

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
# OS DEPENDENT BEHAVIOUR, NOTE THAT WINDOWS HAVE EXTRA DEPENDENCIES!
#

USER_OS = os_tools.OS()

if USER_OS.name == "linux":
    import src.resource_sane as scanner_backend
elif USER_OS.name == "windows":
    import twain
    import src.resource_twain as scanner_backend
else:
    print "*** ERROR: Scannomatic has not been ported to your OS, so stopping"
    sys.exit(0)

#
# SCANNING EXPERIMENT GUI
#


class Scanning_Experiment(gtk.Frame):
    def __init__(self, owner, parent_window, scanner, interval, counts, prefix,
        description, root, gtk_target, native=True, matrices = None, 
        fixture="fixture_a", include_analysis=True, p_uuid = None):

        if p_uuid is None:
            p_uuid = uuid.uuid1()
        self._p_uuid = uuid

        self.USE_CALLBACK = owner.USE_CALLBACK

        self.owner = owner
        self.DMS = self.owner.DMS

        continue_load = True

        if scanner is None:
            
            continue_load = False
            self.DMS('Experiment', "You're trying to start a project on no scanner.", level=1100,
                debug_level='error')

        if continue_load: 
            try:
                os.mkdir(str(root) + os.sep + str(prefix))
            except:
                self.DMS('Experiment conflict',
                    'An experiment with that prefix already exists...\nAborting.', 
                    level=1100, debug_level='error')
                continue_load = False
            
        gtk.Frame.__init__(self, prefix)

        self._fixture_config_root = self.owner._program_config_root + os.sep + "fixtures"
        self.f_settings = fixture_settings.Fixture_Settings(\
            self._fixture_config_root, fixture=fixture)

        self.pinning_string = ""
        self._matrices = matrices
        self._watch_colony = None
        self._supress_other = False
        self._watch_time = '1'
        self._destroy_request = None

        self._looked_for_scanner = 0
        self._scanning = False
        self._force_quit = False
        self._analysis_output = 'analysis'

        self._include_analysis = include_analysis
        self._analysis_running = False
        self._scanner_id = int(scanner[-1])
        self._scanner_name = scanner
        self._interval_time = float(interval)
        self._iterations_max = int(counts)
        self._prefix = str(prefix)
        self._root = str(root)
        self._gtk_target = gtk_target
        self._last_rejected = -1
        self._description = description
        self._subprocesses = []
        self._analysis_log_file_path = self._root + os.sep + self._prefix + os.sep + self._prefix + ".log"
        self._heatMapPath = self._root + os.sep + self._prefix + os.sep + "progress.png"

        #HACK
        if USER_OS.name == "windows":
            self._power_manager = power_manager.Power_Manager(installed=True, 
                path='"C:\Program Files\Gembird\Power Manager\pm.exe"',
                on_string="-on -PW1 -Scanner1", 
                off_string="-off -PW1 -Scanner1", DMS=self.owner.DMS)

        elif USER_OS.name == "linux":
            self._power_manager = power_manager.Power_Manager(installed=True, 
                path="sispmctl",on_string="-o %d" % self._scanner_id, 
                off_string="-f %d" % self._scanner_id,
                DMS=self.owner.DMS)
            

        self._power_manager.on()

        if USER_OS.name == "windows":
            self._scanner = scanner_backend.Twain_Base()
        elif USER_OS.name == "linux":
            self._scanner = scanner_backend.Sane_Base(self)

        self._scanner.owner = self


        self._next_scan = None
        self._handle = parent_window
        if not self._handle:
            self._handle = int(0)
            
        if native == True:
            self._scan = self._scanner.AcquireNatively
        else:
            self._scan = self._scanner.AcquireByFile
            

        self._end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + 60*self._interval_time*(self._iterations_max+1)))
        self._iteration = 0

        #Make GTK-stuff

        vbox = gtk.VBox()
        
        hbox = gtk.HBox()
        label = gtk.Label("Location: " + self._root + os.sep + self._prefix)
        label.set_max_width_chars(90)
        label.set_ellipsize(pango.ELLIPSIZE_MIDDLE)
        label.show()
        hbox.pack_start(label, False, False, 2)
        label = gtk.Label("Scanner: " + str(scanner))
        label.show()
        hbox.pack_end(label, False, False, 2)
        hbox.show()
        vbox.pack_start(hbox, False, False, 2)

        hbox = gtk.HBox()
        self._measurement_label = gtk.Label("")
        self._measurement_label.show()
        hbox.pack_start(self._measurement_label, False, False, 2)
        self._timer = gtk.Label("")
        self._timer.show()
        hbox.pack_start(self._timer, False, False, 2)
        self._warning_text = gtk.Label("")
        self._warning_text.show()
        hbox.pack_start(self._warning_text, False, False, 10)
        label = gtk.Label("Estimated to end: " + self._end_time)
        label.show()
        hbox.pack_end(label, False, False, 2)
        button = gtk.Button('Terminate experiment')
        button.connect('clicked', self.terminate_experiment)
        button.show()
        hbox.pack_end(button, False, False, 2)
        hbox.show()
        vbox.pack_start(hbox, False, False, 2)
        
        

        vbox.show()
        self.add(vbox)
        self.show()

        self.update_gtk()

            
        self._gtk_target.pack_start(self, False, False, 20)

        if continue_load:
            try:
                shutil.copy(self.f_settings.conf_location,
                    self._root + os.sep + self._prefix + os.sep + fixture + ".config") 
                
            except:
                self.DMS('Scanning',
                    'Could not make a local copy of fixture settings file,'+
                    ' probably the template file will be used in analysis. '+
                    'Lets hope no-one fiddles with it', level=1101, debug_level='warning')
        
            self.log_file({'Prefix':prefix,'Description':description,
                'Interval':self._interval_time, 'Measurments':counts, 
                'Start Time':time.time(), 'Pinning Matrices':self._matrices,
                'Fixture':fixture, 'UUID':str(self._p_uuid)},
                append=False)

            self._loaded = True
            gobject.timeout_add(30, self.running_Experiment) 
            gobject.timeout_add(1000*60, self.running_timer)
        else:
            self._loaded = False
            self._measurement_label.set_text("Conflict in name, aborting...")
            gobject.timeout_add(1000*60*int(self._interval_time), self.destroy)

    def _quality_OK(self):
        log_reader.load_data(self._analysis_log_file_path)
        if log_reader.count_histograms() > 1:
            A = log_reader.display_histograms(draw_plot=False, 
                mark_rejected=True, threshold=0.995, 
                threshold_less_than=True, 
                log_file=self._analysis_log_file_path, manual_value=None, 
                max_value=255, save_path=self._heatMapPath)

            #Here should ask image analysis module to test grayscales

        #Dummy check so far
        return True

    def log_file(self, message, append=True):

        if append:            
            fs = open(self._analysis_log_file_path,'a')
        else:
            fs = open(self._analysis_log_file_path,'w')
        message = str(message)
        fs.write(message+"\n\r")
        fs.close()
        
    def update_gtk(self, widget=None, event=None, data=None):
        self._measurement_label.set_text("Measurment ({0}/{1}):".\
            format(self._iteration, self._iterations_max))
        
    def running_timer(self, widget=None, event=None, data=None):
        if self._scanning:
            gobject.timeout_add(1000*60, self.running_timer)
        elif self._next_scan and self._force_quit == False:
            self._timer.set_text("Next scan in {0} min".\
                format(int(self._next_scan - time.time())/60))

            gobject.timeout_add(1000*60, self.running_timer)

    def do_scan(self):
        if not self._force_quit:

            self._scanning = True

            scanner_address = self.owner.get_scanner_address(self._scanner_name[-1])

            if scanner_address is not None:
                self._looked_for_scanner = 0

                self._timer.set_text("Scanning...")
                scan = self._scan(handle=self._handle, scanner=scanner_address)
                if scan: 
                    if type(scan) == types.TupleType:
                        if scan[0] == "SANE-CALLBACK":
                            self._subprocesses.append(scan)
                            if len(self._subprocesses) == 1:
                                gobject.timeout_add(1000*30, self._callback)
                else:
                    self._scanning = False
                    self._power_manager.off()
                    self._timer.set_text("Unknown error initiating scan - do you have the capability?")
                    self.DMS('Scanning', 'Unknown error initiating scan - do you have the capability?',
                        110, debug_level='warning')

            elif self._looked_for_scanner < 12*4:
                self._looked_for_scanner += 1
                gobject.timeout_add(1000*5, self.do_scan)
            else:
                self._scanning = False
                self._power_manager.off()
                self._looked_for_scanner = 0
                self._timer.set_text('Scanner was never turned on')
                self.DMS('Scanning', 'Scanner was never turned on', 110, debug_level='warning')

    def _write_log(self, file_list=None):
        if file_list:
            gs_data = []

            if type(file_list) != types.ListType:
                file_list = [file_list]

            for f in file_list:
                gs_data.append({'Time':time.time()})
                self.DMS("Analysis", "Grayscale analysis of" + str(f), 
                    level=101, debug_level='debug')

                self.f_settings.image_path = f
                self.f_settings.marker_analysis()
                self.f_settings.set_areas_positions()
                dpi_factor = 4.0
                self.f_settings.A.load_other_size(f, dpi_factor)
                grayscale = self.f_settings.A.get_subsection(\
                    self.f_settings.current_analysis_image_config.get("grayscale_area"))

                gs_data[-1]['mark_X'] = list(self.f_settings.mark_X)
                gs_data[-1]['mark_Y'] = list(self.f_settings.mark_Y)

                if grayscale != None:
                    gs = img_base.Analyse_Grayscale(image=grayscale)
                    gs_data[-1]['grayscale_values'] = gs._grayscale
                    gs_data[-1]['grayscale_indices'] = gs.get_kodak_values()
                else:
                    gs_data[-1]['grayscale_values'] = None
                    gs_data[-1]['grayscale_indices'] = None
            
                sections_areas = self.f_settings.current_analysis_image_config.get_all("plate_%n_area")
                for i, a in enumerate(sections_areas):
                    #s = self.f_settings.A.get_subsection(a)
                    gs_data[-1]['plate_' + str(i) + '_area'] = list(a)

            fs = open(self._analysis_log_file_path,'a')
            log_maker.make_entries(fs, file_list=file_list, extra_info=gs_data,
                verboise=False, quiet=False)
            fs.close()
            self.DMS("Analysis","Done. Nothing more to do for that image...", 
                level=101, debug_level='debug')

    def _callback(self):
        for i, sp in enumerate(self._subprocesses):
            if sp[0] == "SANE-CALLBACK":
                if sp[2].poll() != None:
                    self.DMS("Scanning", "Aqcuired image " + str(sp[1]),
                        level=111, debug_level='debug')
                    sp[1].close

                    got_image = True

                    try:

                        if os.path.getsize(sp[3]) < 1000:
                            got_image = False
                    except:
                            got_image = False

                    if got_image:
                        gobject.timeout_add(1000*25,self._write_log, sp[3])
                    
                    del self._subprocesses[i]

                    if (got_image and self._quality_OK()) or self._force_quit:
                        self._scanning = False
                        gobject.timeout_add(1000*20,self._power_manager.off)
                    else:
                        self.DMS("Scanning", "Quality of scan histogram indicates" + 
                            " that rescan needed! So that I do...",
                            level=110, debug_level='warning')

                        self._scanner.next_file_name =  self._root + os.sep + \
                            self._prefix + os.sep + self._prefix + "_" + \
                             str(self._iteration).zfill(4) + "_rescan.tiff"

                        if os.path.exists(self._scanner.next_file_name) == False:
                            self._power_manager.on()
                            gobject.timeout_add(1000*20, self.do_scan)
                        
        if self._subprocesses:
            gobject.timeout_add(1000*30, self._callback)

    def terminate_experiment(self, widget=None, event=None, data=None):

        dialog = gtk.MessageDialog(self.owner.window, gtk.DIALOG_DESTROY_WITH_PARENT,
           gtk.MESSAGE_WARNING, gtk.BUTTONS_NONE,
           "This will terminate the ongoing experiment on %s and free up \
that scanner.\n\nDo you wish to continiue"  % self._scanner_name)

        dialog.add_button(gtk.STOCK_YES, gtk.RESPONSE_YES)
        dialog.add_button(gtk.STOCK_NO, gtk.RESPONSE_NO)
        resp = dialog.run()
        dialog.destroy()
        if resp == gtk.RESPONSE_YES:
            self._iteration = self._iterations_max + 1
            self._measurement_label.set_text("The program will switch stop aquiring ASAP")
            self._timer.set_text("")
            self._force_quit = True
            self.running_Analysis()          

    def _terminate(self, ask=True):

        
        if self._scanning == True and self._loaded:
            if self._destroy_request is None:
                self._destroy_request = time.time()
                
            if time.time() - self._destroy_request < 5:
                gobject.timeout_add(500, self._terminate)
                return
            
        if self._loaded:
            self._power_manager.off()
            self.owner.set_unclaim_scanner(self._scanner_name)
        self.destroy()

    def running_Experiment(self, widget=None, event=None, data=None):
        if self._force_quit == False:
            self.update_gtk()
            if self._iteration < self._iterations_max:
                gobject.timeout_add(1000*60*int(self._interval_time), self.running_Experiment)
                self._next_scan = (time.time() + 60*self._interval_time)
            else:
                    gobject.timeout_add(1000*60*int(self._interval_time), self.running_Analysis)          
                    self._next_scan = None #(time.time() + 60*self._interval_time)
                    self._measurement_label.set_text("Aquiring last image:")

            self._timer.set_text("Waiting for scanner to come online...")
            self._scanner.next_file_name =  self._root + os.sep + self._prefix + \
                os.sep + self._prefix + "_" + str(self._iteration).zfill(4) + \
                ".tiff"

            self._power_manager.on()
            gobject.timeout_add(1000*10, self.do_scan)

            self._iteration += 1

    def running_Analysis(self):

        if self._scanning == False and self._loaded == True:
            self._power_manager.off() #Security measure, since many ways to get here and
                #won't do anything if already switched off
            self.owner.set_unclaim_scanner(self._scanner_name)
            self._loaded = False
            self._measurement_label.set_text("Scanning done:")
            if self._include_analysis:
                self._timer.set_text("Starting analysis...")
                self.DMS('EXPERIMENT', 'Starting analysis...', level=100, debug_level='debug')
                self._matrices = None
                self.owner.analysis_Start_New(widget = self)
            else:
                self._timer.set_text("No automatic analysis...")
                self.DMS('EXPERIMENT', 'Not starting analysis...', level=100, debug_level='debug')
            gobject.timeout_add(1000*3, self.destroy)        
        elif self._loaded == False:
            self.owner.set_unclaim_scanner(self._scanner_name)
            self.DMS('EXPERIMENT', 'Failed to start...', level=100, debug_level='debug')
            self.destroy()
        else:  
            self.DMS('EXPERIMENT', 'Waiting for scan to finnish...', level=100, debug_level='debug')
            gobject.timeout_add(1000*4, self.running_Analysis)  
          
class Scanning_Experiment_Setup(gtk.Frame):
    def __init__(self, owner, simple_scan = False, p_uuid=None):
        gtk.Frame.__init__(self, "NEW SET-UP EXPERIMENT")

        self.connect("hide", self._hide_function)
        self.connect("show", self._show_function)

        if p_uuid is None:
            p_uuid = uuid.uuid1()

        self.p_uuid = p_uuid

        self._GUI_updating = False
        self._owner = owner
        self.DMS = owner.DMS

        vbox2 = gtk.VBox(False, 0)
        vbox2.show()
        self.add(vbox2)

        hbox = gtk.HBox()
        self._selected_scanner = None
        self.scanner = gtk.combo_box_new_text()
        self.reload_scanner()
        self.scanner.connect("changed", self.set_scanner)
        hbox.pack_start(self.scanner, False, False, 2)
        vbox2.pack_start(hbox)

        label = gtk.Label("Select root directory of experiment:")
        label.show()
        hbox = gtk.HBox()
        hbox.pack_start(label, False, False, 2)
        self.experiment_root = gtk.Label(str(owner._config_file.get("data_root")))
        self.experiment_root.set_max_width_chars(90)
        self.experiment_root.set_ellipsize(pango.ELLIPSIZE_START)
        self.experiment_root.show()
        self.experiment_root.set_selectable(True)
        button = gtk.Button(label = 'New experiment root directory')
        button.connect("clicked", self.select_experiment_root)
        button.show()
        hbox.pack_end(button, False, False, 2)
        hbox.pack_end(self.experiment_root, False, False, 2)
        hbox.show()
        vbox2.pack_start(hbox, False, False, 2)

        label = gtk.Label("Name of project (prefix for images):")
        label.show()
        hbox = gtk.HBox()
        hbox.pack_start(label, False, False, 2)
        self.experiment_name = gtk.Entry()
        self.experiment_name.show()
        hbox.pack_end(self.experiment_name, False, False, 2)
        hbox.show()
        vbox2.pack_start(hbox, False, False, 2)

        label = gtk.Label("Project discription")
        label.show()
        hbox = gtk.HBox()
        hbox.pack_start(label, False, False, 2)
        self.experiment_description = gtk.Entry()
        self.experiment_description.set_width_chars(70)
        self.experiment_description.show()
        hbox.pack_end(self.experiment_description, False, False, 2)
        hbox.show()
        vbox2.pack_start(hbox, False, False, 2)


        self.experiment_settings_entry_order = []

        hbox = gtk.HBox()
        label = gtk.Label("Number of measurements:")
        label.show()
        hbox.pack_start(label, False, False, 2)
        self.experiment_times = gtk.Entry()
        self.experiment_times.connect("focus-out-event",self.experiment_Duration_Calculation)
        self.experiment_times.show()
        hbox.pack_end(self.experiment_times, False, False, 2)
        hbox.show()
        vbox2.pack_start(hbox, False, False, 2)
        
        hbox = gtk.HBox()
        label = gtk.Label("Interval (min) you want between measurements:")
        label.show()
        hbox.pack_start(label, False, False, 2)
        self.experiment_interval = gtk.Entry()
        self.experiment_interval.connect("focus-out-event",self.experiment_Duration_Calculation)
        self.experiment_interval.show()
        hbox.pack_end(self.experiment_interval, False, False, 2)
        hbox.show()
        vbox2.pack_start(hbox, False, False, 2)

        hbox = gtk.HBox()
        label = gtk.Label("Experiment duration:")
        label.show()
        self.experiment_duration = gtk.Entry()
        self.experiment_duration.connect("focus-out-event",self.experiment_Duration_Calculation)
        self.experiment_duration.show()
        hbox.pack_start(label, False, False, 2)
        hbox.pack_end(self.experiment_duration, False, False, 2)
        hbox.show()
        vbox2.pack_start(hbox, False, False, 20)
        
        #scanner_settings = Scanner_Settings(scanner=None, scanners_manager="twain", SM_HWND=0)
        #scanner_settings.show()
        #vbox2.pack_start(scanner_settings, False, False, 2)

        vbox3 = gtk.VBox()

        frame = gtk.Frame('Fixture settings')
        frame.add(vbox3)

        hbox = gtk.HBox()
        vbox3.pack_start(hbox,False,False, 10)

        self.fixture = gtk.combo_box_new_text()
        self.fixture.connect("changed", self.set_fixture)
        hbox.pack_start(self.fixture, False, False, 2)
        

        button = gtk.Button("See plate positions on fixture")
        #Change this...
        button.connect("clicked", self.view_config)
        hbox.pack_start(button, False, False, 2)

        self.plate_pinnings = gtk.HBox()
        vbox3.pack_start(self.plate_pinnings,False, False, 2)
        self.plate_matrices = []

        vbox2.pack_start(frame,False, False, 2)

        self.pinning_matrices = {'8 x 12 (96)':(8,12), 
            '16 x 24 (384)': (16,24), 
            '32 x 48 (1536)': (32,48),
            '64 x 96 (6144)': (64,96),
            '--Empty--': None}
        self.set_fixture()

        hbox = gtk.HBox()

        button = gtk.Button("Start experiment")
        button.connect("clicked", self._start_experiment) 
        hbox.pack_start(button, False, False, 2)
        vbox2.pack_start(hbox, False, False, 2)
        
        self.experiment_name.set_text("Test")
        self.experiment_times.set_text("217")
        self.experiment_interval.set_text("20")
        self.experiment_iteration = 0

        self.experiment_Duration_Calculation()
        vbox2.show_all()

    def view_config(self, widget=None, event=None, data=None):
        if self.fixture.get_active() >= 0:
            self._owner.config_fixture(event='view', data=\
                self.fixture.get_model()[self.fixture.get_active()][0].replace(" ","_"))

    def experiment_started(self):

        self._selected_scanner = None

    def _start_experiment(self, widget=None, event=None, data=None):

        self._GUI_updating = True

        self.hide()

        self.experiment_Duration_Calculation()

        self._GUI_updating = False

        self._owner.experiment_Start_New(widget, event, data)

    def _hide_function(self, widget=None, event=None, data=None):

        if self._GUI_updating == False:
            self.DMS('EXPERIMENT SETUP', 'Aborted setup', level=100, debug_level='debug')
            self._owner.set_unclaim_scanner(self._selected_scanner) 
            self._selected_scanner = None
            self.scanner.set_active(-1)

    def _show_function(self, widget=None, event=None, data=None):


        self.reload_scanner()
         

    def reload_scanner(self, active_text=None):

        scanner_list = self._owner.get_unclaimed_scanners()

        self.DMS('EXPERIMENT SETUP', 'Available scanners %s' % str(scanner_list),
            level=100, debug_level='debug')

        for s in scanner_list:
            
            need_input = True

            for pos in xrange(len(self.scanner.get_model())):

                cur_text = self.scanner.get_model()[pos][0]
                if cur_text == s:
                    need_input = False
                    break

            if need_input:
                self.scanner.append_text(s)

        start_len = len(self.scanner.get_model())

        for i in xrange(start_len):
            pos = start_len - i - 1
            if self.scanner.get_model()[pos][0] not in scanner_list:
                self.scanner.remove_text(pos)


        if active_text != None:

            found_text = False
            start_len = len(self.scanner.get_model())
            for i in xrange(start_len):
                if active_text == self.scanner.get_model()[i][0]:

                    self.scanner.set_active(i)
                    return None

        self.scanner.set_active(-1)

        return None


    def reload_fixtures(self, active_text=None):

        fixture_list = sorted(self._owner.fixture_config.get_all_fixtures())
        self.fixture_positions = {}
        for fixture in fixture_list:
            self.fixture_positions[fixture[0]] = fixture[2]
            need_input = True 

            #Adding that fixture if needed
            for i in xrange(len(self.fixture.get_model())):
                cur_text = self.fixture.get_model()[i][0]
                if cur_text == fixture[0]:
                    need_input = False
                    break

            if need_input:
                self.fixture.append_text(fixture[0])                

        #Cleaining up if fixtures have been removed
        start_len = len(self.fixture.get_model())
        for i in xrange(start_len):
            pos = start_len - i - 1
            cur_text = self.fixture.get_model()[pos][0]
            if not cur_text in self.fixture_positions.keys():
                self.fixture.remove_text(pos)

        #Setting the right entry again
        found_text = False
        start_len = len(self.fixture.get_model())
        for i in xrange(start_len):
            cur_text = self.fixture.get_model()[i][0]
            if cur_text == active_text:
                self.fixture.set_active(i)
                found_text = True
                break
        if not found_text:
            if start_len > 0 and active_text == None:
        
                self.fixture.set_active(0)
            else:
                self.fixture.set_active(-1)
               
    def set_scanner(self, widget=None, event=None, data=None):

        if not self._GUI_updating:
            self._GUI_updating = True

            scanner = self.scanner.get_active()
            if scanner >= 0:

                if self._owner.set_claim_scanner(\
                    self.scanner.get_model()[scanner][0]):

                    scanner_text = self.scanner.get_model()[scanner][0]
                    if self._selected_scanner != None:
                        self._owner.set_unclaim_scanner(self._selected_scanner)
                    self._selected_scanner = scanner_text
                    

                else:
                    if self._selected_scanner != None:
                        self._owner.set_unclaim_scanner(self._selected_scanner)
                    self._selected_scanner = None
                    self.reload_scanner()

            self._GUI_updating = False
             

    def set_fixture(self, widget=None, event=None, data=None):

        if not self._GUI_updating:
            self._GUI_updating = True
            #Get info of what user has selected 
            active = self.fixture.get_active()

            if active < 0:
                active_text = None
            else:
                active_text = self.fixture.get_model()[active][0]


            self.reload_fixtures(active_text = active_text)

            active = self.fixture.get_active()

            if active < 0:
                active_text = None
            else:
                active_text = self.fixture.get_model()[active][0]

            #self.DMS('Fixture change',str(active_text),level = 1000)

            self.plate_matrices = []

            #Empty self.plate_pinnings since fixture may not be what it was:
            for child in self.plate_pinnings.get_children():
                self.plate_pinnings.remove(child)

            if active >= 0:
                slots = self.fixture_positions[active_text]

                
                for pos in xrange(slots):

                    label = gtk.Label('Position %d' % pos)
                    self.plate_pinnings.pack_start(label, False, False, 2)

                    dropbox = gtk.combo_box_new_text()                   
                    for m in self.pinning_matrices.keys():
                        dropbox.append_text(m)
                    dropbox.set_active(0)
                    self.plate_matrices.append(dropbox)
                    self.plate_pinnings.pack_start(dropbox, False, False, 2)
                    
                self.plate_pinnings.show_all()

            self._GUI_updating = False

    def select_experiment_root(self, widget=None, event=None, data=None):
        newroot = gtk.FileChooserDialog(title="Select new experiment root", action=gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER, 
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_APPLY, gtk.RESPONSE_APPLY))

        result = newroot.run()
        
        if result == gtk.RESPONSE_APPLY:
            self.experiment_root.set_text(newroot.get_filename())
            
        newroot.destroy()

    def experiment_Duration_Calculation(self, widget=None, event=None, data=None):
        if self._GUI_updating == False:
            self._GUI_updating = True

            ##Allowing thighter scans may cause problems.
            min_interval = 7

            ##Veryfying that the input is valid and putting 'standard' settings if not

            #Number of measurments
            try:
                self.experiment_times.set_text(str(int(self.experiment_times.get_text())))
            except:
                self.experiment_times.set_text("217")

            #Interval time
            try:
                self.experiment_interval.set_text(str(float(self.experiment_interval.get_text())))
            except:
                self.experiment_interval.set_text("20")
            if float(self.experiment_interval.get_text()) < min_interval:
                self.experiment_interval.set_text(str(min_interval))

            #Duration time
            duration_str = self.experiment_duration.get_text()
            duration_array = duration_str.split(",")
            days = 0
            hours = 0
            minutes = 0
            for setting in duration_array:
                s_array = setting.split(" ")
                for i, v in enumerate(s_array):
                    if v == '':
                        del s_array[i]
                if len(s_array) == 1:
                    if len(s_array[0]) > 0:
                        match = re.search('[^0-9.]{1,10}',s_array[0])
                        unit = match.string[match.start(match.group(0)):match.end(match.group(0))]
                        value = match.string[:match.start(match.group(0))] 
                        if unit[0].upper() == 'H':
                            try:
                                hours = int(value)
                            except:
                                pass
                        elif unit[0].upper() == "D":
                            try:
                                days = int(days)
                            except:
                                pass
                        elif unit[0].upper() == "M":
                            try:
                                minutes = int(minutes)
                            except:
                                pass
                            
                elif len(s_array) == 0:
                    pass
                else: 
                    if str(s_array[1])[0].upper() == 'D':
                        try:
                            days = int(s_array[0])
                        except:
                            pass
                    elif str(s_array[1])[0].upper() == 'H':
                        try:
                            hours = int(s_array[0])
                        except:
                            pass
                    elif str(s_array[1])[0].upper() == 'M':
                        try:
                            minutes = int(s_array[0])
                        except:
                            pass

            ##Compensating for if user doesn't know how many minutes to the hour etc.
            ##Also calculating the full runtime
            hours += minutes / 60
            minutes = minutes % 60
            days += hours / 24
            hours =  hours % 24
            run_time = days * 24 * 60 + hours * 60 + minutes
            if run_time < min_interval:
                run_time = min_interval
                minutes = min_interval
            self.experiment_duration.set_text(str(days) + " days, " + str(hours) + " h, " + str(minutes) + " min")
 
            ##Checking which order things have been entered and faking last entry if this was the
            ##first.

            self.experiment_settings_entry_order.append(widget)

            if len(self.experiment_settings_entry_order) < 2:
                if self.experiment_interval not in self.experiment_settings_entry_order:
                    self.experiment_settings_entry_order.insert(0, self.experiment_interval)
                elif self.experiment_times not in self.experiment_settgins_entry_order:
                    self.experiment_settings_entry_order.insert(0, self.experiment_times)
                else:
                    self.experiment_settings_entry_order.insert(0, self.experiment_duration)

            ##Calculating the third parameter, whichever it is

            #Calculating duration        
            if self.experiment_duration not in self.experiment_settings_entry_order[-2:]:
                run_time = float(self.experiment_interval.get_text()) * (int(self.experiment_times.get_text()) + 1)
                minutes = int(run_time % 60)
                hours = int(run_time) % (60*24) / 60
                days = int(run_time) / (60*24)
                out_str = str(days) + " days, " + str(hours) + " h, " + str(minutes) + " min"
                self.experiment_duration.set_text(out_str)
            #Calculating measurement times
            elif self.experiment_times not in self.experiment_settings_entry_order[-2:]:
                times = int(run_time / float(self.experiment_interval.get_text()) - 1)
                self.experiment_times.set_text(str(times))
            #Calculating the interval
            else:
                interval = 0
                measurements = int(self.experiment_times.get_text()) + 1
                while interval < min_interval:
                    if interval != 0:
                        measurements -= 1

                    interval = float(run_time/measurements)                    

                self.experiment_times.set_text(str(measurements - 1))
                self.experiment_interval.set_text(str(interval))

            self._GUI_updating = False


class Scanner_Settings(gtk.TreeView):
    def __init__(self, scanner=None, scanners_manager="twain", SM_HWND=0):

        if scanners_manager == "twain":
            self._SM = twain.SourceManager(SM_HWND)
            self._scanners = self._SM.GetSourceList()
            self._settings_list = {'Lightpath': [twain.ICAP_LIGHTPATH,twain.TWTY_UINT16,'options', 'Transmissive', twain.TWLP_TRANSMISSIVE, 'Reflective', twain.TWLP_REFLECTIVE],
                                   'Lighsource': [twain.ICAP_LIGHTSOURCE, twain.TWTY_UINT16,'int', 0],
                                   'Unit': ['options', 'Inches', twain.TWUN_INCHES, 'Centimeters', twain.TWUN_CENTIMETERS, 'Pixels', twain.TWUN_PIXELS],
                                   'Pixeltype': [twain.ICAP_PIXELTYPE, twain.TWTY_UINT16, 'options', 'Gray', twain.TWPT_GRAY, 'Color', twain.TWPT_RGB],
                                   'Capability 32829': [32829, 4, 'int', 1],
                                   'Capability 32805': [32805, 6, 'bool', 0],
                                   'Capability 32793': [32793, 6, 'bool', 0],
                                   'Orientation': [twain.ICAP_ORIENTATION, twain.TWTY_UINT16, 'options', '90 degrees', twain.TWOR_ROT90, '0 degrees', twain.TWOR_ROT0, '180 degrees', twain.TWOR_ROT180, '270 degrees', twain.TWOR_ROT270],
                                   'X-resolution': [twain.ICAP_XRESOLUTION,twain.TWTY_FIX32, 'float', 600],
                                   'Y-resolution': [twain.ICAP_YRESOLUTION,twain.TWTY_FIX32, 'float', 600],
                                   'Bit Depth': [twain.ICAP_BITDEPTH, twain.TWTY_UINT16, 'int', 8],
                                   'Contrast': [twain.ICAP_CONTRAST,twain.TWTY_FIX32,'float', 25.0],
                                   'Brightness': [twain.ICAP_BRIGHTNESS,twain.TWTY_FIX32,'float', 125.0],
                                   'Scnning area (TOP, LEFT, BOTTOM, RIGHT)': [None, None, 'float-list', '(0.0, 0.0, 11.69, 8.27)']}

        else:
            self._scanners = None
            self._settings_ilst = None
            
        self._treestore = gtk.TreeStore(str, str)
        
        for scanner in self._scanners:
            scanner_iter = self._treestore.append(None, [scanner, None])
            for key, item in self._settings_list.items():
                
                self._treestore.append(scanner_iter, [key, item[3]])
                                      
        gtk.TreeView.__init__(self, self._treestore)

        self._columns = []
        self._columns.append(gtk.TreeViewColumn('Setting'))
        self._columns.append(gtk.TreeViewColumn('Value'))
                                      
        for i, c in enumerate(self._columns):
            self.append_column(c)
            cell = gtk.CellRendererText()
            cell.connect("edited", self.verify_input)
            cell.connect("editing-started", self.guide_input)
            self._columns[i].pack_start(cell, True)
            self._columns[i].add_attribute(cell, 'text', i)
            cell.set_property('editable', i)

            
        self.show_all()

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
        pass

