#!/usr/bin/env python
"""
Scannomatic is a high-throughput biological growth phenotype aquisition and
primary analysis program. 

It typically has two starting starting points, either this script that launches
a GTK (+2.0) application, within witch most operations can be performed. The
second way to run it is to run specific types of analysis on already aquired
images using the script 'src/analysis.py'. (Run it with --help for further
instructions.)

"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.996"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

from PIL import Image, ImageWin

import pygtk
pygtk.require('2.0')
import logging, traceback
import gtk, pango
import gobject
import os, os.path, sys
import time
import types
import subprocess 

#
# SCANNOMATIC LIBRARIES
#

import src.resource_os as os_tools
import src.resource_config as conf
import src.gui_experiment as experiment
import src.gui_fixture as fixture
import src.gui_analysis as analysis
import src.gui_settings as settings
import src.gui_project as project
import src.gui_grid as grid
import src.gui_update as gui_update

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
# GLOBALS
#

DEVELOPMENT = True

#
# CLASSES
#

class Application_Window():

    ui = '''<ui>
    <menubar name="MenuBar">
        <menu action="Scanning">
            <menuitem action="New Experiment"/>
            <menuitem action="Get Drop Test Image"/>
            <menuitem action="Get Pigmentation Image"/>
            <menuitem action="Quit"/>
        </menu>
        <menu action="Analysis">
            <menuitem action="Analyse Project"/>
            <menuitem action="Analyse One Image"/>
            <menuitem action="Inspect and Adjust Gridding"/>
        </menu>
        <menu action="Settings">
            <menuitem action="Application Settings"/>
            <menuitem action="Reset Instances Counter"/>
            <menuitem action="Update Program"/>
            <menuitem action="Installing Scanner"/>
            <menuitem action="Scanner Configurations"/>
            <menuitem action="Unclaim Scanner by Force"/>
            <menuitem action="Configuring Fixtures"/>
        </menu>
    </menubar>
    </ui>'''

    def __init__(self, program_root):

        #The window
        window = self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        window.set_size_request(1450,900)
        window.connect("delete_event", self.close_application)

        #Init of config parameters
        self._logger = None
        self._program_root = program_root
        self._program_code_root = program_root + os.sep + "src"
        self._program_config_root = self._program_code_root + os.sep + "config"
        self._config_file = conf.Config_File(self._program_config_root + os.sep + "main.config")
        self._main_lock_file = self._program_config_root + os.sep + "main.lock"

        #Other instances / catch subprocs from other instances
        self.running_experiments = None
        instances_running = self.set_main_lock_file(delta_instances=1)

        if instances_running > 1:

        
            dialog = gtk.MessageDialog(self.window, gtk.DIALOG_DESTROY_WITH_PARENT,
                gtk.MESSAGE_WARNING, gtk.BUTTONS_NONE,
                "There's already {0} instance{1}".format(instances_running-1, 
                ['','s'][instances_running > 2]) + " running!\nI will cause havoc should" + \
                " scans be initiated from more than one!")


            dialog.add_button(gtk.STOCK_STOP, -1)
            dialog.add_button(gtk.STOCK_OK, 1)

            dialog.show_all()

            resp = dialog.run()

            dialog.destroy()

            if resp == -1:
                self.set_main_lock_file(delta_instances = -1)
                sys.exit()

        #Logging
        log_file_path = self._config_file.get("log_path")
        if log_file_path == None:
            self._config_file.set("log_path","log" + os.sep + "runtime_{0}.log")
            if self._config_file.get("log_level") is None:
                self._config_file.set("log_level", "A")
            self.DMS('Incomplete config file','New entries were temporarly' +
                ' added to config settings.\n' +
                'You should consider saving these.', 
                level="LD")
            log_file_path = self._config_file.get("log_path")

        self._log_file_path = self._program_code_root + os.sep + log_file_path.format(instances_running)

        log_formatter = logging.Formatter('\n\n%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S\n')
        hdlr = logging.FileHandler(self._log_file_path, mode='w')
        hdlr.setFormatter(log_formatter)
        self._logger = logging.getLogger('Scan-o-Matic GUI')
        self._logger.addHandler(hdlr)
        sys.excepthook = self._DMS_tracebacks

        #Callbacks
        self.USE_CALLBACK=True
        self.USER_OS = USER_OS

        self._handle = 0
        if USER_OS.name == "windows":
            self._handle = self.window.window.handle
        window.set_title("Scannomatic v" + __version__)

        self.DMS("Program startup","Loading config","L",debug_level='info')

        
        #The scanner queue et al
        self._scanner_queue = []
        self._live_scanners = {}
        self._claimed_scanners = []

        if self._config_file.get("number_of_scanners") is None:
            self._config_file.set("number_of_scanners", 1)
            self._config_file.save()

        self.set_installed_scanners()

        self.DMS('Scanner Resources', 'Unclaimed at start-up %s' % \
            self.get_unclaimed_scanners(), level="L", debug_level='info')
        self.DMS('Scanner Resources', 'Scanners that are on: %s' % \
            str(self.update_live_scanners()), level="L", debug_level='info')

        #This should only happen on first run
        if self._config_file.get("data_root") == None:
            newroot = gtk.FileChooserDialog(title="Setup: Select experiments root", 
                action=gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER,
                buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_APPLY, gtk.RESPONSE_APPLY))

            result = newroot.run()

            if result == gtk.RESPONSE_APPLY:
                self._config_file.set("data_root", newroot.get_filename())

            newroot.destroy()

            self._config_file.save()


        #widget container
        self.vbox = gtk.VBox(False, 0)
        window.add(self.vbox)
        self.vbox.show()

        #ui manager for menu
        ui_manager = gtk.UIManager()

        #add accelerator group
        accel_group = ui_manager.get_accel_group()
        window.add_accel_group(accel_group)

        #add actiongroup
        action_group = gtk.ActionGroup('Simple GTK Actiongroup')

        #create actions
        action_group.add_actions(
            [
                ("Scanning",   None,   "Scanning",    None,    None,   None),
                ("New Experiment",    None,   "New Experiment", None, None, self.make_Experiment),
                ("Get Drop Test Image", None, "Get Drop Test Image", None, 
                    None, self.experiment_New_One_Scan),
                ("Get Pigmentation Image", None, "Get Pigmentation Image", 
                    None, None, self.experiment_New_One_Scan_Color),
                ("Quit",    None,   "Quit",   None,  None,   self.close_application),
                ("Analysis",   None,   "Analysis",    None,    None,   None),
                ("Analyse Project", None, "Analyse Project", None, None, 
                    self.menu_Project),
                ("Analyse One Image", None, "Analyse One Image", None, None, 
                    self.menu_Analysis),
                ("Inspect and Adjust Gridding", None, 
                    "Inspect and Adjust Gridding", None, None, self.menu_Grid),
                ("Settings", None, "Settings", None, None,   None),
                ("Application Settings", None, "Application Settings", None, 
                    None, self.menu_Settings),
                ("Reset Instances Counter", None, "Reset Instances Counter", None,
                    None, self.reset_Instances_Dialog),
                ("Update Program", None, "Update Program", None,
                    None, self.check_updates),
                ("Installing Scanner",    None,   "Installing Scanner",   None,
                    None,   self.null_thing),
                ("Scanner Configurations",    None,   "Scanner Configurations",
                    None,  None,   self.null_thing),
                ("Unclaim Scanner by Force", None, "Unclaim Scanner by Force", 
                    None, None, self.menu_unclaim_scanner_by_force),
                ("Configuring Fixtures",    None,   "Configuring Fixtures",   
                    None,  None,   self.config_fixture)
            ])

        #attach the actiongroup
        ui_manager.insert_action_group(action_group, 0)

        #add a ui description
        ui_manager.add_ui_from_string(self.ui)

        self.DMS("Program startup","Initialising menu","L", debug_level='info')

        #create a menu-bar to hold the menus and add it to our main window
        menubar = ui_manager.get_widget('/MenuBar')
        self.vbox.pack_start(menubar, False, False)
        menubar.show()

        #Status area
        self.status_area = gtk.HBox()
        self.status_area.show()
        self.vbox.pack_end(self.status_area, False, False, 2)
        self.status_title = gtk.Label("")
        self.status_title.show()
        self.status_area.pack_start(self.status_title, False, False, 20)
        self.status_description = gtk.Label("")
        self.status_description.show()
        self.status_area.pack_start(self.status_description, False, False, 2)

        #Fixture config GUI
        self.DMS("Program startup","Initialising fixture GUI","L", debug_level='info')
        self.fixture_config = fixture.Fixture_GUI(self)
        self.vbox.pack_start(self.fixture_config, False, False, 2)

        #Analyis GUI
        self.DMS("Program startup","Initialising analysis GUI","L", debug_level='info')
        self.analyse_one = analysis.Analyse_One(self)
        self.vbox.pack_start(self.analyse_one, False, False, 2)

        #Analyse Project GUI
        self.DMS("Program startup","Initialising project analysis GUI","L", debug_level='info')
        self.analyse_project = project.Project_Analysis_Setup(self)
        self.vbox.pack_start(self.analyse_project, False, False, 2)
            
        #Grid GUI
        self.DMS("Program startup","Initialising reGrid GUI","L", debug_level='info')
        self.grid = grid.Grid(self)
        self.vbox.pack_start(self.grid, False, False, 2)

        #Application Settings GUI
        self.DMS("Program startup","Initialising settings GUI","L", debug_level='info')
        self.app_settings = settings.Config_GUI(self, conf_file=self._config_file)
        self.vbox.pack_start(self.app_settings, False, False, 2)

        #Setup new Experiment
        self.DMS("Program startup","Initialising experiment GUI","L", debug_level='info')
        self.experiment_layout = experiment.Scanning_Experiment_Setup(self)
        self.vbox.pack_start(self.experiment_layout, False, False, 2)

        #LOGO
        self.logo_image = gtk.Image()
        self.logo_image.set_from_file('src/images/scan-o-matic.png')
        hbox = gtk.HBox()
        hbox.pack_start(self.logo_image, True, False, 2)
        hbox.show_all()
        self.vbox.pack_start(hbox, True, False, 2)
    
        #Running epxeriments
        self.running_experiments = gtk.VBox()
        label = gtk.Label("RUNNING PROCESSES:")
        label.show()
        self.running_experiments.pack_start(label, False, False, 2)
        self.vbox.pack_start(self.running_experiments, False, False, 2)
        self.running_experiments.show()                                

 
        #display all
        window.show()

        self.check_updates()

        #Set up the idle timer. To chekc if an image is ready
        if not self.USE_CALLBACK:
            self.idleTimer = gobject.idle_add(self.onIdleTimer)
       
        if DEVELOPMENT:
            self.DMS("Brave one", 
                "I hope you are aware that you are running developmental version of " +\
                "Scan-o-Matic.\n\nIt means it can be highly unstable!\n\n" +\
                "If you only want to track new versions look in settings...", level="D",
                debug_level = "warning")
  

    #
    # CHECK UPDATES
    #

    def check_updates(self, widget=None, event=None, data=None):

        try:
            dev_update = bool( int(self._config_file.get("dev_update", "0")))
        except:
            dev_update = False

        
        try:
            ver_update = bool( int(self._config_file.get("ver_update", "0")))
        except:
            ver_update = False


        if dev_update or ver_update:

            self.running_experiments.pack_end(gui_update.Update_Process(self, dev_update))
            
    #
    # CONFIGS VISIBILITY
    #

    def show_config(self, widget):

        l = len(self.vbox.get_children()) - 2
        for i,w in enumerate(self.vbox.get_children()):
            if 0 < i < l:
                if w == widget:
                    w.show()
                else:
                    w.hide()

    #
    # APPLICATION GRID FUNCTIONS
    #

    def menu_Grid(self, widget=None, event=None, data=None):
        self.DMS('Gridding','Activated', "L", debug_level='info')
        self.show_config(self.grid)

    #
    # APPLICATION SETTINGS FUNCTIONS
    #

    def menu_Settings(self, widget=None, event=None, data=None):
        self.DMS('Settings','Application Settings Activated', "L", debug_level='info')
        self.show_config(self.app_settings)

    #
    # ANALYSIS FUNCTIONS
    #

    def menu_Analysis(self, widget=None, event=None, data=None):
        self.DMS('Analysis','Activated', "L", debug_level='info')
        self.show_config(self.analyse_one)
        #self.analyse_one.f_settings.load()

    #
    #   FIXTURE FUNCTIONS
    #

    def config_fixture(self, widget=None, event=None, data=None):
        self.DMS('Fixture', 'Activated', 'L', debug_level='info')
        self.fixture_config.set_mode(widget, event, data)
        self.show_config(self.fixture_config)

    #
    #   NON-IMPLEMENTED
    #

    def null_thing(self, widget=None, event=None, data=None):
        widget.set_sensitive(False)
        self.DMS('In development',
            'Functionality has not been implemented yet',level="DL", 
            debug_level='error')


    #
    #   EXPERIMENT FUNCTIONS
    #
    def make_Experiment(self, widget=None, event=None, data=None):
        self.show_config(self.experiment_layout)

    def set_installed_scanners(self):

        self._installed_scanners = []
        for scanner in xrange(int(self._config_file.get("number_of_scanners"))):
            self._installed_scanners.append("Scanner {0}".format(scanner+1))    

    def claim_a_scanner_dialog(self):

        scanners = self.get_unclaimed_scanners()

        dialog = gtk.MessageDialog(self.window, gtk.DIALOG_DESTROY_WITH_PARENT,
            gtk.MESSAGE_INFO, gtk.BUTTONS_NONE,
            "Select the scanner that you wish to use:\n" +\
            "(Scanning will start at once)")

        for i, s in enumerate(scanners):
            dialog.add_button(s, i)

        dialog.add_button(gtk.STOCK_CANCEL, -1)

        img = gtk.Image()
        img.set_from_file('./src/images/martin3.png')
        dialog.set_image(img)
        dialog.show_all()

        resp = dialog.run()

        dialog.destroy()

        return resp, scanners

    def experiment_New_One_Scan(self, widget=None, event=None, data=None):

        resp, scanners = self.claim_a_scanner_dialog()

        if resp >= 0:

            self.set_claim_scanner(scanners[resp])

            time_stamp = time.strftime("%d_%b_%Y__%H_%M_%S", time.gmtime())
            experiment.Scanning_Experiment(self, self._handle, scanners[resp],          
                         1,
                         0,
                         "Drop_Test_Scan_"+time_stamp,
                         "",
                         self.experiment_layout.experiment_root.get_text(),
                         self.running_experiments,
                         native=True, include_analysis=False)

    def experiment_New_One_Scan_Color(self, widget=None, event=None, data=None):

        resp, scanners = self.claim_a_scanner_dialog()

        if resp >= 0:

            self.set_claim_scanner(scanners[resp])

            time_stamp = time.strftime("%d_%b_%Y__%H_%M_%S", time.gmtime())
            experiment.Scanning_Experiment(self, self._handle, scanners[resp],          
                         1,
                         0,
                         "Pigment_Scan_"+time_stamp,
                         "",
                         self.experiment_layout.experiment_root.get_text(),
                         self.running_experiments,
                         native=True, include_analysis=False,
                         color = True)

    def experiment_Start_New(self, widget=None, event=None, data=None):


        if self.experiment_layout._selected_scanner != None:

            scanner = self.experiment_layout._selected_scanner
            self.experiment_layout.experiment_started()

            pinning_matrices = []
            for matrix in self.experiment_layout.plate_matrices:
                if matrix.get_active() >= 0:
                    pinning_matrices.append(self.experiment_layout.\
                        pinning_matrices[matrix.get_active_text()])#get_model()\
                        #[matrix.get_active()][0]])
                else:
                    pinning_matrices.append(None)

            experiment.Scanning_Experiment(self, self._handle,
                scanner,
                self.experiment_layout.experiment_interval.get_text(),
                self.experiment_layout.experiment_times.get_text(),
                self.experiment_layout.experiment_name.get_text(),
                self.experiment_layout.experiment_description.get_text(),
                self.experiment_layout.experiment_root.get_text(),
                self.running_experiments,
                native=True, matrices = pinning_matrices,
                fixture=self.experiment_layout.fixture.get_active_text().replace(" ","_"),
                p_uuid = self.experiment_layout.p_uuid)

        else:

            self.DMS('Experiment', "You're trying to start a project on no scanner."+\
                "\n\nIt makes no sense", level = "DL", debug_level='warning')

    def update_live_scanners(self):

        p = subprocess.Popen("sane-find-scanner -v -v |" +
            " sed -n -E 's/^found USB.*(libusb.*$)/\\1/p'", 
            shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

        out, err = p.communicate()

        scanners = map(str, out.split('\n'))


        if len(scanners) == 1 and scanners[0] == '':
            self.DMS('Scanner Resources', 'No scanners on', level="L",
                debug_level='debug')
            return None

        for s in scanners:
            if s not in self._live_scanners.values() and s != '':
                if -1 in self._live_scanners:
                    self.DMS('Scanner Resources',
                        'More than one uncaught scanner, not good at all',
                        level="AL", debug_level='warning')
                else:
                    self._live_scanners[-1] = s
            

        for pos,s in self._live_scanners.items():
            if s not in scanners:
                del self._live_scanners[pos]

        self.DMS('Scanner Resources', 'Live scanners %s' % str(self._live_scanners),
            level="L", debug_level='info')

        return self._live_scanners

    def get_scanner_address(self, experiment):

        self.update_live_scanners()

        if experiment in self._live_scanners:
            return self._live_scanners[experiment] 
        else:
            self._scanner_queue.insert(0, experiment)

            if -1 in self._live_scanners:
                catch_scanner = self._scanner_queue.pop()
                while catch_scanner in self._live_scanners:
                    self.DMS('Scanner Queue',
                        'Trying to add a second scanner to the same project',
                        level="LA", debug_level='warning')
                    if len(self._scanner_queue) > 0:
                        catch_scanner = self._scanner_queue.pop()
                    else:
                        return None

                self._live_scanners[catch_scanner] = self._live_scanners[-1]
                del self._live_scanners[-1]

        if experiment in self._live_scanners:
            return self._live_scanners[experiment]
        else:
            return None

    def get_unclaimed_scanners(self):

        unclaimed_scanners = []
        for scanner in self._installed_scanners:

            lock_file = self._program_config_root + os.sep + "_%s.lock" % \
                scanner.replace(" ","_")

            self.DMS("Scanner Resources", "Looking for lockfile '%s'" % \
                lock_file, level="L", debug_level='debug')
            try:
                fs = open(lock_file, 'r')
                if str(fs.read(1)) == '0':
                    unclaimed_scanners.append(scanner)
                fs.close()
            except:
                unclaimed_scanners.append(scanner)

        return unclaimed_scanners

    def set_claim_scanner(self, scanner):
        lock_file = self._program_config_root + os.sep + "_%s.lock" % \
            scanner.replace(" ","_")
        try:
            fs = open(lock_file, 'r')
            if str(fs.read(1)) == '1':
                self.DMS('Scanner Resources',
                    "Trying to claim taken scanner, should not be possible!",
                    level="L", debug_level='warning')
                fs.close()
                return False
            fs.close()
        except:
            pass

        try:
            fs = open(lock_file,'w')
            fs.write('1')
            fs.close()
            self._claimed_scanners.append(scanner)
            self.DMS('Scanner Resources', 
                "Claimed scanner %s" % scanner,
                level="L", debug_level='debug')
            return True
        except:
            self.DMS('Scanner Resources',
                    "Failed to claim scanner %s" % scanner,
                    level="LA", debug_level='error')
            return False

    def set_unclaim_scanner(self, scanner):
        try:
            fs = open(self._program_config_root + os.sep + "_%s.lock" %\
                 scanner.replace(" ","_"),"w")
            fs.write('0')
            fs.close()
            self.DMS('Scanner Resources', 
                "Released scanner %s" % scanner,
                level="L", debug_level='info')
        except:
            self.DMS('Scanner Resources', 
                "Could not unclaim scanner %s for unkown reasons" % scanner,
                level="LA", debug_level='warning')

    #
    #   FORCE UNCLAIM FUNCTIONS
    #
    def menu_unclaim_scanner_by_force(self, widget=None, event=None, data=None):

        dialog = gtk.MessageDialog(self.window, gtk.DIALOG_DESTROY_WITH_PARENT,
                               gtk.MESSAGE_INFO, gtk.BUTTONS_NONE,
                               "Select the scanner that you wish to free up:")

        unclaimed = self.get_unclaimed_scanners()
        claimed = [s for s in self._installed_scanners if s not in unclaimed]
        for i,s in enumerate(claimed):
            dialog.add_button(s,i)
        dialog.add_button(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL)

        resp = dialog.run()
        dialog.destroy()
        if resp != gtk.RESPONSE_CANCEL:
            
            dialog = gtk.MessageDialog(self.window, gtk.DIALOG_DESTROY_WITH_PARENT,
               gtk.MESSAGE_WARNING, gtk.BUTTONS_NONE,
               "This will free up {0}\n".format(claimed[resp]) + \
                "Thus crash anything running on it.\n" + \
                "Please make sure it really is free.\n" +
                "And make sure no other instance of Scan-o-Matic is using it!")
            
            dialog.add_button("Unclaim it!", gtk.RESPONSE_YES)
            dialog.add_button("Cancel", gtk.RESPONSE_NO)

            resp2 = dialog.run()
            dialog.destroy()
            if resp2 == gtk.RESPONSE_YES:
                self.set_unclaim_scanner(claimed[resp])
                self.DMS("Force Unclaim Scanner","{0} is now free to use".format(\
                    claimed[resp]),level="LA",debug_level="info") 

    def reset_Instances_Dialog(self, widget=None, event=None, data=None):

        dialog = gtk.MessageDialog(self.window, gtk.DIALOG_DESTROY_WITH_PARENT,
               gtk.MESSAGE_INFO, gtk.BUTTONS_NONE,
               "This will cause scan-o-matic to think that this "+\
                "instance is the only one running.\n\n"+\
                "It is not good to lie to Scan-o-Matic,\n"+\
                "so please be sure it is true.\n\nIs this the only instance?")

        dialog.add_button(gtk.STOCK_YES, gtk.RESPONSE_YES)
        dialog.add_button(gtk.STOCK_NO, gtk.RESPONSE_NO)

        resp = dialog.run()
        dialog.destroy()

        if resp == gtk.RESPONSE_YES:
            self.set_main_lock_file(reset_counter=True)
            
            self.DMS("Reset Instance Counter","Instance counter is reset",
                debug_level="info") 


    #
    #   PROJECT ANALYSIS FUNCTIONS
    #
    def menu_Project(self, widget=None, event=None, data=None):
        self.DMS('Project Analysis','Activated', debug_level='info')
        self.show_config(self.analyse_project)
        #self.analyse_one.f_settings.load()

    def analysis_Start_New(self, widget=None, event=None, data=None):
        
        project.Project_Analysis_Running(self, self.running_experiments, 
            widget._analysis_log_file_path, widget.pinning_string, 
            watch_colony = widget._watch_colony, 
            supress_other = widget._supress_other, 
            watch_time = widget._watch_time,
            analysis_output=widget._analysis_output,
            manual_grid = widget._manual_grid)


    #
    # MAIN APPLICATION FUNCTIONS
    #

    def close_application(self, widget=None, event=None, data=None):
        if self.ask_Quit():
            for child in self.running_experiments.get_children():

                try:
                    #not so nice... 
                    child._power_manager.off()

                except:

                    pass

            for scanner in self._claimed_scanners:
                self.set_unclaim_scanner(scanner)

            self.set_main_lock_file(delta_instances = -1)

            self.DMS('Terminating', '', level = "L", debug_level='info')
            self.window.destroy()
            gtk.main_quit()

            return False
        else:
            self.DMS('Keeping alive','', level = "L", debug_level='info')
            return True
         
    def ask_Quit(self):
        if len(self.running_experiments.get_children()) > 1:
            dialog = gtk.MessageDialog(self.window, gtk.DIALOG_DESTROY_WITH_PARENT,
                                   gtk.MESSAGE_WARNING, gtk.BUTTONS_NONE,
                                   "There are still projects running, are you sure you want to quit?")
            dialog.add_button(gtk.STOCK_YES, gtk.RESPONSE_YES)
            dialog.add_button(gtk.STOCK_NO, gtk.RESPONSE_NO)
            resp = dialog.run()
            dialog.destroy()
            if resp == gtk.RESPONSE_YES:

                
                children = self.running_experiments.get_children()
                children[0].set_text("Aborting running processes, if scans are running it may take some time. Be patient.")
                self.DMS("Scan-o-Matic", "Shutting down, but processes need to end nicely first", "LA")

                for child in children[1:]:
                    child._terminate(ask=False)

                return True
            else:
                return False
        else:
            return True

    def auto_reload(self):

        reloaded = False
        prog = self._program_root + os.sep + sys.argv[0].split(os.sep)[-1]
        args = " ".join(sys.argv[1:])
        
        self.set_main_lock_file(delta_instances = -1)

        try:
            os.execl(prog, args)
            reloaded = True
        except OSError:
            pass



        if not reloaded:
            self.DMS("Restart","Failed to restart the application, do it manually instead!", 
                level="D", debug_level="error")
            self.set_main_lock_file(delta_instances = 1)
            return False

        
        return True

    def _DMS_tracebacks(self, excType, excValue, traceback):
        self._logger.critical("Uncaught exception:",
                 exc_info=(excType, excValue, traceback))

        self.DMS("Code Error Encountered", "An error in the code was encountered.\n"+\
            "Please close the program as soon as no experiment is running and send "+\
            "the file '{0}' to martin.zackrisson@gu.se".format(self._log_file_path),
            level="D", debug_level="critical")


    def DMS(self, title, subject, level=None, debug_level='debug'):
        """
            Display Message System

            This function outputs a message comprised of a title and a potentially
            multi-row subject to one or more places.

            Arguments:

                @title      A one-row string having the title

                @subject    A string containing the subject. Rows are split at \\n

                @level      A string specifying where message should appear,
                            if it conains...

                            'A' it will be in the message area of the GUI

                            'L' it will be reported using logging (at set
                            debug_level)

                            'D' a dialog will be produced

                @debug_level    Sets logging level, ('critical', 'error',
                                'warning', 'info', 'debug')

        """

        if level == None or type(level) == types.IntType:
            level = self._config_file.get("log_level")
            if level == None:
                level = "A"
            elif type(level) == types.IntType:
                level = "A"

        s_arr = subject.split('\n')

        #Last digit says if message should be displayed in application
        if "A" in level:
            self.status_title.set_text(title.upper())
            self.status_description.set_text(subject)

        #2nd last digit says if message should be written to applications log-file:
        if "L" in level:
            if self._logger is None:
                self._logger = logging.getLogger("Scan-o-Matic GUI - early bird logging")

            if debug_level in ['critical', 'error', 'warning', 'info', 'debug']:
                eval("self._logger." + debug_level)("{0}: {1}".format(str(title).upper(), subject))

        #4th last digit says if message should be displayed as a pop-up dialog
        if "D" in level:
            gtk_debug_levels = {'critical': gtk.MESSAGE_ERROR, 'error': gtk.MESSAGE_ERROR,
                'warning': gtk.MESSAGE_WARNING, 'info': gtk.MESSAGE_INFO, 'debug':
                gtk.MESSAGE_INFO}
            if debug_level not in gtk_debug_levels.keys():
                debug_level = "info"

            dialog = gtk.MessageDialog(self.window, gtk.DIALOG_DESTROY_WITH_PARENT,
                                   gtk_debug_levels[debug_level], gtk.BUTTONS_NONE,
                                   subject)
            dialog.set_title(title)
            dialog.add_button(gtk.STOCK_OK, gtk.RESPONSE_OK)
            resp = dialog.run()
            dialog.destroy()

    def set_main_lock_file(self, delta_instances = 0, reset_counter=False):
        instances_running = 0
        other_instance = True
        try:
            fs = open(self._main_lock_file, 'r')
        except:
            other_instance = False

        lock_contents = []
        if other_instance:
            lock_contents = fs.read().split("\n")
            try:
                instances_running = int(lock_contents[0])
            except:
                pass
            fs.close()

        if reset_counter:
            delta_instances = 1 - instances_running

        instances_running += delta_instances
        if instances_running < 0:
            instances_running = 0

        new_lock = [str(instances_running)]

        for p in lock_contents[1:]:
            try:
                proc = psutil.Process(int(p))
            except:
                proc = None
            if proc is not None:
                if proc.is_running() and 'python' in p.name:
                    new_lock.append(p)

        if self.running_experiments is not None:
            for c in self.running_experiments.get_children()[1:]:
                pid = str(c.get_pid())
                if pid is not None and pid not in new_lock[1:]:
                    new_lock.append(pid)


        fs = open(self._main_lock_file,'w')
        fs.write('\n'.join(new_lock))
        fs.close() 

        return instances_running

    def get_subprocs_from_lock_file(self):

        try:
            fs = open(self._main_lock_file, 'r')
            lock_contents = fs.read().split("\n")[1:]
            lock_contents = map(int, lock_contents)
            fs.close()
        except:
            lock_contents = []

        return lock_contents 

if __name__ == "__main__":
    #the following two methods should be equal...
    #script_path_root = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1])
    script_path_root = os.path.dirname(os.path.abspath(__file__))
    if os.getcwd() != script_path_root:
        os.chdir(script_path_root)


    from argparse import ArgumentParser

    parser = ArgumentParser(description='This is the main GUI. It requires no arguments to run')

    parser.add_argument("--debug", dest="debug_level", default="info",
        type=str, help="Set debugging level")    

    args = parser.parse_args()

    #DEBUGGING
    LOGGING_LEVELS = {'critical': logging.CRITICAL,

                      'error': logging.ERROR,

                      'warning': logging.WARNING,

                      'info': logging.INFO,

                      'debug': logging.DEBUG}

    if args.debug_level in LOGGING_LEVELS.keys():

        logging_level = LOGGING_LEVELS[args.debug_level]

    else:

        logging_level = LOGGING_LEVELS['warning']

    logging.basicConfig(level=logging_level, format='\n\n%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S\n')

    app = Application_Window(program_root = script_path_root)

    gtk.main()
