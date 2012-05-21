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
__version__ = "0.992"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

from PIL import Image, ImageWin

import pygtk
pygtk.require('2.0')
import logging
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

#KNOWN ISSUES ETC.
#
# See CHANGELOG and TODO

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




class Application_Window():

    ui = '''<ui>
    <menubar name="MenuBar">
        <menu action="Scanning">
            <menuitem action="New Experiment"/>
            <menuitem action="Acquire One Image"/>
            <menuitem action="Quit"/>
        </menu>
        <menu action="Analysis">
            <menuitem action="Analyse Project"/>
            <menuitem action="Analyse One Image"/>
        </menu>
        <menu action="Settings">
            <menuitem action="Application Settings"/>
            <menuitem action="Installing Scanner"/>
            <menuitem action="Scanner Configurations"/>
            <menuitem action="Configuring Fixtures"/>
        </menu>
    </menubar>
    </ui>'''

    def __init__(self, program_root):

        #The window
        window = self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        window.set_size_request(1450,900)
        window.connect("delete_event", self.close_application)

        self.USE_CALLBACK=True
        self.USER_OS = USER_OS

        self._handle = 0
        if USER_OS.name == "windows":
            self._handle = self.window.window.handle
        window.set_title("Scannomatic v" + __version__)

        self.DMS("Program startup","Loading config",100,debug_level='info')

        #Init of config parameters
        self._program_root = program_root
        self._program_code_root = program_root + os.sep + "src"
        self._program_config_root = self._program_code_root + os.sep + "config"
        self._config_file = conf.Config_File(self._program_config_root + os.sep + "main.config")
        
        #The scanner queue et al
        self._scanner_queue = []
        self._live_scanners = {}
        self._claimed_scanners = []

        ###HACK
        self._installed_scanners = ['Scanner 1', 'Scanner 2']
        ###END HACK
        self.DMS('Scanner Resources', 'Unclaimed at start-up %s' % \
            self.get_unclaimed_scanners(), level=100, debug_level='debug')
        self.DMS('Scanner Resources', 'Scanners that are on: %s' % \
            str(self.update_live_scanners()), level=100, debug_level='debug')

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
                ("Acquire One Image", None, "Acquire One Image", None, None, self.experiment_New_One_Scan),
                ("Quit",    None,   "Quit",   None,  None,   self.close_application),
                ("Analysis",   None,   "Analysis",    None,    None,   None),
                ("Analyse Project", None, "Analyse Project", None, None, self.menu_Project),
                ("Analyse One Image", None, "Analyse One Image", None, None, self.menu_Analysis),
                ("Settings", None, "Settings", None, None,   None),
                ("Application Settings", None, "Application Settings", None, None, self.menu_Settings),
                ("Installing Scanner",    None,   "Installing Scanner",   None,  None,   self.null_thing),
                ("Scanner Configurations",    None,   "Scanner Configurations",   None,  None,   self.null_thing),
                ("Configuring Fixtures",    None,   "Configuring Fixtures",   None,  None,   self.config_fixture)
            ])

        #attach the actiongroup
        ui_manager.insert_action_group(action_group, 0)

        #add a ui description
        ui_manager.add_ui_from_string(self.ui)

        self.DMS("Program startup","Initialising menu",100, debug_level='info')

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
        self.DMS("Program startup","Initialising fixture GUI",100, debug_level='info')
        self.fixture_config = fixture.Fixture_GUI(self)
        self.vbox.pack_start(self.fixture_config, False, False, 2)

        #Analyis GUI
        self.DMS("Program startup","Initialising analysis GUI",100, debug_level='info')
        self.analyse_one = analysis.Analyse_One(self)
        self.vbox.pack_start(self.analyse_one, False, False, 2)

        #Analyse Project GUI
        self.DMS("Program startup","Initialising project analysis GUI",100, debug_level='info')
        self.analyse_project = project.Project_Analysis_Setup(self)
        self.vbox.pack_start(self.analyse_project, False, False, 2)
            
        #Application Settings GUI
        self.DMS("Program startup","Initialising settings GUI",100, debug_level='info')
        self.app_settings = settings.Config_GUI(self, 'main.config')
        self.vbox.pack_start(self.app_settings, False, False, 2)

        #Setup new Experiment
        self.DMS("Program startup","Initialising experiment GUI",100, debug_level='info')
        self.experiment_layout = experiment.Scanning_Experiment_Setup(self)
        self.vbox.pack_start(self.experiment_layout, False, False, 2)

        #LOGO
        image = gtk.Image()
        image.set_from_file('src/images/scan-o-matic.png')
        hbox = gtk.HBox()
        hbox.pack_start(image, True, False, 2)
        hbox.show_all()
        self.vbox.pack_start(hbox, True, False, 2)
    
        #Running epxeriments
        self.running_experiments = gtk.VBox()
        label = gtk.Label("RUNNING EXPERIMENTS:")
        label.show()
        self.running_experiments.pack_start(label, False, False, 2)
        self.vbox.pack_start(self.running_experiments, False, False, 2)
        self.running_experiments.show()                                

 
        #display all
        window.show()

        #Set up the idle timer. To chekc if an image is ready
        if not self.USE_CALLBACK:
            self.idleTimer = gobject.idle_add(self.onIdleTimer)

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
    # APPLICATION SETTINGS FUNCTIONS
    #

    def menu_Settings(self, widget=None, event=None, data=None):
        self.DMS('Settings','Application Settings Activated', debug_level='info')
        self.show_config(self.app_settings)

    #
    # ANALYSIS FUNCTIONS
    #

    def menu_Analysis(self, widget=None, event=None, data=None):
        self.DMS('Analysis','Activated', debug_level='info')
        self.show_config(self.analyse_one)
        #self.analyse_one.f_settings.load()

    #
    #   FIXTURE FUNCTIONS
    #

    def config_fixture(self, widget=None, event=None, data=None):
        self.DMS('Fixture', 'Activated', debug_level='info')
        self.fixture_config.set_mode(widget, event, data)
        self.show_config(self.fixture_config)

    #
    #   NON-IMPLEMENTED
    #

    def null_thing(self, widget=None, event=None, data=None):
        widget.set_sensitive(False)
        self.DMS('In development',
            'Functionality has not been implemented yet',level=1110, 
            debug_level='error')


    #
    #   EXPERIMENT FUNCTIONS
    #
    def make_Experiment(self, widget=None, event=None, data=None):
        self.show_config(self.experiment_layout)

    def experiment_New_One_Scan(self, widget=None, event=None, data=None):

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

        if resp >= 0:

            self.set_claim_scanner(scanners[resp])

            time_stamp = time.strftime("%d_%b_%Y__%H_%M_%S", time.gmtime())
            experiment.Scanning_Experiment(self, self._handle, scanners[resp],          
                         1,
                         0,
                         "Single_Scan_"+time_stamp,
                         "",
                         self.experiment_layout.experiment_root.get_text(),
                         self.running_experiments,
                         native=True)

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
                fixture=self.experiment_layout.fixture.get_active_text().replace(" ","_"))

        else:

            self.DMS('Experiment', "You're trying to start a project on no scanner."+\
                "\n\nIt makes no sense", level = 1100, debug_level='warning')

    def update_live_scanners(self):

        p = subprocess.Popen("sane-find-scanner -v -v |" +
            " sed -n -E 's/^found USB.*(libusb.*$)/\\1/p'", 
            shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

        out, err = p.communicate()

        scanners = map(str, out.split('\n'))


        if len(scanners) == 1 and scanners[0] == '':
            self.DMS('Scanner Resources', 'No scanners on', level=100,
                debug_level='debug')
            return None

        for s in scanners:
            if s not in self._live_scanners.values() and s != '':
                if -1 in self._live_scanners:
                    self.DMS('Scanner Resources',
                        'More than one uncaught scanner, not good at all',
                        level=100, debug_level='warning')
                else:
                    self._live_scanners[-1] = s
            

        for pos,s in self._live_scanners.items():
            if s not in scanners:
                del self._live_scanners[pos]

        self.DMS('Scanner Resources', 'Live scanners %s' % str(self._live_scanners),
            level=100, debug_level='debug')

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
                        level=101, debug_level='warning')
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
                lock_file, level=100, debug_level='debug')
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
                    level=100, debug_level='warning')
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
                level=100, debug_level='debug')
            return True
        except:
            self.DMS('Scanner Resources',
                    "Failed to claim scanner %s" % scanner,
                    level=100, debug_level='error')
            return False

    def set_unclaim_scanner(self, scanner):
        try:
            fs = open(self._program_config_root + os.sep + "_%s.lock" %\
                 scanner.replace(" ","_"),"w")
            fs.write('0')
            fs.close()
            self.DMS('Scanner Resources', 
                "Released scanner %s" % scanner,
                level=100, debug_level='debug')
        except:
            self.DMS('Scanner Resources', 
                "Could not unclaim scanner %s when done with it" % scanner,
                level=100, debug_level='warning')
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
            analysis_output=widget._analysis_output)


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

            self.DMS('Terminating', '', level = 110, debug_level='info')
            self.window.destroy()
            gtk.main_quit()
            return False
        else:
            self.DMS('Keeping alive','', level = 110, debug_level='info')
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
                return True
            else:
                return False
        else:
            return True

    def DMS(self, title, subject, level = -1, debug_level='debug'):
        """
            Display Message System

            This function outputs a message comprised of a title and a potentially
            multi-row subject to one or more places.

            Arguments:

                @title      A one-row string having the title

                @subject    A string containing the subject. Rows are split at \\n

                @level      An int describing where the data should be output
                            It works as a fake binary array with each position
                            being either 0 or 1. Default behaviour is using 
                            application's configurated standard "log_level" 
                            (activated by passing level = -1). 
                            If none has been set level is set to 11

                            Pos -1 (last):      Output to GUI-status area
                            Pos -2:             Write to application's log file
                            Pos -3:             Report using logging
                            Pos -4:             Activate a message box GUI

                            Example: 
                            level = 101 prints and puts in GUI-status area
        """

        if level == -1:
            level = self._config_file.get("log_level")
            if level == None:
                level = 11

        s_arr = subject.split('\n')

        #Last digit says if message should be displayed in application
        if level % 10 == 1:
            self.status_title.set_text(title.upper())
            self.status_description.set_text(subject)

        #2nd last digit says if message should be written to applications log-file:
        if level % 100 / 10 == 1:
            log_file_path = self._config_file.get("log_path")
            if log_file_path == None:
                self._config_file.set("log_path","log" + os.sep + "runtime.log")
                self.DMS('Incomplete config file','New entries were temporarly' +
                    ' added to config settings.\n' +
                    'You should consider saving these.', 
                    level=1000)
                log_file_path = self._config_file.get("log_path")

            log_file_path = self._program_code_root + os.sep + log_file_path
            time_stamp = time.strftime("%d %b %Y %H:%M:%S", time.gmtime())
            no_file = False

            try:
                fs = open(log_file_path, 'a')
            except:
                self.DMS('File path error', 'Failed to open log-file at:\n\n' + log_file_path +
                    '\n\nPlease update your configuration', level = 1100, debug_level='error')
                no_file = True

            if not no_file:
                fs.write("*** " + title.upper() + "\n\r")
                fs.write("* " + time_stamp + "\n\r")
                fs.write("*\n\r")
                for s_row in s_arr:
                    fs.write("* " + str(s_row) +"\n\r")
                fs.write("\n\r")

                fs.close()

        #3rd last digit says if message should just be printed
        if level % 1000 / 100 == 1:
            if debug_level in ['critical', 'error', 'warning', 'info', 'debug']:
                eval("logging." + debug_level)("%s: %s" % (str(title).upper(), str(subject)))

        #4th last digit says if message should be displayed as a pop-up dialog
        if level % 10000 / 1000 == 1:
            dialog = gtk.MessageDialog(self.window, gtk.DIALOG_DESTROY_WITH_PARENT,
                                   gtk.MESSAGE_INFO, gtk.BUTTONS_NONE,
                                   subject)
            dialog.set_title(title)
            dialog.add_button(gtk.STOCK_OK, gtk.RESPONSE_OK)
            resp = dialog.run()
            dialog.destroy()

if __name__ == "__main__":
    #the following two methods should be equal...
    #script_path_root = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1])
    script_path_root = os.path.dirname(os.path.abspath(__file__))
    if os.getcwd() != script_path_root:
        os.chdir(script_path_root)


    from argparse import ArgumentParser

    parser = ArgumentParser(description='This is the main GUI. It requires no arguments to run')

    parser.add_argument("--debug", dest="debug_level", default="warning",
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
