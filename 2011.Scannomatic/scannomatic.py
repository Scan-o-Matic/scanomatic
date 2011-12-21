#!/usr/bin/env python

#
# DEPENDENCIES
#

from PIL import Image, ImageWin

import pygtk
pygtk.require('2.0')

import gtk, pango
import gobject
import os, os.path, sys
import time
import types

#
# SCANNOMATIC LIBRARIES
#

import src.os_tools as os_tools
import src.simple_conf as conf
import src.experiment as experiment
import src.fixture as fixture
import src.analysis as analysis
import src.settings as settings

#KNOWN ISSUES ETC.
#
# See CHANGELOG and TODO

#
# OS DEPENDENT BEHAVIOUR, NOTE THAT WINDOWS HAVE EXTRA DEPENDENCIES!
#

USER_OS = os_tools.OS()

if USER_OS.name == "linux":
    import src.sane as scanner_backend
elif USER_OS.name == "windows":
    import twain
    import src.twain as scanner_backend
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
        window.set_size_request(1200,900)
        window.connect("delete_event", self.close_application)

        self.USE_CALLBACK=True
        self.USER_OS = USER_OS

        self._handle = 0
        if USER_OS.name == "windows":
            self._handle = self.window.window.handle
        window.set_title("Scannomatic v0.97")

        self.DMS("Program startup","Loading config",100)

        #Init of config parameters
        self._program_root = program_root
        self._program_code_root = program_root + os.sep + "src"
        self._program_config_root = self._program_code_root + os.sep + "config"
        self._config_file = conf.Config_File(self._program_config_root + os.sep + "main.config")
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

        self.DMS("Program startup","Initialising menu",100)

        #create a menu-bar to hold the menus and add it to our main window
        menubar = ui_manager.get_widget('/MenuBar')
        self.vbox.pack_start(menubar, False, False)
        menubar.show()

        #Fixture config GUI
        self.DMS("Program startup","Initialising fixture GUI",100)
        self.fixture_config = fixture.Fixture_GUI(self)
        self.vbox.pack_start(self.fixture_config, False, False, 2)

        #Analyis GUI
        self.DMS("Program startup","Initialising analysis GUI",100)
        self.analyse_one = analysis.Analyse_One(self)
        self.vbox.pack_start(self.analyse_one, False, False, 2)

        #Application Settings GUI
        self.DMS("Program startup","Initialising settings GUI",100)
        self.app_settings = settings.Config_GUI(self, 'main.config')
        self.vbox.pack_start(self.app_settings, False, False, 2)

        #Setup new Experiment
        self.DMS("Program startup","Initialising experiment GUI",100)
        self.experiment_layout = experiment.Scanning_Experiment_Setup(self)
        self.vbox.pack_start(self.experiment_layout, False, False, 2)

        #Running epxeriments
        self.running_experiments = gtk.VBox()
        label = gtk.Label("RUNNING EXPERIMENTS:")
        label.show()
        self.running_experiments.pack_start(label, False, False, 2)
        self.vbox.pack_start(self.running_experiments, False, False, 2)
        self.running_experiments.show()                                

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
        self.DMS('Settings','Application Settings Activated')
        self.show_config(self.app_settings)

    #
    # ANALYSIS FUNCTIONS
    #

    def menu_Analysis(self, widget=None, event=None, data=None):
        self.DMS('Analysis','Activated')
        self.show_config(self.analyse_one)
        #self.analyse_one.f_settings.load()

    #
    #   FIXTURE FUNCTIONS
    #

    def config_fixture(self, widget=None, event=None, data=None):
        self.DMS('Fixture', 'Activated')
        self.show_config(self.fixture_config)

    #
    #   NON-IMPLEMENTED
    #

    def null_thing(self, widget=None, event=None, data=None):
        widget.set_sensitive(False)
        self.DMS('Error','Functionality has not been implemented yet',Level=1110)


    #
    #   EXPERIMENT FUNCTIONS
    #
    def make_Experiment(self, widget=None, event=None, data=None):
        self.show_config(self.experiment_layout)

    def experiment_New_One_Scan(self, widget=None, event=None, data=None):
        time_stamp = time.strftime("%d_%b_%Y__%H_%M_%S", time.gmtime())
        experiment.Scanning_Experiment(self, self._handle, "Scanner 1",          
                                         1,
                                         0,
                                         "Single_Scan_"+time_stamp,
                                         "",
                                         self.experiment_layout.experiment_root.get_text(),
                                         self.running_experiments,
                                         native=True)

    def experiment_Start_New(self, widget=None, event=None, data=None):
        self.experiment_layout.experiment_Duration_Calculation()
        self.experiment_layout.hide()
        experiment.Scanning_Experiment(self, self._handle, "Scanner 1",          
                                         self.experiment_layout.experiment_interval.get_text(),
                                         self.experiment_layout.experiment_times.get_text(),
                                         self.experiment_layout.experiment_name.get_text(),
                                         self.experiment_layout.experiment_description.get_text(),
                                         self.experiment_layout.experiment_root.get_text(),
                                         self.running_experiments,
                                         native=True)


    #
    # MAIN APPLICATION FUNCTIONS
    #

    def close_application(self, widget=None, event=None, data=None):
        if self.ask_Quit():
            self.DMS('Terminating', '', level = 110)
            self.window.destroy()
            gtk.main_quit()
            return False
        else:
            self.DMS('Keeping alive','', level = 110)
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

    def DMS(self, title, subject, level = -1):
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
                            Pos -3:             Write to prompt/bash (print)
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
                    '\n\nPlease update your configuration', level = 1000)
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
            print "*** " + title.upper()
            for s_row in s_arr:
                print "* " + s_row

            print ""

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

    app = Application_Window(program_root = script_path_root)

    gtk.main()
