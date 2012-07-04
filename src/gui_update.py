#!/usr/bin/env python
"""GTK-GUI for running program updates"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.994"
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
from subprocess import call, Popen

#
# SCANNOMATIC LIBRARIES
#

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


class Update_Process(gtk.Frame):
    def __init__(self, owner, dev_update):


        self._proc = None
        self._owner = owner
        self.DMS = owner.DMS
        self.ver_update = not dev_update
        self._log_path = os.sep.join(self._owner._log_file_path.split(os.sep)[:-1]) + \
            os.sep + "update.log"

        try:
            self._log = open(self._log_path, 'w')
        except:
            self._log = None

        #Make GTK-stuff

        gtk.Frame.__init__(self, "Checking for updates...")

        vbox = gtk.VBox()
            
        self.hbox = gtk.HBox()
        label = gtk.Label("Checking for new {0} releases:".format(\
            ["development","version"][self.ver_update]))
        self.hbox.pack_start(label, False, False, 2)


        self._ok_button = gtk.Button(label="Restart program")
        self._ok_button.connect("clicked", self.reboot)
        self.hbox.pack_end(self._ok_button,False, False, 2)
        vbox.pack_start(self.hbox, False, False, 2)
        self.add(vbox)

        self._status = gtk.Label("Preparing...")
        self.hbox.pack_end(self._status, False, False, 2)

        self.show_all()
        self._ok_button.hide()

        if self._log is None:
            self._status.set_text("Could not create update log - update won't run!")
            gobject.timeout_add(1000*60*1, self.done)

        else:
            gobject.timeout_add(1000*2, self.run_update)


    def get_log_contents(self):

        fs_lines = ""
        try:
            fs = open(self._log_path,'r')

        except:
            fs = None

        if fs is not None:
            fs_lines = fs.read()
            fs.close()

        return fs_lines

    def _callback(self):

        if self._proc.poll() != None:

            self._log.close()
            self._proc = None

            log = self.get_log_contents()
            destroy_time = 5

            if "Already up-to-date" in log:
                self._status.set_text("Program already up-to-date")
            elif "xx" in log:
                self._status.set_text("Local changes makes automatic update impossible")
            elif "fatal" in log:
                self._status.set_text("Unexpected error, maybe internet connection isn't working...")
            elif "Updating" in log:
                self._status.set_text("Updated!")

                self._ok_button.show()
                destroy_time = 30
        
        if self._proc is not None:
            gobject.timeout_add(100, self._callback)
        else:
            gobject.timeout_add(1000*destroy_time, self._terminate)

    def run_update(self):

        if self.ver_update:
            self.DMS("Update", "Version update not implemented yet, sorry!", level="D")
        
            self._status.set_text("Only developmental update works")
            gobject.timeout_add(1000*30, self._terminate)
            return None

        update_query = ["git","pull"]

        try:
            self._proc = Popen(map(str, update_query), 
                stdout=self._log, shell=False)
        except OSError:
            self._status.set_text("Automatic updating requires git installed, can't find it")
            gobject.timeout_add(1000*30, self._terminate)
            return None

        self._status.set_text("Comunicating with repo...")
        gobject.timeout_add(100, self._callback)

    def reboot(self, widget=None, event=None, data=None):

        self._terminate()
        self._owner.auto_reload()

    def _terminate(self, ask=False):

        self.destroy()
