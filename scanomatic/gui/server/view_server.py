"""The GTK-GUI view"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import pygtk
pygtk.require('2.0')
import gtk
import gobject
#import pango


class Server_Status(gtk.Frame):

    def __init__(self, model, controller):

        self._model = model
        self._controller = controller

        super(Server_Status, self).__init__(model['status-server'])

        vbox = gtk.VBox(False, spacing=4)
        self.add(vbox)

        self._statusIcon = gtk.image_new_from_stock(
            gtk.STOCK_EXECUTE,
            gtk.ICON_SIZE_MENU)

        self._statusLabel = gtk.Label(
            model['status-server-checking'])

        hbox = gtk.HBox(False, spacing=2)
        hbox.pack_start(self._statusIcon, expand=False, fill=False)
        hbox.pack_start(self._statusLabel, expand=True, fill=True)
        vbox.pack_start(hbox)

        vbox.pack_start(gtk.HSeparator(), expand=False, fill=False)

        hbox = gtk.HBox(False, spacing=2)
        hbox.pack_start(gtk.Label(model['status-scanners']), expand=True,
                        fill=False)
        self._scanners = gtk.Button(label=model['status-unknown-count'])
        self._scanners.set_sensitive(False)
        self._scanners.connect("clicked", self._showScanners)

        hbox.pack_start(self._scanners, expand=False, fill=False)
        vbox.pack_start(hbox)

        hbox = gtk.HBox(False, spacing=2)
        hbox.pack_start(gtk.Label(model['status-jobs']), expand=True,
                        fill=False)
        self._jobs = gtk.Button(label=model['status-unknown-count'])
        self._jobs.set_sensitive(False)
        self._jobs.connect("clicked", self._showJobs)

        hbox.pack_start(self._jobs, expand=False, fill=False)
        vbox.pack_start(hbox)

        hbox = gtk.HBox(False, spacing=2)
        hbox.pack_start(gtk.Label(model['status-queue']), expand=True,
                        fill=False)
        self._queue = gtk.Button(label=model['status-unknown-count'])
        self._queue.set_sensitive(False)
        self._queue.connect("clicked", self._showQueue)

        hbox.pack_start(self._queue, expand=False, fill=False)

        vbox.pack_start(hbox)

        gobject.timeout_add(1001, self.update)

    def update(self, *args):

        m = self._model
        self._controller.update()
        if (m['serverOnline']):
            self._statusLabel.set_text(
                m['status-server-running'])
            self._statusIcon.set_from_stock(
                gtk.STOCK_CONNECT,
                gtk.ICON_SIZE_MENU)

            for b, modelKey in ((self._queue, 'queueLength'),
                                (self._jobs, 'jobsLength'),
                                (self._scanners, 'scannersFree')):

                v = m[modelKey]
                if v < 0:
                    b.set_label(m['status-unknown-count'])
                    b.set_sensitive(False)
                else:
                    b.set_label(str(v))
                    b.set_sensitive(True)
        else:

            if (m['serverLocal']):

                if (m['serverLaunchChecking']):

                    self._statusLabel.set_text(
                        m['status-server-launching'])
                    self._statusIcon.set_from_stock(
                        gtk.STOCK_OK,
                        gtk.ICON_SIZE_MENU)

                else:

                    self._statusLabel.set_text(
                        m['status-server-local-error'])
                    self._statusIcon.set_from_stock(
                        gtk.STOCK_DIALOG_ERROR,
                        gtk.ICON_SIZE_MENU)
            else:

                self._statusLabel.set_text(
                    m['satus-server-remote-no-connection'])
                self._statusIcon.set_from_stock(
                    gtk.STOCK_DIALOG_WARNING,
                    gtk.ICON_SIZE_MENU)

        return True

    def error(self, message):

        dialog = gtk.MessageDialog(
            flags=gtk.DIALOG_DESTROY_WITH_PARENT,
            type=gtk.MESSAGE_ERROR,
            buttons=gtk.BUTTONS_OK,
            message_format=message)

        dialog.run()
        dialog.destroy()

    def _showQueue(self, widget):

        self.error(self._model['status-not-implemented-error'])

    def _showJobs(self, widget):

        self.error(self._model['status-not-implemented-error'])

    def _showScanners(self, widget):

        self.error(self._model['status-not-implemented-error'])
