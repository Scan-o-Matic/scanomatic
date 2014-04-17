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

#
# GLOBALS
#

from scanomatic.gui.generic.view_generic import \
    PADDING_SMALL, PADDING_MEDIUM


class Server_Status(gtk.Frame):

    def __init__(self, model, controller):

        self._model = model
        self._controller = controller

        super(Server_Status, self).__init__(model['status-server'])

        vbox = gtk.VBox(False, spacing=PADDING_MEDIUM)
        self.add(vbox)

        #CONNECTION STATUS
        self._statusIcon = gtk.image_new_from_stock(
            gtk.STOCK_EXECUTE,
            gtk.ICON_SIZE_MENU)

        self._statusLabel = gtk.Label(
            model['status-server-checking'])

        hbox = gtk.HBox(False, spacing=PADDING_SMALL)
        hbox.pack_start(self._statusIcon, expand=False, fill=False)
        hbox.pack_start(self._statusLabel, expand=True, fill=True)
        vbox.pack_start(hbox, expand=False, fill=False,
                        padding=PADDING_SMALL)

        #LOCAL/REMOTE INFO
        hbox = gtk.HBox(False, spacing=PADDING_SMALL)
        hbox.pack_start(gtk.Label(model['server-host-label']),
                        expand=False, fill=False)
        self._hostText = gtk.Label()
        hbox.pack_start(self._hostText, expand=False, fill=False)
        vbox.pack_start(hbox, expand=False, fill=False,
                        padding=PADDING_SMALL)

        #CPU & MEMORY
        self._cpuAndMemBox = gtk.HBox(False, spacing=PADDING_SMALL)
        self._statusCpu = gtk.image_new_from_stock(
            gtk.STOCK_DISCONNECT,
            gtk.ICON_SIZE_MENU)
        self._cpuAndMemBox.pack_start(self._statusCpu, expand=False, fill=False)
        self._cpuAndMemBox.pack_start(
            gtk.Label(model['server-cpu-label']), expand=False, fill=False)
        self._cpuAndMemBox.pack_start(gtk.Label(""), expand=True, fill=True)
        self._statusMem = gtk.image_new_from_stock(
            gtk.STOCK_DISCONNECT,
            gtk.ICON_SIZE_MENU)
        self._cpuAndMemBox.pack_start(self._statusMem, expand=False, fill=False)
        self._cpuAndMemBox.pack_start(
            gtk.Label(model['server-mem-label']), expand=False, fill=False)
        vbox.pack_start(self._cpuAndMemBox, expand=False, fill=False,
                        padding=PADDING_SMALL)

        #SEPARATOR
        vbox.pack_start(gtk.HSeparator(), expand=False, fill=False)

        #SCANNERS
        hbox = gtk.HBox(False, spacing=PADDING_SMALL)
        hbox.pack_start(gtk.Label(model['status-scanners']), expand=True,
                        fill=False)
        self._scanners = gtk.Button(label=model['status-unknown-count'])
        self._scanners.set_sensitive(False)
        self._scanners.connect("clicked", self._showScanners)

        hbox.pack_start(self._scanners, expand=False, fill=False)
        vbox.pack_start(hbox, expand=False, fill=False)

        #JOBS
        hbox = gtk.HBox(False, spacing=PADDING_SMALL)
        hbox.pack_start(gtk.Label(model['status-jobs']), expand=True,
                        fill=False)
        self._jobs = gtk.Button(label=model['status-unknown-count'])
        self._jobs.set_sensitive(False)
        self._jobs.connect("clicked", self._showJobs)

        hbox.pack_start(self._jobs, expand=False, fill=False)
        vbox.pack_start(hbox, expand=False, fill=False, padding=PADDING_SMALL)

        #QUEUE
        hbox = gtk.HBox(False, spacing=PADDING_SMALL)
        hbox.pack_start(gtk.Label(model['status-queue']), expand=True,
                        fill=False)
        self._queue = gtk.Button(label=model['status-unknown-count'])
        self._queue.set_sensitive(False)
        self._queue.connect("clicked", self._showQueue)

        hbox.pack_start(self._queue, expand=False, fill=False)
        vbox.pack_start(hbox, expand=False, fill=False, padding=PADDING_SMALL)

        #UPDATE TIMER
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

            for b, k in ((self._statusCpu, 'cpu-ok'),
                         (self._statusMem, 'mem-ok')):

                if m[k]:
                    im = gtk.STOCK_YES
                else:
                    im = gtk.STOCK_NO

                b.set_from_stock(im, gtk.ICON_SIZE_MENU)

            self._cpuAndMemBox.show()

            if m['serverLocal']:
                self._hostText.set_text(m['server-local-label'])
            else:
                host = m['serverHost']
                if host.startswith("http://"):
                    host = host[7:]
                self._hostText.set_text(host)
            self._hostText.show()

        else:

            self._cpuAndMemBox.hide()
            self._hostText.hide()
            if (m['server-offline']):

                self._statusLabel.set_text(
                    m['status-server-offline'])
                self._statusIcon.set_from_stock(
                    gtk.STOCK_DIALOG_WARNING,
                    gtk.ICON_SIZE_MENU)

            elif (m['serverLocal']):

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

        return True

    def warning(self, message, yn=False):

        dialog = gtk.MessageDialog(
            flags=gtk.DIALOG_DESTROY_WITH_PARENT,
            type=gtk.MESSAGE_WARNING,
            buttons=yn and gtk.BUTTONS_YES_NO or gtk.BUTTONS_OK,
            message_format=message)

        val = dialog.run() in (gtk.RESPONSE_YES, gtk.RESPONSE_OK)
        dialog.destroy()
        return val

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
