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

#
# INTERNAL DEPENDENCIES
#

import scanomatic.gui.generic.view_generic as view_generic

#
# STATIC GLOBALS
#

from scanomatic.gui.generic.view_generic import \
    PADDING_SMALL, PADDING_MEDIUM

#
# CLASSES
#


class Analysis_Convert_Top(view_generic.Top):

    def __init__(self, controller, model):

        super(Analysis_Convert_Top, self).__init__(controller, model)

        self.pack_back_button(model['analysis-top-root_button-text'],
                              controller.set_abort, None)

        self.show_all()


class Analysis_Convert_Stage(gtk.VBox):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model
        super(Analysis_Convert_Stage, self).__init__(False,
                                                     spacing=PADDING_SMALL)

        hbox = gtk.HBox(False, spacing=PADDING_SMALL)
        hbox.pack_start(gtk.Label(model['convert-xml-select-label']),
                        expand=True, fill=False)
        button = gtk.Button(label=model['convert-xml-select-button'])
        button.connect("clicked", self._selectXmlDialog)
        hbox.pack_start(button, expand=False, fill=False)

        self.pack_start(hbox, expand=False, fill=False, padding=PADDING_MEDIUM)

        frame = gtk.Frame(model['convert-xml-conversions'])
        self._conversions = gtk.VBox(False, spacing=PADDING_SMALL)
        frame.add(self._conversions)

        self.pack_start(frame, expand=True, fill=True, padding=PADDING_MEDIUM)

        frame = gtk.Frame(model['convert-xml-conversions-done'])
        self._conversionsDone = gtk.VBox(False, spacing=PADDING_SMALL)
        frame.add(self._conversionsDone)

        self.pack_start(frame, expand=True, fill=True, padding=PADDING_MEDIUM)

        gobject.timeout_add(71, self._update)

    def addWorker(self, process, path):

        lMax = 120
        if (len(path) > lMax):
            path = path[:lMax / 2] + "..." + path[-lMax / 2:]
        f = gtk.Frame(label=path)
        f.process = process
        f.progress = gtk.ProgressBar()
        f.add(f.progress)
        f.progress.set_text(self._model['convert-progress'])
        f.show_all()
        self._conversions.pack_start(f, expand=False, fill=False, padding=2)

    def _update(self, *args):

        for progress in self._conversions.get_children():

            if (hasattr(progress, "process")):

                val = progress.process.poll()
                if val is not None:

                    l = gtk.Label(
                        self._model['convert-completed'].format(
                            self._model['convert-completed-status'][val == 0],
                            progress.get_label()))
                    l.set_justify(gtk.JUSTIFY_LEFT)
                    self._conversionsDone.pack_start(
                        l, expand=False, fill=False, padding=2)
                    self._conversionsDone.reorder_child(l, 0)
                    l.show()

                    self._conversions.remove(progress)

                elif (hasattr(progress, "progress")):
                    progress.progress.pulse()

        return True

    def _selectXmlDialog(self, widget):

        dialog = gtk.FileChooserDialog(
            title=self._model['convert-dialog-title'],
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                     gtk.STOCK_OPEN, gtk.RESPONSE_OK))

        dialog.set_action(gtk.FILE_CHOOSER_ACTION_OPEN)

        path = (dialog.run() == gtk.RESPONSE_OK and
                dialog.get_filename() or None)

        dialog.destroy()

        if path is not None:
            self._controller.start(path)
