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
import os

#
# INTERNAL DEPENDENCIES
#

import scanomatic.gui.generic.view_generic as view_generic

#
# STATIC GLOBALS
#

from scanomatic.gui.generic.view_generic import \
    PADDING_SMALL, PADDING_MEDIUM, PADDING_LARGE

#
# CLASSES
#


class Analysis_Extract_Top(view_generic.Top):

    def __init__(self, controller, model):

        super(Analysis_Extract_Top, self).__init__(controller, model)

        self.pack_back_button(model['analysis-top-root_button-text'],
                              controller.set_abort, None)

        self._start_button = view_generic.Start_Button(controller, model)
        self.pack_end(self._start_button, False, False, PADDING_LARGE)
        self.set_allow_next(False)

        self.show_all()

    def hide_button(self):

        for child in self.children():
            self.remove(child)

    def set_allow_next(self, val):

        self._start_button.set_sensitive(val)


class Analysis_Extract_Stage(gtk.VBox):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model
        self._specific_model = controller.get_specific_model()
        super(Analysis_Extract_Stage, self).__init__(False,
                                                     spacing=PADDING_SMALL)

        hbox = gtk.HBox(False, spacing=PADDING_SMALL)
        self._path = gtk.Label("")
        hbox.pack_start(self._path, expand=True, fill=False)
        button = gtk.Button(label=model['extract-dialog'])
        button.connect("clicked", self._selectDirectory)
        hbox.pack_start(button, expand=False, fill=False)

        self.pack_start(hbox, expand=False, fill=False, padding=PADDING_MEDIUM)

        hbox = gtk.HBox(False, spacing=PADDING_SMALL)
        hbox.pack_start(gtk.Label(model['extract-tag-label']),
                        expand=False, fill=False)
        self._tag = gtk.Entry()
        self._tag.connect("changed", self._newTag)
        hbox.pack_start(self._tag, expand=True, fill=True)

        self.pack_start(hbox, expand=False, fill=False, padding=PADDING_MEDIUM)

    def error(self, message):

        dialog = gtk.MessageDialog(
            flags=gtk.DIALOG_DESTROY_WITH_PARENT,
            type=gtk.MESSAGE_ERROR,
            buttons=gtk.BUTTONS_OK,
            message_format=message)

        dialog.run()
        dialog.destroy()

    def _newTag(self, widget):

        self._specific_model['tag'] = widget.get_text()
        self._controller.test_allow_start()

    def _selectDirectory(self, widget):

        dialog = gtk.FileChooserDialog(
            title=self._model['extract-dialog-title'],
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                     gtk.STOCK_OPEN, gtk.RESPONSE_OK))

        dialog.set_action(gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER)

        path = (dialog.run() == gtk.RESPONSE_OK and
                dialog.get_filename() or None)

        dialog.destroy()

        if path is not None:
            if (self._controller.check_path(path) and
                    self._tag.get_text() == ""):

                self._path.set_text(path)
                p = os.path.sep.split(path)
                i = None
                try:
                    i = p.index("analysis")
                except ValueError:

                    i = len(p) - 3

                if i >= 0:
                    self._tag.set_text(p[i])
            else:
                self.error(self._model['extract-bad-directory'])
