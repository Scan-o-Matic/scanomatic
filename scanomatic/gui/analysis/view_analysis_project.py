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


class Analysis_Top_Project(view_generic.Top):

    def __init__(self, controller, model):

        super(Analysis_Top_Project, self).__init__(controller, model)

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


class Analysis_Stage_Project_Running(gtk.VBox):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model
        self._specific_model = controller.get_specific_model()

        sm = self._specific_model

        super(Analysis_Stage_Project_Running, self).__init__(0, False)

        label = gtk.Label()
        label.set_markup(
            model['analysis-stage-project-running-info'].format(
                sm['analysis-project-log_file']))

        self.pack_start(label, False, False, PADDING_LARGE)

        self.show_all()


class Analysis_Stage_Project(gtk.VBox):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model
        self._specific_model = controller.get_specific_model()

        sm = self._specific_model

        super(Analysis_Stage_Project, self).__init__(0, False)

        #Title
        label = gtk.Label()
        label.set_markup(model['analysis-stage-project-title'])
        self.pack_start(label, False, False, PADDING_LARGE)

        #File - dialog
        frame = gtk.Frame(model['analysis-stage-project-file'])
        self.pack_start(frame, False, False, PADDING_SMALL)
        vbox = gtk.VBox(0, False)
        frame.add(vbox)
        hbox = gtk.HBox(0, False)
        hbox.set_border_width(PADDING_MEDIUM)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)
        self.log_file = gtk.Entry()
        self.log_file.set_text(sm['analysis-project-log_file'])
        self.log_file.set_sensitive(False)
        hbox.pack_start(self.log_file, True, True, PADDING_SMALL)
        button = gtk.Button(
            label=model['analysis-stage-project-log_file_button-text'])
        button.connect("clicked", controller.set_log_file)
        hbox.pack_end(button, False, False, PADDING_SMALL)
        #File - info
        hbox = gtk.HBox(0, False)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)
        vbox_labels = gtk.VBox(0, False)
        vbox_data = gtk.VBox(0, False)
        hbox.pack_start(vbox_labels, False, False, PADDING_SMALL)
        hbox.pack_start(vbox_data, True, True, PADDING_SMALL)
        #File - version valid
        self.version_valid = gtk.Label(
            model['analysis-stage-project-file-invalid'])
        vbox_data.pack_start(self.version_valid, True, True, PADDING_SMALL)
        #File - prefix
        label = gtk.Label(model['analysis-stage-project-file-prefix'])
        label.set_justify(gtk.JUSTIFY_LEFT)
        vbox_labels.pack_start(label, True, True, PADDING_SMALL)
        self.file_prefix = gtk.Label("")
        self.file_prefix.set_justify(gtk.JUSTIFY_LEFT)
        vbox_data.pack_start(self.file_prefix, True, True, PADDING_SMALL)
        #File - description
        label = gtk.Label(model['analysis-stage-project-file-desc'])
        label.set_justify(gtk.JUSTIFY_LEFT)
        vbox_labels.pack_start(label, True, True, PADDING_SMALL)
        self.file_desc = gtk.Label("")
        self.file_desc.set_justify(gtk.JUSTIFY_LEFT)
        vbox_data.pack_start(self.file_desc, True, True, PADDING_SMALL)
        #File - images
        label = gtk.Label(model['analysis-stage-project-file-images'])
        label.set_justify(gtk.JUSTIFY_LEFT)
        vbox_labels.pack_start(label, True, True, PADDING_SMALL)
        self.file_images = gtk.Label("")
        self.file_images.set_justify(gtk.JUSTIFY_LEFT)
        vbox_data.pack_start(self.file_images, True, True, PADDING_SMALL)

        #OUTPUT
        frame = gtk.Frame(model['analysis-stage-project-output_folder'])
        self.pack_start(frame, False, False, PADDING_MEDIUM)
        hbox = gtk.HBox()
        hbox.set_border_width(PADDING_MEDIUM)
        frame.add(hbox)
        self.output = gtk.Entry()
        hbox.pack_start(self.output, True, True, PADDING_MEDIUM)
        self.output.connect(
            "changed",
            controller.set_output, self, "change")
        self.output.connect(
            "focus-out-event",
            self._output_focus_out)
        self.output_warning = gtk.Image()
        controller.set_output(self.output, self, "exit")
        hbox.pack_end(self.output_warning, False, False, PADDING_LARGE)

        #PINNING
        frame = gtk.Frame(model['analysis-stage-project-plates'])
        self.pack_start(frame, False, False, PADDING_MEDIUM)
        vbox = gtk.VBox()
        frame.add(vbox)
        self.keep_gridding = gtk.CheckButton(
            label=model['analysis-stage-project-keep_gridding'])
        self.keep_gridding.connect(
            "clicked",
            controller.toggle_set_pinning, self)
        hbox = gtk.HBox()
        hbox.pack_start(self.keep_gridding, False, False, PADDING_SMALL)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)
        self._pm_box = gtk.HBox()
        vbox.pack_start(self._pm_box, False, True, PADDING_SMALL)
        self.keep_gridding.clicked()

        self.show_all()

    def _output_focus_out(self, widget, *args, **kwargs):

        self._controller.set_output(widget, self, "exit")

    def set_log_file_data(self, file_prefix, file_desc, file_images):

        self.file_prefix.set_text(file_prefix)
        self.file_desc.set_text(file_desc)
        self.file_images.set_text(file_images)

    def correct_output_path(self, new_path):

        self.output.set_text(new_path)

    def set_output_warning(self, val=False):

        if val is False:

            self.output_warning.set_from_stock(
                gtk.STOCK_APPLY,
                gtk.ICON_SIZE_SMALL_TOOLBAR)
            self.output_warning.set_tooltip_text(
                self._model['analysis-stage-project-output_folder-ok'])

        else:

            self.output_warning.set_from_stock(
                gtk.STOCK_DIALOG_WARNING,
                gtk.ICON_SIZE_SMALL_TOOLBAR)
            self.output_warning.set_tooltip_text(
                self._model['analysis-stage-project-output_folder-warning'])

    def set_valid_log_file(self, isValid):

        if isValid:
            self.version_valid.hide()
        else:
            self.version_valid.show()

    def set_log_file(self):

        self.log_file.set_text(self._specific_model['analysis-project-log_file'])

    def set_pinning(self, pinnings_list, sensitive=None):

        box = self._pm_box

        children = box.children()

        if pinnings_list is not None:

            if len(children) < len(pinnings_list):

                for p in xrange(len(pinnings_list) - len(children)):

                    box.pack_start(view_generic.Pinning(
                        self._controller, self._model, self,
                        len(children) + p + 1,
                        pinning=pinnings_list[p]))

                children = box.children()

            if len(children) > len(pinnings_list):

                for p in xrange(len(children) - len(pinnings_list)):
                    box.remove(children[-1 - p])

                children = box.children()

            for i, child in enumerate(children):

                child.set_sensitive(sensitive)
                child.set_pinning(pinnings_list[i])

        box.show_all()
