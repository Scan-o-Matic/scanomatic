#!/usr/bin/env python
"""The GTK-GUI view"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.997"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import pygtk
pygtk.require('2.0')
import gtk

from matplotlib import pyplot as plt
from matplotlib import patches as plt_patches
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas

#
# INTERNAL DEPENDENCIES
#

from src.view_generic import *

#
# STATIC GLOBALS
#

"""Gotten from view_generic instead
PADDING_LARGE = 10
PADDING_MEDIUM = 4
PADDING_SMALL = 2
"""
#
# CLASSES
#


class Analysis(gtk.VBox):

    def __init__(self, controller, model, top=None, stage=None):

        super(Analysis, self).__init__()

        self._controller = controller
        self._model = model

        self.set_top(top)
        self.set_stage(stage)

    def _remove_child(self, pos=0):

        children = self.get_children()

        if len(children) - pos > 0:

            self.remove(children[pos])
            
    def get_controller(self):

        return self._controller

    def set_top(self, widget=None):

        if widget is None:

            widget = Analysis_Top_Root(self._controller,
                        self._model)

        self._top = widget
        self._remove_child(pos=0)
        self.pack_start(widget, False, True, PADDING_LARGE)

    def get_top(self):

        return self._top

    def set_stage(self, widget=None):

        if widget is None:

            widget = Analysis_Stage_About(self._controller,
                        self._model)

        self._stage = widget
        self._remove_child(pos=1)
        self.pack_end(widget, True, True, 10)
        widget.show_all()

    def get_stage(self):

        return self._stage

class Analysis_Top(gtk.HBox):

    def __init__(self, controller, model):

        super(Analysis_Top, self).__init__()

        self._controller = controller
        self._model = model

    def _pack_root_button(self):

        self.pack_start(Analysis_Top_Root_Button(
                self._controller, self._model), False, False, PADDING_SMALL)

class Analysis_Top_Root(Analysis_Top):

    def __init__(self, controller, model):

        super(Analysis_Top_Root, self).__init__(controller, model)

        button = gtk.Button()
        button.set_label(model["analysis-top-root-project_button-text"])
        self.pack_start(button, False, False, PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "project")

        button = gtk.Button()
        button.set_label(model["analysis-top-root-tpu_button-text"])
        self.pack_start(button, False, False, PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "transparency")

        button = gtk.Button()
        button.set_label(model["analysis-top-root-color_button-text"])
        self.pack_start(button, False, False, PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "colour")

        self.show_all()


class Analysis_Top_Project(Analysis_Top):

    def __init__(self, controller, model):

        super(Analysis_Top_Project, self).__init__(controller, model)

        self._pack_root_button()

        self._start_button = Analysis_Top_Project_Start_Button(controller, model)
        self.pack_end(self._start_button, False, False, PADDING_LARGE)
        self.set_allow_next(False)

        self.show_all()

    def set_allow_next(self, val):

        self._start_button.set_sensitive(val)


class Analysis_Top_Image_Generic(Analysis_Top):

    def __init__(self, controller, model, specific_model,
            specific_controller, next_text=None,
            next_stage_signal=None):

        super(Analysis_Top_Image_Generic, self).__init__(controller, model)

        self._specific_model = specific_model
        self._specific_controller = specific_controller

        self._pack_root_button()

        if next_text is not None:

            self._next_button = Analysis_Top_Next_Button(controller,
                model, specific_model, next_text, next_stage_signal)

            self.pack_end(self._next_button, False, False, PADDING_LARGE)
            self.set_allow_next(False)

        self.show_all()

    def set_allow_next(self, val):

        self._next_button.set_sensitive(val)


class Analysis_Top_Done(Analysis_Top_Image_Generic):

    def __init__(self, controller, model):

        next_text = model['analysis-top-image-sectioning-next']
        next_stage_signal = 'plate'

        super(Analysis_Top_Done, self).__init__(controller,
            model, None, None)


class Analysis_Top_Image_Sectioning(Analysis_Top_Image_Generic):

    def __init__(self, controller, model, specific_model, specific_controller):

        next_text = model['analysis-top-image-sectioning-next']
        next_stage_signal = 'plate'

        super(Analysis_Top_Image_Sectioning, self).__init__(controller,
            model, specific_model, specific_controller, next_text,
            next_stage_signal)


class Analysis_Top_Auto_Norm_and_Section(Analysis_Top_Image_Generic):

    def __init__(self, controller, model, specific_model, specific_controller):

        next_text = model['analysis-top-auto-norm-and-section-next']
        next_stage_signal = 'plate'

        super(Analysis_Top_Auto_Norm_and_Section, self).__init__(controller,
            model, specific_model, specific_controller, next_text,
            next_stage_signal)


class Analysis_Top_Image_Normalisation(Analysis_Top_Image_Generic):

    def __init__(self, controller, model, specific_model, specific_controller):

        next_text = model['analysis-top-image-normalisation-next']
        next_stage_signal = 'sectioning'

        super(Analysis_Top_Image_Normalisation, self).__init__(controller,
            model, specific_model, specific_controller, next_text,
            next_stage_signal)


class Analysis_Top_Image_Selection(Analysis_Top_Image_Generic):

    def __init__(self, controller, model, specific_model, specific_controller):

        next_text = model['analysis-top-image-selection-next']
        next_stage_signal = 'normalisation'

        super(Analysis_Top_Image_Selection, self).__init__(controller,
            model, specific_model, specific_controller, next_text,
            next_stage_signal)

class Analysis_Top_Image_Plate(Analysis_Top_Image_Generic):

    def __init__(self, controller, model, specific_model, specific_controller):

        if specific_model['plate'] + 1 < len(specific_model['plate-coords']):

            next_text = model['analysis-top-image-plate-next_plate']
            next_stage_signal = 'plate'

        elif specific_model['image'] + 1 < len(specific_model['images-list-model']):

            next_text = model['analysis-top-image-plate-next_image']
            next_stage_signal = 'normalisation'

        else:

            next_text = model['analysis-top-image-plate-next_done']
            next_stage_signal = 'log_book'


        super(Analysis_Top_Image_Plate, self).__init__(controller,
            model, specific_model, specific_controller, next_text,
            next_stage_signal)

class Analysis_Top_Next_Button(gtk.Button):

    def __init__(self, controller, model, specific_model, label_text,
        stage_signal_text):

        self._controller = controller
        self._model = model
        self._specific_model = specific_model

        super(Analysis_Top_Next_Button, self).__init__(
                                stock=gtk.STOCK_GO_FORWARD)

        al = self.get_children()[0]
        hbox = al.get_children()[0]
        im, l = hbox.get_children()

        l.set_text(label_text)
        hbox.remove(im)
        hbox.remove(l)
        hbox.pack_start(l, False, False, PADDING_SMALL)
        hbox.pack_end(im, False, False, PADDING_SMALL)

        self.connect("clicked", controller.set_analysis_stage, 
                            stage_signal_text, specific_model)

class Analysis_Top_Root_Button(gtk.Button):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model

        super(Analysis_Top_Root_Button, self).__init__(
                                stock=gtk.STOCK_GO_BACK)

        al = self.get_children()[0]
        hbox = al.get_children()[0]
        im, l = hbox.get_children()

        l.set_text(model['analysis-top-root_button-text'])

        self.connect("clicked", controller.set_analysis_stage, "about")

class Analysis_Top_Project_Start_Button(gtk.Button):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model

        super(Analysis_Top_Project_Start_Button, self).__init__(
                                stock=gtk.STOCK_EXECUTE)

        al = self.get_children()[0]
        hbox = al.get_children()[0]
        im, l = hbox.get_children()

        l.set_text(model['analysis-top-project-start-text'])

        self.connect("clicked", controller.project.start)


class Analysis_Stage_About(gtk.Label):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model

        super(Analysis_Stage_About, self).__init__()

        self.set_justify(gtk.JUSTIFY_LEFT)
        self.set_markup(model['analysis-stage-about-text'])

        self.show()


class Analysis_Stage_Project(gtk.VBox):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model
        self._specific_model = controller.project.get_specific_model()

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
        button.connect("clicked", controller.project.set_log_file)
        hbox.pack_end(button, False, False, PADDING_SMALL)
        #File - info
        hbox = gtk.HBox(0, False)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)
        vbox_labels = gtk.VBox(0, False)
        vbox_data = gtk.VBox(0, False)
        hbox.pack_start(vbox_labels, False, False, PADDING_SMALL)
        hbox.pack_start(vbox_data, True, True, PADDING_SMALL)
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
        self.output.connect("changed",
            controller.project.set_output, self, "change")
        self.output.connect("focus-out-event",
            self._output_focus_out)
        self.output_warning = gtk.Image()
        controller.project.set_output(self.output, self, "exit")
        hbox.pack_end(self.output_warning, False, False, PADDING_LARGE)

        #PINNING
        frame = gtk.Frame(model['analysis-stage-project-plates'])
        self.pack_start(frame, False, False, PADDING_MEDIUM)
        vbox = gtk.VBox()
        frame.add(vbox)
        self.keep_gridding = gtk.CheckButton(
            label=model['analysis-stage-project-keep_gridding'])
        self.keep_gridding.connect("clicked", 
            controller.project.toggle_set_pinning, self)
        hbox = gtk.HBox()
        hbox.pack_start(self.keep_gridding, False, False, PADDING_SMALL)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)
        self.pm_box = gtk.HBox()
        vbox.pack_start(self.pm_box, False, True, PADDING_SMALL)
        self.keep_gridding.clicked()

        self.show_all()

    def _output_focus_out(self, widget, *args, **kwargs):

        self._controller.project.set_output(widget, self, "exit")

    def set_log_file_data(self, file_prefix, file_desc, file_images):

        self.file_prefix.set_text(file_prefix)
        self.file_desc.set_text(file_desc)
        self.file_images.set_text(file_images)

    def correct_output_path(self, new_path):

        self.output.set_text(new_path) 

    def set_output_warning(self, val=False):

        if val == False:

            self.output_warning.set_from_stock(gtk.STOCK_APPLY,
                    gtk.ICON_SIZE_SMALL_TOOLBAR)
            self.output_warning.set_tooltip_text(
                self._model['analysis-stage-project-output_folder-ok'])

        else:

            self.output_warning.set_from_stock(gtk.STOCK_DIALOG_WARNING,
                    gtk.ICON_SIZE_SMALL_TOOLBAR)
            self.output_warning.set_tooltip_text(
                self._model['analysis-stage-project-output_folder-warning'])

    def set_log_file(self):

        self.log_file.set_text(self._specific_model['analysis-project-log_file'])

    def set_pinning(self, pinnings_list, sensitive=None):

        box = self.pm_box

        children = box.children()

        if pinnings_list is not None:

            if len(children) < len(pinnings_list):

               for p in xrange(len(pinnings_list) - len(children)):

                    box.pack_start(Pinning(
                        self._controller, self._model, self,
                        len(children) + p + 1, self._specific_model,
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


class Analysis_Stage_Image_Selection(gtk.VBox):

    def __init__(self, controller, model, specific_model, specific_controller):

        super(Analysis_Stage_Image_Selection, self).__init__(0, False)

        self._controller = controller
        self._specific_controller = specific_controller
        self._model = model
        self._specific_model = specific_model

        #TITLE
        label = gtk.Label()
        label.set_markup(model['analysis-stage-image-selection-title'])
        self.pack_start(label, False, False, PADDING_LARGE)
        label.show()

        #LEFT AND RIGHT SIDE BOXES
        hbox = gtk.HBox(0, False)
        self.pack_start(hbox, True, True, PADDING_SMALL)
        left_vbox = gtk.VBox(0, False)
        hbox.pack_start(left_vbox, True, True, PADDING_SMALL)
        right_vbox = gtk.VBox(0, False)
        hbox.pack_start(right_vbox, True, True, PADDING_SMALL)

        #SELECTED FILE LIST
        treemodel = gtk.ListStore(str)
        specific_model['images-list-model'] = treemodel
        self.treeview = gtk.TreeView(treemodel)
        tv_cell = gtk.CellRendererText()
        tv_column = gtk.TreeViewColumn(
            model['analysis-stage-image-selection-list-column-title'],
            tv_cell, text=0)
        self.treeview.append_column(tv_column)
        self.treeview.set_reorderable(True)
        self.treeview.connect('key_press_event', specific_controller.handle_keypress)
        left_vbox.pack_start(self.treeview, True, True, PADDING_SMALL)
        hbox = gtk.HBox()
        button = gtk.Button(
            label=model['analysis-stage-image-selection-dialog-button'])
        button.connect("clicked", specific_controller.set_new_images, self)
        hbox.pack_start(button, False, False, PADDING_SMALL)
        left_vbox.pack_end(hbox, False, False, PADDING_SMALL)

        #USE FIXTURE FOR CALIBRATION ETC
        check_button = gtk.CheckButton(
            label=model['analysis-stage-image-selection-fixture'])
        check_button.set_active(specific_model['fixture'])
        check_button.connect("clicked", specific_controller.set_images_has_fixture)
        self.pack_start(check_button, False, False, PADDING_LARGE)

        #CONTINUE ON PREVIOUS LOG FILE
        frame = gtk.Frame(
            label=model['analysis-stage-image-selection-continue-log'])
        hbox = gtk.HBox(0, False)
        frame.add(hbox)
        self.previous_log = gtk.Label("")
        hbox.pack_start(self.previous_log, True, True, PADDING_SMALL)
        button = gtk.Button(
            label=model['analysis-stage-image-selection-continue-button'])
        hbox.pack_end(button, False, False, PADDING_SMALL)
        button.connect('clicked', specific_controller.load_previous_log_file, self)
        self.pack_start(frame, False, False, PADDING_MEDIUM)

        #LOGGING INTEREST SELECTION
        frame = gtk.Frame(
            label=model['analysis-stage-image-selection-logging-title'])
        vbox = gtk.VBox(0, False)

        selections = (('analysis-stage-image-selection-compartments',
            'log-compartments-default',
            specific_controller.log_compartments),
            ('analysis-stage-image-selection-measures',
            'log-measures-default',
            specific_controller.log_measures))

        self._interests = {'model': list(), 'selection': list(),
            'handler': list(), 'widget': list()}

        for item_list_title, item_list, callback in selections:

            treemodel = gtk.ListStore(str)
            treeview = gtk.TreeView(treemodel)
            tv_cell = gtk.CellRendererText()
            tv_column = gtk.TreeViewColumn(
                model[item_list_title],
                tv_cell, text=0)
            treeview.append_column(tv_column)
            treeview.set_reorderable(False)
            for s in specific_model[item_list]:
                treemodel.append((s,))
            selection = treeview.get_selection()
            selection.set_mode(gtk.SELECTION_MULTIPLE)
            selection_handler = selection.connect("changed", callback)
            for i in xrange(len(treemodel)):
                selection.select_path("{0}".format(i))
            vbox.pack_start(treeview, False, False, PADDING_SMALL)
            self._interests['model'].append(treemodel)
            self._interests['selection'].append(selection)
            self._interests['handler'].append(selection_handler)
            self._interests['widget'].append(treeview)

        frame.add(vbox)
        right_vbox.pack_start(frame, False, True, PADDING_LARGE)

        self.show_all()

    def set_previous_log_file(self, path):

        self.previous_log.set_text(path)

    def set_lock_selection_of_interests(self, val):

        val = not(val)

        for i in xrange(2):

            self._interests['widget'][i].set_sensitive(val)

    def set_interests_from_model(self):

        interests = self._specific_model['log-interests']

        #Checking both 0: Compartments, 1: Measures
        for i in xrange(2):

            selection = self._interests['selection'][i]
            selection_handler = self._interests['handler'][i]
            treemodel = self._interests['model'][i]

            selection.handler_block(selection_handler)

            for row in xrange(len(treemodel)):

                if treemodel[row][0] in interests[i]:

                    selection.select_path("{0}".format(row))

                else:

                    selection.unselect_path("{0}".format(row))

            selection.handler_unblock(selection_handler)

    def delete_selection(self):

        selection = self.treeview.get_selection()
        model = self._specific_model['images-list-model']

        result = selection.get_selected()

        if result is not None:

            model, iter = result

            if iter is not None:

                model.remove(iter)

class Analysis_Stage_Auto_Norm_and_Section(gtk.VBox):

    def __init__(self, controller, model, specific_model, specific_controller):

        super(Analysis_Stage_Auto_Norm_and_Section, self).__init__(0, False)

        self._controller = controller
        self._specific_controller = specific_controller
        self._model = model
        self._specific_model = specific_model

        self._specific_model['image-array'] = plt.imread(
            self._specific_model['images-list-model']\
            [self._specific_model['image']][0])

        hbox = gtk.HBox(0, False)
        self.pack_start(hbox, False, False, PADDING_LARGE)

        label = gtk.Label()
        label.set_markup(model['analysis-stage-auto-norm-and-section-title'])
        hbox.pack_start(label, False, False, PADDING_LARGE)

        #Previously detected option

        hbox = gtk.HBox(0, False)
        self.pack_start(hbox, False, False, PADDING_SMALL)

        self.cb = gtk.CheckButton(
            label=model['analysis-stage-auto-norm-and-section-file'])
        self._cb_signal = self.cb.connect("clicked",
            self._pre_process_pre_detected)
        hbox.pack_start(self.cb, False, False, PADDING_SMALL)

        #Re-detect option
        frame = gtk.Frame(model['analysis-stage-auto-norm-and-section-fixture'])
        self.pack_start(frame, False, False, PADDING_SMALL)
        vbox = gtk.VBox(0, False)
        frame.add(vbox)

        hbox = gtk.HBox(0, False)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)

        self.fixture = gtk.combo_box_new_text()
        hbox.pack_start(self.fixture, False, False, PADDING_SMALL)
        self._fixture_signal = self.fixture.connect("changed",
                self._pre_process_fixture_signal)
        self.run_button = gtk.Button()
        self.run_button.set_label(model['analysis-stage-auto-norm-and-section-run'])
        self.run_button.connect("clicked", specific_controller.execute_fixture,
            (self, specific_model))
        self.run_button.set_sensitive(False)
        hbox.pack_start(self.run_button, False, False, PADDING_SMALL)


        #Progress bar
        hbox = gtk.HBox(0, False)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)
        self.progress = gtk.ProgressBar()
        hbox.pack_start(self.progress, False, False, PADDING_SMALL)
        
        #Gray-Scale plot frame
        frame = gtk.Frame(
            self._model['analysis-stage-auto-norm-and-section-gs-title'])
        vbox = gtk.VBox(0, False)
        frame.add(vbox)
        self.pack_start(frame, False, False, PADDING_SMALL)
        
        #Gray-Scale plot
        self.figure = plt.Figure(figsize=(300, 400), dpi=150)
        self.figure.add_axes()
        self.figure_ax = self.figure.gca()
        self.figure.subplots_adjust(left=0.10, bottom=0.15)

        self.image_canvas = FigureCanvas(self.figure)

        self.image_canvas.set_size_request(500, 180)
        hbox = gtk.HBox(0, False)
        hbox.pack_start(self.image_canvas, False, False, PADDING_SMALL)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)

        label = gtk.Label(
            model['analysis-stage-auto-norm-and-section-gs-help'])
        hbox = gtk.HBox(0, False)
        hbox.pack_start(label, False, False, PADDING_SMALL)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)
        self.set_fixtures_from_model()
        self.cb.set_active(True)
        self.set_image()

        self.show_all()

    def set_image(self, X=None, Y=None):

        sm = self._specific_model
        self.figure_ax.cla()

        if X is None or Y is None:
            if len(sm['auto-transpose']) > sm['image']:
                Y, X = sm['auto-transpose'][sm['image']].get_source_and_target()

        if X is not None and Y is not None:

            self.figure_ax.plot(X, Y)

        plt.setp(self.figure_ax.get_yticklabels(), 
            fontsize='xx-small')
        plt.setp(self.figure_ax.get_xticklabels(), 
            fontsize='xx-small')
        self.image_canvas.draw()

    def set_detect_lock(self, val=True):

        self.cb.handler_block(self._cb_signal)
        self.cb.set_active(val)
        self.fixture.set_sensitive(val==False)
        self.cb.handler_unblock(self._cb_signal)
        
    def run_lock(self):

        self.run_button.set_sensitive(False)

    def run_release(self):

        self.run_button.set_sensitive(True)

    def set_progress(self, value):

        f = self.progress.get_fraction()

        if 1.0 > self.progress.get_fraction() > 0.97:

            self.progress.pulse()

        else:

            self.progress.set_fraction(float(value))

    def set_fixtures_from_model(self, keep_name=True):

        self.fixture.handler_block(self._fixture_signal)

        active_name = None

        if keep_name and len(self.fixture.get_model()) > 0:

            active_name = self.fixture.get_model()[
                        self.fixture.get_active()][0]

        while len(self.fixture.get_model()) > 0:

            self.fixture.remove_text(0)

        for f_name in self._model['fixtures']:

            self.fixture.append_text(f_name.replace("_"," ").capitalize())

        if keep_name:

            if active_name is None:

                self.fixture.set_active(-1)

            else:

                model = self.fixture.get_model()

                for row in xrange(len(model)):

                    if model[row][0] == active_name:

                        self.fixture.set_active(row)
                        break

        self.fixture.handler_unblock(self._fixture_signal)
        
    def _pre_process_fixture_signal(self, widget, *args, **kwargs):

        active = self.fixture.get_active()

        self.run_button.set_sensitive(active >= 0)

        if active < 0:
            active_text = None
        else:
            active_text = self.fixture.get_model()[active][0]
            active_text = active_text.lower().replace(" ", "_")

        self._specific_controller.set_fixture(self, active_text, self._specific_model)

    def _pre_process_pre_detected(self, widget, *args, **kwargs):

        val = widget.get_active()
        self.set_detect_lock(val)

        if val:

            self._specific_controller.get_previously_detected(self, self._specific_model)

        else:

            self._specific_controller.set_no_auto_norm()

class Analysis_Stage_Image_Norm_Manual(gtk.VBox):

    def __init__(self, controller, model, specific_model, specific_controller):

        super(Analysis_Stage_Image_Norm_Manual, self).__init__(0, False)

        self._controller = controller
        self._specific_controller = specific_controller
        self._model = model
        self._specific_model = specific_model

        self.patches = list()

        label = gtk.Label()
        label.set_markup(model['analysis-stage-image-norm-manual-title'])
        self.pack_start(label, False, False, PADDING_LARGE)

        label = gtk.Label(
            specific_model['images-list-model'][specific_model['image']][0])
        self.pack_start(label, False, False, PADDING_SMALL)

        hbox = gtk.HBox(0, False)
        self.pack_start(hbox, True, True, PADDING_SMALL)

        self.figure = plt.Figure(figsize=(300, 400), dpi=150)
        self.figure.add_axes()
        self.figure_ax = self.figure.gca()
        self.set_image()

        self.image_canvas = FigureCanvas(self.figure)
        self.image_canvas.mpl_connect('button_press_event', specific_controller.mouse_button_press)
        self.image_canvas.mpl_connect('button_release_event', specific_controller.mouse_button_release)
        self.image_canvas.mpl_connect('motion_notify_event', specific_controller.mouse_move)

        self.figure_ax.get_xaxis().set_visible(False)
        self.figure_ax.get_yaxis().set_visible(False)

        self.image_canvas.set_size_request(300, 400)
        vbox = gtk.VBox(0, False)
        vbox.pack_start(self.image_canvas, False, False, PADDING_SMALL)
        hbox.pack_start(vbox, False, False, PADDING_SMALL)

        self.treemodel = gtk.ListStore(str)
        self.treeview = gtk.TreeView(self.treemodel)
        self.treeview.connect('key_press_event', specific_controller.handle_keypress)
        tv_cell = gtk.CellRendererText()
        tv_column = gtk.TreeViewColumn(
            model['analysis-stage-image-norm-manual-measures'],
            tv_cell, text=0)
        self.treeview.append_column(tv_column)
        self.treeview.set_reorderable(False)
        
        hbox.pack_start(self.treeview, True, True, PADDING_SMALL)

        self.show_all()

    def set_image(self):

        self._specific_model['image-array'] = plt.imread(
            self._specific_model['images-list-model']\
            [self._specific_model['image']][0])

        im = self._specific_model['image-array']
        if im.max() > 1:
            vmax = 255
        else:
            vmax = 1
        self.figure_ax.imshow(im,
            cmap=plt.cm.gray_r, vmin=0, vmax=vmax)

    def delete_selection(self):

        selection = self.treeview.get_selection()
        model = self.treemodel

        result = selection.get_selected()

        if result is not None:

            model, iter = result

            if iter is not None:

                pos = self._specific_controller.remove_selection(model[iter][0])
                if pos >= 0:
                    model.remove(iter)
                    self.remove_patch(pos)

    def add_measure(self, val):

        self.treemodel.append((val,))

    def place_patch_origin(self, pos):

        p = plt_patches.Rectangle(pos, 0, 0, ec='b', fill=False, lw=2)
        self.figure_ax.add_patch(p)
        self.image_canvas.draw()
        self.patches.append(p)

    def move_patch_target(self, w, h):

        p = self.patches[-1]
        p.set_width(w)
        p.set_height(h)
        self.image_canvas.draw()

    def remove_patch(self, i):

        self.patches[i].remove()
        self.image_canvas.draw()
        del self.patches[i]
        

class Analysis_Stage_Image_Sectioning(gtk.VBox):

    def __init__(self, controller, model, specific_model, specific_controller, window):

        super(Analysis_Stage_Image_Sectioning, self).__init__(0, False)

        self._controller = controller
        self._specific_controller = specific_controller
        self._model = model
        self._specific_model = specific_model
        self._window = window

        self.patches = list()

        label = gtk.Label()
        label.set_markup(model['analysis-stage-image-sectioning-title'])
        self.pack_start(label, False, False, PADDING_LARGE)
        
        label = gtk.Label()
        label.set_markup(model['analysis-stage-image-sectioning-help_text'])
        self.pack_start(label, False, False, PADDING_LARGE)

        self.figure = plt.Figure(figsize=(300, 400), dpi=150)
        self.figure.add_axes()
        self.figure_ax = self.figure.gca()
        self.figure_ax.imshow(self._specific_model['image-array'],
            cmap=plt.cm.gray_r)

        self.image_canvas = FigureCanvas(self.figure)
        self.image_canvas.mpl_connect('key_press_event', specific_controller.handle_mpl_keypress)
        self.image_canvas.mpl_connect('button_press_event', specific_controller.mouse_button_press)
        self.image_canvas.mpl_connect('button_release_event', specific_controller.mouse_button_release)
        self.image_canvas.mpl_connect('motion_notify_event', specific_controller.mouse_move)
        self.figure_ax.get_xaxis().set_visible(False)
        self.figure_ax.get_yaxis().set_visible(False)

        self.image_canvas.set_size_request(300, 400)
        hbox = gtk.VBox(0, False)
        hbox.pack_start(self.image_canvas, False, False, PADDING_SMALL)
        self.pack_start(hbox, False, False, PADDING_SMALL)

        self.show_all()

    def place_patch_origin(self, pos):

        p = plt_patches.Rectangle(pos, 0, 0, ec='b', fill=False, lw=2)
        self.figure_ax.add_patch(p)
        self.image_canvas.draw()
        self.patches.append(p)

    def move_patch_target(self, w, h):

        p = self.patches[-1]
        p.set_width(w)
        p.set_height(h)
        self.image_canvas.draw()

    def set_focus_on_im(self):

        self._window.set_focus(self.image_canvas)

    def remove_patch(self):

        self.patches[-1].remove()
        self.image_canvas.draw()
        del self.patches[-1]


class Analysis_Stage_Image_Plate(gtk.HBox):

    def __init__(self, controller, model, specific_model, specific_controller):

        super(Analysis_Stage_Image_Plate, self).__init__(0, False)

        self._controller = controller
        self._specific_controller = specific_controller
        self._model = model
        self._specific_model = specific_model

        left_vbox = gtk.VBox(0, False)
        self.pack_start(left_vbox, False, True, PADDING_LARGE)

        right_vbox = gtk.VBox(0, False)
        self.pack_start(right_vbox, False, True, PADDING_LARGE)

        self.pack_end(self._specific_controller._log.get_view(),
            True, True, PADDING_LARGE)

        #TITLE
        label = gtk.Label()
        label.set_markup(model['analysis-stage-image-plate-title'].format(
            specific_model['plate'] + 1))
        left_vbox.pack_start(label, False, False, PADDING_LARGE)

        hbox = gtk.HBox(0, False)

        #PLATE DESCIPTION
        label = gtk.Label(model['analysis-stage-image-plate-name'])
        hbox.pack_start(label, False, False, PADDING_SMALL)
        self.plate_description = gtk.Entry()

        plate = specific_controller._log.get_suggested_plate_name(
            specific_model['plate'])

        if plate is not None:
            self.plate_description.set_text(plate)

        self.plate_description.connect("changed",
            specific_controller.set_in_log, "plate")
        hbox.pack_start(self.plate_description, True, True, PADDING_SMALL)
        left_vbox.pack_start(hbox, False, True, PADDING_SMALL)

        self.figure = plt.Figure(figsize=(300, 400), dpi=150)
        self.figure.add_axes()
        self.figure_ax = self.figure.gca()
        self.figure_ax.imshow(self._specific_model['plate-im-array'],
            cmap=plt.cm.gray_r)

        self.image_canvas = FigureCanvas(self.figure)
        self.image_signals = list()
        self.image_signals.append(
                self.image_canvas.mpl_connect('button_press_event',
                specific_controller.mouse_button_press))
        self.image_signals.append(
                self.image_canvas.mpl_connect('button_release_event',
                specific_controller.mouse_button_release))
        self.image_signals.append(
                self.image_canvas.mpl_connect('motion_notify_event',
                specific_controller.mouse_move))

        self.figure_ax.get_xaxis().set_visible(False)
        self.figure_ax.get_yaxis().set_visible(False)

        self.image_canvas.set_size_request(300, 400)
        hbox = gtk.HBox(0, False)
        hbox.pack_start(self.image_canvas, False, False, PADDING_SMALL)
        left_vbox.pack_start(hbox, False, False, PADDING_SMALL)

        #LOCK SIZE
        vbox = gtk.VBox(0, False)
        self.lock_selection = gtk.CheckButton(
            label=model['analysis-stage-image-plate-lock_selection'])
        vbox.pack_start(self.lock_selection, False, False, PADDING_SMALL)
        self.lock_selection.set_active(specific_model['lock-selection'] is None)
        self.lock_selection.connect("clicked", specific_controller.set_selection_lock)

        hbox = gtk.HBox(0, False)
        label = gtk.Label(model['analysis-stage-image-plate-selection-width'])
        hbox.pack_start(label, False, False, PADDING_SMALL)
        self.selection_width = gtk.Entry()
        if specific_model['lock-selection'] is not None:
            self.selection_width.set_text(
                str(specific_model['lock-selection'][0]))
        else:
            self.selection_width.set_text("0")
        self.selection_width.connect("changed", specific_controller.set_cell,
            'width')
        hbox.pack_start(self.selection_width, False, False, PADDING_SMALL)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)

        hbox = gtk.HBox(0, False)
        label = gtk.Label(model['analysis-stage-image-plate-selection-height'])
        hbox.pack_start(label, False, False, PADDING_SMALL)
        self.selection_height = gtk.Entry()
        if specific_model['lock-selection'] is not None:
            self.selection_height.set_text(
                str(specific_model['lock-selection'][1]))
        else:
            self.selection_height.set_text("0")
        self.selection_height.connect("changed", specific_controller.set_cell,
            'height')
        hbox.pack_start(self.selection_height, False, False, PADDING_SMALL)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)

        left_vbox.pack_start(vbox, False, False, PADDING_SMALL)

        #STRAIN
        label = gtk.Label(model['analysis-stage-image-plate-colony-name'])
        right_vbox.pack_start(label, False, False, PADDING_SMALL)

        self.strain_name = gtk.Entry()
        self.strain_name.connect("changed", specific_controller.set_in_log, 'strain')
        right_vbox.pack_start(self.strain_name, False, False, PADDING_SMALL)

        #STRAIN SECTION IM
        self.section_figure = plt.Figure(figsize=(40, 40), dpi=150)
        self.section_figure.add_axes()
        self.section_figure_ax = self.section_figure.gca()

        self.section_image_canvas = FigureCanvas(self.section_figure)
        """
        self.section_image_canvas.mpl_connect('button_press_event',
            specific_controller.mouse_button_press)
        self.section_image_canvas.mpl_connect('button_release_event',
            specific_controller.mouse_button_release)
        self.section_image_canvas.mpl_connect('motion_notify_event',
            specific_controller.mouse_move)
        """
        self.section_figure_ax.get_xaxis().set_visible(False)
        self.section_figure_ax.get_yaxis().set_visible(False)

        hbox = gtk.HBox(0, False)
        hbox.pack_start(self.section_image_canvas, False, False, PADDING_SMALL)
        right_vbox.pack_start(hbox, False, False, PADDING_SMALL)

        #STRAIN SECTION ANALYSIS IM
        self.analysis_figure = plt.Figure(figsize=(40, 40), dpi=150)
        self.analysis_figure.add_axes()
        self.analysis_figure_ax = self.analysis_figure.gca()

        self.analysis_image_canvas = FigureCanvas(self.analysis_figure)
        """
        self.analysis_image_canvas.mpl_connect('button_press_event',
            specific_controller.mouse_button_press)
        self.analysis_image_canvas.mpl_connect('button_release_event',
            specific_controller.mouse_button_release)
        self.analysis_image_canvas.mpl_connect('motion_notify_event',
            specific_controller.mouse_move)
        """
        self.analysis_figure_ax.get_xaxis().set_visible(False)
        self.analysis_figure_ax.get_yaxis().set_visible(False)

        hbox = gtk.HBox(0, False)
        hbox.pack_start(self.analysis_image_canvas, False, False, PADDING_SMALL)
        right_vbox.pack_start(hbox, False, False, PADDING_SMALL)

        self.log_button = gtk.Button(label=model['analysis-stage-image-plate-log-button'])
        self.log_button.connect("clicked", specific_controller.set_in_log, 'measures')
        right_vbox.pack_start(self.log_button, False, False, PADDING_LARGE)
        self.set_allow_logging(False)

        self._selection = None
        if specific_model['lock-selection'] is not None:
            pos = [d/2 for d in specific_model['plate-im-array'].shape][:2]
            pos.reverse()
            self.place_patch_origin(pos, specific_model['lock-selection'])
            self.set_section_image()
        else:
            specific_controller.set_selection(pos=None, wh=None)

        self.show_all()

    def set_image_sensitivity(self, value):


        if value:
            self.image_canvas.flush_events()

        self.image_canvas.set_sensitive(value)

        """
        for signal in self.image_signals:

            if value:
                self.image_canvas.handler_unblock(signal)
            else:
                self.image_canvas.handler_block(signal)
        
        """

    def set_strain(self, value):

        self.strain_name.set_text(value)

    def set_plate(self, value):

        self.plate_description.set_text(value)

    def run_lock_select_check(self):

        self.lock_selection.emit("clicked")

    def set_allow_logging(self, val):

        self.log_button.set_sensitive(val)

    def set_section_image(self):

        if self._specific_model['plate-section-im-array'] is not None and \
                self._specific_model['plate-section-im-array'].size > 0:
 
            self.section_figure_ax.imshow(self._specific_model['plate-section-im-array'],
                cmap=plt.cm.gray_r)
            self.section_image_canvas.set_size_request(150, 150)
            self.section_image_canvas.draw()

    def set_analysis_image(self):

        if self._specific_model['plate-section-grid-cell'] is not None:
            blob = self._specific_model[
                    'plate-section-grid-cell'].get_item("blob").filter_array
 
            if blob is not None and blob.size > 0:
                self.analysis_figure_ax.imshow(blob, cmap=plt.cm.gray_r)
                self.analysis_image_canvas.set_size_request(150, 150)
                self.analysis_image_canvas.draw()

    def place_patch_origin(self, pos, wh):

        w, h = wh
        p = plt_patches.Rectangle(pos, w, h, ec='b', fill=False, lw=2)
        self.figure_ax.add_patch(p)
        self.image_canvas.draw()
        self._selection = p
        self._specific_controller.set_selection(pos=pos)

    def move_patch_origin(self, pos):

        if self._selection is None:

            self.place_patch_origin(pos, (0, 0))

        else:

            self._selection.set_xy(pos)
            self.image_canvas.draw()

    def move_patch_target(self, w, h):

        if self._selection is not None:

            self._selection.set_width(w)
            self._selection.set_height(h)
            self.selection_width.set_text(str(w))
            self.selection_height.set_text(str(h))
            self.image_canvas.draw()

    def remove_patch(self, x, y):

        self.figure_ax.remove_patch(self._selection)
        self.image_canvas.draw()
        self._selection = None

    def get_selection_size(self):

        try:

            wh = (float(self.selection_width.get_text()),
                float(self.selection_height.get_text()))

        except:

            wh = (0, 0)

        return wh

    def set_allow_selection_size_change(self, val):

        self.selection_height.set_sensitive(val)
        self.selection_width.set_sensitive(val)

class Analysis_Stage_Log(gtk.VBox):

    def __init__(self, controller, model, specific_model, parent_model):

        super(Analysis_Stage_Log, self).__init__(0, False)

        self._controller = controller
        self._model = model
        self._specific_model = specific_model

        label = gtk.Label(model['analysis-stage-log-title'])
        self.pack_start(label, False, False, PADDING_MEDIUM)

        self.treemodel = gtk.ListStore(*([str] * (
            len(parent_model['log-meta-features']) +
            len(parent_model['log-interests'][0]) *
            len(parent_model['log-interests'][1]))))

        for row in specific_model['measures']:

            self.add_data_row(row)

        self.treeview = gtk.TreeView(self.treemodel)
        self.treeview.connect('key_press_event', controller.handle_keypress)

        tv_cell = gtk.CellRendererText()

        for col in (0, 3, 5):

            tv_column = gtk.TreeViewColumn(
                parent_model['log-meta-features'][col],
                tv_cell, text=col)

            self.treeview.append_column(tv_column)

        start_col = len(parent_model['log-meta-features'])

        for c_i, compartment in enumerate(parent_model['log-interests'][0]):

            for m_i, measure in enumerate(parent_model['log-interests'][1]):

                tv_column = gtk.TreeViewColumn(
                    "{0}: {1}".format(compartment, measure),
                    tv_cell, text=start_col + (c_i + 1) * m_i)

                tv_column.set_resizable(True)
                tv_column.set_reorderable(True)
                self.treeview.append_column(tv_column)

        scrolled_window = gtk.ScrolledWindow()
        scrolled_window.add_with_viewport(self.treeview)

        self.pack_start(scrolled_window, True, True, PADDING_SMALL)
        
        button = gtk.Button(label=model['analysis-stage-log-save'])
        hbox = gtk.HBox(0, False)
        hbox.pack_end(button, False, False, PADDING_SMALL)
        button.connect("clicked", self._controller.save_data)
        self.pack_end(hbox, False, False, PADDING_SMALL)

        self.show_all()

    def add_data_row(self, measure):

        self.treemodel.append(measure)

    def delete_selection(self):

        selection = self.treeview.get_selection()
        model = self.treemodel

        result = selection.get_selected()

        if result is not None:

            model, iter = result

            if iter is not None:

                pos = self._controller.remove_selection(
                    model[iter][0], model[iter][2], model[iter][4])

                if pos >= 0:
                    model.remove(iter)
