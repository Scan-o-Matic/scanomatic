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
# STATIC GLOBALS
#

PADDING_LARGE = 10
PADDING_MEDIUM = 4
PADDING_SMALL = 2

#
# FUNCTIONS
#

def select_file(title, multiple_files=False, file_filter=None):

    d = gtk.FileChooserDialog(title=title, 
        action=gtk.FILE_CHOOSER_ACTION_OPEN, 
        buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, 
        gtk.STOCK_APPLY, gtk.RESPONSE_APPLY))

    d.set_select_multiple(multiple_files)

    if file_filter is not None:

        f = gtk.FileFilter()
        f.set_name(file_filter['filter_name'])
        for m, p in file_filter['mime_and_patterns']:
            f.add_mime_type(m)
            f.add_pattern(p)
        d.add_filter(f)

    res = d.run()
    file_list = d.get_filenames()
    d.destroy()

    if res == gtk.RESPONSE_APPLY:

        return file_list

    else:

        return list()

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
        self.set_allow_start(False)

        self.show_all()

    def set_allow_start(self, val):

        self._start_button.set_sensitive(val)


class Analysis_Top_Image_Generic(Analysis_Top):

    def __init__(self, controller, model, specific_model,
            specific_controller, next_text, next_stage_signal):

        super(Analysis_Top_Image_Generic, self).__init__(controller, model)

        self._specific_model = specific_model
        self._specific_controller = specific_controller

        self._pack_root_button()

        self._next_button = Analysis_Top_Next_Button(controller,
            model, specific_model, next_text, next_stage_signal)

        self.pack_end(self._next_button, False, False, PADDING_LARGE)
        self.set_allow_next(False)

        self.show_all()

    def set_allow_next(self, val):

        self._next_button.set_sensitive(val)


class Analysis_Top_Image_Sectioning(Analysis_Top_Image_Generic):

    def __init__(self, controller, model, specific_model, specific_controller):

        next_text = model['analysis-top-image-sectioning-next']
        next_stage_signal = 'plate'

        super(Analysis_Top_Image_Sectioning, self).__init__(controller,
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

        super(Analysis_Stage_Project, self).__init__()

        label = gtk.Label()
        label.set_markup(model['analysis-stage-project-title'])
        self.pack_start(label, False, False, PADDING_LARGE)

        frame = gtk.Frame(model['analysis-stage-project-file'])
        self.pack_start(frame, False, False, PADDING_SMALL)
        hbox = gtk.HBox()
        hbox.set_border_width(PADDING_MEDIUM)
        frame.add(hbox)
        self.log_file = gtk.Entry()
        self.log_file.set_text(model['analysis-project-log_file'])
        self.log_file.set_sensitive(False)
        hbox.pack_start(self.log_file, True, True, PADDING_SMALL)
        button = gtk.Button(
            label=model['analysis-stage-project-log_file_button-text'])
        button.connect("clicked", controller.project.set_log_file)
        hbox.pack_end(button, False, False, PADDING_SMALL)
        frame.show_all()
 
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
        frame.show_all()
 
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
        frame.show_all()

        self.show()

    def _output_focus_out(self, widget, *args, **kwargs):

        self._controller.project.set_output(widget, self, "exit")

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


    def set_pinning(self, pinnings_list, sensitive):

        box = self.pm_box

        children = box.children()

        if len(children) < len(pinnings_list):

           for p in xrange(len(pinnings_list) - len(children)):

                box.pack_start(Analysis_Stage_Project_Pinning(
                    self._controller, self._model, self,
                    len(children) + p + 1))

           children = box.children()

        if len(children) > len(pinnings_list):

            for p in xrange(len(children) - len(pinnings_list)):
                box.remove[children[-1 - p]]

            children = box.children()

        for i, child in enumerate(children):

            child.set_sensitive(sensitive)
            child.set_pinning(pinnings_list[i])


class Analysis_Stage_Project_Pinning(gtk.VBox):

    def __init__(self, controller, model, project_veiw, 
            plate_number, pinning = None):

            self._model = model
            self._controller = controller
            self._project_view = project_veiw

            super(Analysis_Stage_Project_Pinning, self).__init__()

            label = gtk.Label(
                model['analysis-stage-project-plate-label'].format(
                plate_number))


            self.pack_start(label, False, False, PADDING_SMALL)

            if pinning is None:

                pinning = model['analysis-project-pinning-default']

            self.dropbox = gtk.combo_box_new_text()                   

            def_key = 0
            for i, m in enumerate(sorted(model['pinning_matrices'].keys())):

                self.dropbox.append_text(m)

                if pinning in m:
                    def_key = i

            self.dropbox.set_active(def_key)
            
            self.dropbox.connect("changed",
                self.controller.project.set_pinning, project_veiw,
                plate_number)

    def set_sensitive(self, val):

        self.dropbox.set_sensitive(val)


    def set_pinning(self, pinning):

        orig_key = self.dropbox.get_active()
        new_key = -2

        for i, m in enumerate(sorted(model['pinning_matrices'].keys())):

            if pinning in m:
                new_key = i

        if new_key != orig_key:

            self.dropbox.set_active(new_key)
           
class Analysis_Stage_Image_Selection(gtk.VBox):

    def __init__(self, controller, model, specific_model, specific_controller):

        super(Analysis_Stage_Image_Selection, self).__init__(0, False)

        self._controller = controller
        self._specific_controller = specific_controller
        self._model = model
        self._specific_model = specific_model

        label = gtk.Label()
        label.set_markup(model['analysis-stage-image-selection-title'])
        self.pack_start(label, False, False, PADDING_LARGE)
        label.show()

        hbox = gtk.HBox(0, False)
        self.pack_start(hbox, True, True, PADDING_SMALL)
        left_vbox = gtk.VBox(0, False)
        hbox.pack_start(left_vbox, True, True, PADDING_SMALL)
        right_vbox = gtk.VBox(0, False)
        hbox.pack_start(right_vbox, True, True, PADDING_SMALL)

        treemodel = gtk.ListStore(str)
        specific_model['images-list-model'] = treemodel
        treeview = gtk.TreeView(treemodel)
        tv_cell = gtk.CellRendererText()
        tv_column = gtk.TreeViewColumn(
            model['analysis-stage-image-selection-list-column-title'],
            tv_cell, text=0)
        treeview.append_column(tv_column)
        treeview.set_reorderable(True)
        left_vbox.pack_start(treeview, True, True, PADDING_SMALL)
        hbox = gtk.HBox()
        button = gtk.Button(
            label=model['analysis-stage-image-selection-dialog-button'])
        button.connect("clicked", specific_controller.set_new_images, self)
        hbox.pack_start(button, False, False, PADDING_SMALL)
        left_vbox.pack_end(hbox, False, False, PADDING_SMALL)

        check_button = gtk.CheckButton(
            label=model['analysis-stage-image-selection-fixture'])
        check_button.set_active(specific_model['fixture'])
        check_button.connect("clicked", specific_controller.set_images_has_fixture)
        self.pack_start(check_button, False, False, PADDING_LARGE)
        check_button.show()

        frame = gtk.Frame(
            label=model['analysis-stage-image-selection-logging-title'])
        vbox = gtk.VBox(0, False)

        selections = (('analysis-stage-image-selection-compartments',
            'log-compartments-default',
            specific_controller.log_compartments),
            ('analysis-stage-image-selection-measures',
            'log-measures-default',
            specific_controller.log_measures))

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
            selecton = treeview.get_selection()
            selecton.set_mode(gtk.SELECTION_MULTIPLE)
            selecton.connect("changed", callback)
            for i in xrange(len(treemodel)):
                selecton.select_path("{0}".format(i))
            vbox.pack_start(treeview, False, False, PADDING_SMALL)

        frame.add(vbox)
        right_vbox.pack_start(frame, False, True, PADDING_LARGE)

        self.show_all()


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
        treeview = gtk.TreeView(self.treemodel)
        tv_cell = gtk.CellRendererText()
        tv_column = gtk.TreeViewColumn(
            model['analysis-stage-image-norm-manual-measures'],
            tv_cell, text=0)
        treeview.append_column(tv_column)
        treeview.set_reorderable(False)
        
        hbox.pack_start(treeview, False, False, PADDING_SMALL)

        self.show_all()

    def set_image(self):

        self._specific_model['image-array'] = plt.imread(
            self._specific_model['images-list-model']\
            [self._specific_model['image']][0])

        self.figure_ax.imshow(self._specific_model['image-array'],
            cmap=plt.cm.gray_r)

    def add_measure(self, val):

        self.treemodel.append((val,))

    def place_patch_origin(self, pos):

        p = plt_patches.Rectangle(pos, 0, 0, ec='k', fill=False, lw=0.5)
        self.figure_ax.add_patch(p)
        self.image_canvas.draw()
        self.patches.append(p)

    def move_patch_target(self, w, h):

        p = self.patches[-1]
        p.set_width(w)
        p.set_height(h)
        self.image_canvas.draw()

    def remove_patch(self, x, y):

        self.fixure_ax.remove_patch(self.patches[-1])
        self.image_canvas.draw()
        del self.patches[-1]
        

class Analysis_Log_Book(gtk.VBox):

    def __init__(self, controller, model, specific_model,
                                    specific_controller):

        super(Analysis_Log_Book, self).__init__(0, False)

        self._model = model
        self._specific_model = specific_model
        self._controller = controller
        self._specific_controller = specific_controller

        self.label = gtk.Label('analysis-log-title')
        self.pack_start(self.label, False, False, PADDING_SMALL)


class Analysis_Stage_Image_Sectioning(gtk.VBox):

    def __init__(self, controller, model, specific_model, specific_controller):

        super(Analysis_Stage_Image_Sectioning, self).__init__(0, False)

        self._controller = controller
        self._specific_controller = specific_controller
        self._model = model
        self._specific_model = specific_model

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

    def remove_patch(self, x, y):

        self.fixure_ax.remove_patch(self.patches[-1])
        self.image_canvas.draw()
        del self.patches[-1]
"""
"""
