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

from matplotlib import pyplot as plt
from matplotlib import patches as plt_patches
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas

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


class Analysis_Top_Image_Generic(view_generic.Top):

    def __init__(self, controller, model, specific_model,
                 specific_controller, next_text=None,
                 next_stage_signal=None, back_to_root=True):

        super(Analysis_Top_Image_Generic, self).__init__(controller, model)

        self._specific_model = specific_model
        self._specific_controller = specific_controller

        if back_to_root:
            self.pack_back_button(
                model['analysis-top-root_button-text'],
                controller.set_analysis_stage, "about")

        if next_text is not None:

            self._next_button = view_generic.Top_Next_Button(
                controller, model, specific_model, next_text,
                controller.set_analysis_stage, next_stage_signal)

            self.pack_end(self._next_button, False, False, PADDING_LARGE)
            self.set_allow_next(False)

        self.show_all()

    def set_allow_next(self, val):

        self._next_button.set_sensitive(val)


class Analysis_Top_Done(Analysis_Top_Image_Generic):

    def __init__(self, controller, model):

        """
        next_text = model['analysis-top-image-sectioning-next']
        next_stage_signal = 'plate'
        """

        super(Analysis_Top_Done, self).__init__(
            controller, model, None, None, back_to_root=False)


class Analysis_Top_Image_Sectioning(Analysis_Top_Image_Generic):

    def __init__(self, controller, model, specific_model, specific_controller):

        next_text = model['analysis-top-image-sectioning-next']
        next_stage_signal = 'plate'

        super(Analysis_Top_Image_Sectioning, self).__init__(
            controller, model, specific_model, specific_controller,
            next_text, next_stage_signal, back_to_root=False)


class Analysis_Top_Auto_Norm_and_Section(Analysis_Top_Image_Generic):

    def __init__(self, controller, model, specific_model, specific_controller):

        next_text = model['analysis-top-auto-norm-and-section-next']
        next_stage_signal = 'plate'

        super(Analysis_Top_Auto_Norm_and_Section, self).__init__(
            controller, model, specific_model, specific_controller, next_text,
            next_stage_signal, back_to_root=False)


class Analysis_Top_Image_Normalisation(Analysis_Top_Image_Generic):

    def __init__(self, controller, model, specific_model, specific_controller):

        next_text = model['analysis-top-image-normalisation-next']
        next_stage_signal = 'sectioning'

        super(Analysis_Top_Image_Normalisation, self).__init__(
            controller, model, specific_model, specific_controller, next_text,
            next_stage_signal, back_to_root=False)


class Analysis_Top_Image_Selection(Analysis_Top_Image_Generic):

    def __init__(self, controller, model, specific_model, specific_controller):

        next_text = model['analysis-top-image-selection-next']
        next_stage_signal = 'normalisation'

        super(Analysis_Top_Image_Selection, self).__init__(
            controller, model, specific_model, specific_controller, next_text,
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

        super(Analysis_Top_Image_Plate, self).__init__(
            controller, model, specific_model, specific_controller, next_text,
            next_stage_signal, back_to_root=False)


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
        if specific_model['images-list-model'] is None:
            treemodel = gtk.ListStore(str)
            specific_model['images-list-model'] = treemodel
        else:
            treemodel = specific_model['images-list-model']

        self.treeview = gtk.TreeView(treemodel)
        tv_cell = gtk.CellRendererText()
        tv_column = gtk.TreeViewColumn(
            model['analysis-stage-image-selection-list-column-title'],
            tv_cell, text=0)
        self.treeview.append_column(tv_column)
        self.treeview.set_reorderable(True)
        self.treeview.connect(
            'key_press_event',
            specific_controller.handle_keypress)
        scrolled_window = gtk.ScrolledWindow()
        scrolled_window.add_with_viewport(self.treeview)
        left_vbox.pack_start(scrolled_window, True, True, PADDING_SMALL)
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
        check_button.connect(
            "clicked",
            specific_controller.set_images_has_fixture)
        self.pack_start(check_button, False, False, PADDING_LARGE)

        #CONTINUE ON PREVIOUS LOG FILE
        frame = gtk.Frame(
            label=model['analysis-stage-image-selection-continue-log'])
        hbox = gtk.HBox(0, False)
        frame.add(hbox)
        self.previous_log = gtk.Label("")
        hbox.pack_start(self.previous_log, True, True, PADDING_SMALL)
        self.log_file_button = gtk.Button(
            label=model['analysis-stage-image-selection-continue-button'])
        hbox.pack_end(self.log_file_button, False, False, PADDING_SMALL)
        self.log_file_button.connect(
            'clicked',
            specific_controller.load_previous_log_file, self)
        self.pack_start(frame, False, False, PADDING_MEDIUM)

        #LOGGING INTEREST SELECTION
        frame = gtk.Frame(
            label=model['analysis-stage-image-selection-logging-title'])
        vbox = gtk.VBox(0, False)

        selections = ((
            'analysis-stage-image-selection-compartments',
            'log-compartments-default',
            specific_controller.log_compartments),
            ('analysis-stage-image-selection-measures',
             'log-measures-default',
             specific_controller.log_measures))

        self._interests = {
            'model': list(), 'selection': list(),
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

        self.only_calibration = gtk.CheckButton(
            label=model['analysis-stage-image-selection-calibration'])
        self.only_calibration.set_active(False)
        self.only_calibration.connect(
            "clicked",
            specific_controller.toggle_calibration)
        vbox.pack_end(self.only_calibration, False, False, PADDING_LARGE)

        frame.add(vbox)
        right_vbox.pack_start(frame, False, True, PADDING_LARGE)

        self.show_all()

    def set_is_calibration(self, val):

        for w in self._interests['widget']:
            w.set_sensitive(val is False)

        self.log_file_button.set_sensitive(val is False)

        if True:

            self.previous_log.set_text("")

    def set_previous_log_file(self, path):

        self.previous_log.set_text(path)

    def set_lock_selection_of_interests(self, val):

        val = not(val)

        for i in xrange(2):

            self._interests['widget'][i].set_sensitive(val)

    def set_interests(self, *args):

        sc = self._specific_controller
        callbacks = (sc.log_compartments, sc.log_measures)

        for i, interest in enumerate(args):

            s = self._interests['selection'][i]
            m = self._interests['model'][i]

            for c in interest:

                for row in m:
                    if c == row[0]:
                        s.select_path(row.path)

            callbacks[i](s)

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
        self._paths = controller.get_top_controller().paths
        self._specific_controller = specific_controller
        self._model = model
        self._specific_model = specific_model

        self._specific_model['image-array'] = plt.imread(
            self._specific_model['images-list-model'][
                self._specific_model['image']][0])

        hbox = gtk.HBox(0, False)
        self.pack_start(hbox, False, False, PADDING_LARGE)

        label = gtk.Label()
        label.set_markup(model['analysis-stage-auto-norm-and-section-title'])
        hbox.pack_start(label, False, False, PADDING_LARGE)

        hbox = gtk.HBox(0, False)
        self.pack_start(hbox, False, False, PADDING_LARGE)

        label = gtk.Label()
        label.set_text(specific_model['images-list-model'][
            specific_model['image']][0])

        hbox.pack_start(label, False, False, PADDING_LARGE)
        #Previously detected option

        hbox = gtk.HBox(0, False)
        self.pack_start(hbox, False, False, PADDING_SMALL)

        self.cb = gtk.CheckButton(
            label=model['analysis-stage-auto-norm-and-section-file'])
        self._cb_signal = self.cb.connect(
            "clicked",
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
        self._fixture_signal = self.fixture.connect(
            "changed",
            self._pre_process_fixture_signal)
        self.run_button = gtk.Button()
        self.run_button.set_label(model['analysis-stage-auto-norm-and-section-run'])
        self.run_button.connect(
            "clicked", specific_controller.execute_fixture,
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
        self.progress.set_sensitive(False)

        if X is None or Y is None:
            if len(sm['auto-transpose']) > sm['image']:
                X = sm['auto-transpose'][sm['image']].source
                Y = sm['auto-transpose'][sm['image']].target

        if X is not None and Y is not None:

            self.figure_ax.set_xlim(0, 256)
            self.figure_ax.set_ylim(min(Y), max(Y))
            self.figure_ax.plot(X, Y)

        else:

            self.figure_ax.set_xlim(0, 1)
            self.figure_ax.set_ylim(0, 1)
            self.figure_ax.text(0.1, 0.4, "Error, could not analyse")

        plt.setp(self.figure_ax.get_yticklabels(),
                 fontsize='xx-small')
        plt.setp(self.figure_ax.get_xticklabels(),
                 fontsize='xx-small')
        self.image_canvas.draw()

    def set_detect_lock(self, val=True):

        self.cb.handler_block(self._cb_signal)
        self.cb.set_active(val)
        self.fixture.set_sensitive(val is False)
        self.cb.handler_unblock(self._cb_signal)

    def run_lock(self):

        self.figure_ax.cla()
        self.figure_ax.set_xlim(0, 1)
        self.figure_ax.set_ylim(0, 1)
        self.figure_ax.text(0.1, 0.4, "Wait while analysing...")
        self.image_canvas.draw()
        self.run_button.set_sensitive(False)

    def run_release(self):

        self.run_button.set_sensitive(True)

    def set_progress(self, value):

        self.progress.set_sensitive(True)
        f = self.progress.get_fraction()

        if 1.0 > f > 0.97:

            self.progress.pulse()

        else:

            self.progress.set_fraction(float(value))

    def set_fixtures_from_model(self, keep_name=True):

        self.fixture.handler_block(self._fixture_signal)

        fixtures = self._controller.get_top_controller().fixtures.get_names()
        #m = self._model

        widget_model = self.fixture.get_model()

        for row in widget_model:
            if row[0] not in fixtures:  # and row[0] != m['one-stage-no-fixture']:
                widget_model.remove(row.iter)
            fixtures = [fix for fix in fixtures if fix != row[0]]

        for f in fixtures:
            self.fixture.append_text(f)

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

        #TODO: Better Size
        imSize = (400, 500)
        self.figure = plt.Figure(figsize=imSize, dpi=150)
        self.figure.add_axes()
        self.figure_ax = self.figure.gca()
        self.set_image()

        self.image_canvas = FigureCanvas(self.figure)
        self.image_canvas.mpl_connect('button_press_event', specific_controller.mouse_button_press)
        self.image_canvas.mpl_connect('button_release_event', specific_controller.mouse_button_release)
        self.image_canvas.mpl_connect('motion_notify_event', specific_controller.mouse_move)

        self.figure_ax.get_xaxis().set_visible(False)
        self.figure_ax.get_yaxis().set_visible(False)

        self.image_canvas.set_size_request(*imSize)

        self._useGrayscale = gtk.CheckButton(label=model[
            'analysis-stage-image-norm-manual-useGrayscale'])
        self._useGrayscale.set_active(
            specific_model['manual-calibration-grayscale'])
        self._useGrayscale.connect('clicked', self._setUseGs)

        self._grayscales = view_generic.get_grayscale_combo()
        self._grayscales.set_activeGrayscale(
            specific_model['manual-calibration-grayscaleName'])
        self._grayscales.connect("changed", self._setUsingGs)

        vbox = gtk.VBox(0, False)
        vbox.pack_start(self.image_canvas, True, True, PADDING_SMALL)
        vbox.pack_start(self._useGrayscale, False, False, PADDING_SMALL)
        vbox.pack_start(self._grayscales, False, False, PADDING_SMALL)
        hbox.pack_start(vbox, False, False, PADDING_SMALL)

        self.treemodel = gtk.ListStore(float, float)
        self.treeview = gtk.TreeView(self.treemodel)
        self.treeview.connect('key_press_event', specific_controller.handle_keypress)
        tv_cell = gtk.CellRendererText()
        tv_cell.connect('edited', self._cell_edited, 'source')
        tv_cell.set_property('editable', True)
        tv_column = gtk.TreeViewColumn(
            model['analysis-stage-image-norm-manual-measures'],
            tv_cell, text=0)
        self.treeview.append_column(tv_column)
        tv_cell = gtk.CellRendererText()
        tv_cell.connect('edited', self._cell_edited, 'target')
        tv_cell.set_property('editable', True)
        tv_column = gtk.TreeViewColumn(
            model['analysis-stage-image-norm-manual-targets'],
            tv_cell,
            text=1)
        self.treeview.append_column(tv_column)
        self.treeview.set_reorderable(False)

        hbox.pack_start(self.treeview, True, True, PADDING_SMALL)

        self.show_all()

    def _cell_edited(self, widget, row, newValue, dtype):

        self._specific_controller.updateManualCalibrationValue(
            dtype, row, newValue)

    def _setUsingGs(self, widget, data=None):

        self._specific_controller.setManualNormNewGrayscale(widget.get_text())

    def _setUseGs(self, widget):

        useGs = widget.get_active()
        self._grayscales.set_active(useGs)
        self._grayscales.set_sensitive(useGs)
        self._specific_controller.setManualNormWithGrayScale(useGs)
        if useGs is False:
            self._grayscales.set_activeGrayscale(useGs)

    def set_image(self):

        self._specific_model['image-array'] = plt.imread(
            self._specific_model['images-list-model'][
                self._specific_model['image']][0])

        im = self._specific_model['image-array']
        """
        if im.max() > 1:
            vmax = 255
        else:
            vmax = 1
        """
        self.figure_ax.imshow(
            im, cmap=plt.cm.gray_r)  # , vmin=0, vmax=vmax)

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

    def add_measure(self, source, target):

        self.treemodel.append((source, target))

    def clear_measures(self):

        self.treemodel.clear()

    def set_measures_from_lists(self, source, target):

        mixList = zip(source, target)

        for row in mixList:
            self.add_measure(*row)

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

    def remove_all_patches(self):

        for p in self.patches:
            p.remove()

        self.patches = []
        self.image_canvas.draw()


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

        self.plate_description.connect(
            "changed", specific_controller.set_in_log, "plate")
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
            self.image_canvas.mpl_connect(
                'button_press_event',
                specific_controller.mouse_button_press))
        self.image_signals.append(
            self.image_canvas.mpl_connect(
                'button_release_event',
                specific_controller.mouse_button_release))
        self.image_signals.append(
            self.image_canvas.mpl_connect(
                'motion_notify_event',
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

        #WARNING
        self._warning = gtk.Label('')
        self._warning_eb = gtk.EventBox()
        self._warning_eb.add(self._warning)
        right_vbox.pack_start(self._warning_eb, False, False, PADDING_SMALL)
        self._warning_eb.modify_bg(gtk.STATE_NORMAL, gtk.gdk.color_parse("red"))
        self._warning.modify_fg(gtk.STATE_NORMAL, gtk.gdk.color_parse("white"))

        #STRAIN
        label = gtk.Label(model['analysis-stage-image-plate-colony-name'])
        right_vbox.pack_start(label, False, False, PADDING_SMALL)

        self.strain_name = gtk.Entry()
        self.strain_name.connect(
            "changed", specific_controller.set_in_log, 'strain')
        right_vbox.pack_start(self.strain_name, False, False, PADDING_SMALL)

        #INDIE CALIBRATION MEASURE
        if specific_model['log-only-calibration']:
            label = gtk.Label(model['analysis-stage-image-plate-calibration'])
            right_vbox.pack_start(label, False, False, PADDING_SMALL)
            self.colony_indie_count = gtk.Entry()
            self.colony_indie_count.connect(
                "changed", specific_controller.set_in_log, 'indie-count')
            self.colony_indie_count.connect(
                "focus", self.select_everything)
            right_vbox.pack_start(
                self.colony_indie_count, False, False, PADDING_SMALL)

        #STRAIN SECTION IM
        self.section_figure = plt.Figure(figsize=(40, 40), dpi=150)
        self.section_figure.add_axes()
        self.section_figure_ax = self.section_figure.gca()

        self.section_image_canvas = FigureCanvas(self.section_figure)
        self.section_image_canvas.mpl_connect(
            'button_press_event',
            specific_controller.man_detect_mouse_press)
        self.section_image_canvas.mpl_connect(
            'button_release_event',
            specific_controller.man_detect_mouse_release)
        self.section_image_canvas.mpl_connect(
            'motion_notify_event',
            specific_controller.man_detect_mouse_move)
        self._man_selection = plt_patches.Circle(
            (-10, -10), 0, ec='b', fill=False, lw=1)
        self.section_figure_ax.add_patch(self._man_selection)

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

        self.log_button = gtk.Button(
            label=model['analysis-stage-image-plate-log-button'])
        self.log_button.connect("clicked", specific_controller.set_in_log,
                                'measures')
        right_vbox.pack_start(self.log_button, False, False, PADDING_LARGE)
        self.set_allow_logging(False)

        self._selection = None
        if specific_model['lock-selection'] is not None:
            pos = [d / 2 for d in specific_model['plate-im-array'].shape][:2]
            pos.reverse()
            self.place_patch_origin(pos, specific_model['lock-selection'])
            self.set_section_image()
        else:
            specific_controller.set_selection(pos=None, wh=None)

        self.show_all()
        self.unset_warning()

    def unset_warning(self):

        self._warning_eb.hide_all()
        self._warning.set_text("")

    def set_warning(self):

        m = self._model
        self._warning.set_text(m['analysis-stage-image-plate-overshoot-warning'])
        self._warning_eb.show_all()

    def set_man_detect_circle(self, origo=(-10, -10), radius=0):

        #HACK: due to inverted axis, this simply works
        self._man_selection.center = origo[::-1]
        self._man_selection.set_radius(radius)
        self.section_image_canvas.draw()

    def select_everything(self, widget, *args, **kwargs):

        widget.select_region(0, -1)

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

        if (self._specific_model['plate-section-im-array'] is not None and
                self._specific_model['plate-section-im-array'].size > 0):

            section = self._specific_model['plate-section-im-array']
            self.section_figure_ax.imshow(
                section, cmap=plt.cm.gray_r,
                vmin=section.min(), vmax=section.max())
            self.section_image_canvas.set_size_request(150, 150)
            self.section_image_canvas.draw()

    def set_analysis_image(self):

        if self._specific_model['plate-section-grid-cell'] is not None:
            blob = self._specific_model[
                'plate-section-grid-cell'].get_item("blob").filter_array

            if blob is not None and blob.size > 0:
                self.analysis_figure_ax.imshow(blob, cmap=plt.cm.Greens)
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

        if parent_model['log-only-calibration']:

            #MEASURES ARE INDIE-COUNT, KEYS, COUNTS
            self.treemodel = gtk.ListStore(*([str] * (
                len(parent_model['log-meta-features']) + 3)))
        else:

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

        if parent_model['log-only-calibration']:

            for c_i, calibration in enumerate(
                    parent_model['calibration-interests']):

                tv_column = gtk.TreeViewColumn(
                    calibration,
                    tv_cell, text=start_col + c_i)

                tv_column.set_resizable(True)
                tv_column.set_reorderable(True)
                self.treeview.append_column(tv_column)
        else:

            totInterest = len(parent_model['log-interests'][1])
            for c_i, compartment in enumerate(parent_model['log-interests'][0]):

                for m_i, measure in enumerate(parent_model['log-interests'][1]):

                    tv_column = gtk.TreeViewColumn(
                        "{0}: {1}".format(compartment, measure),
                        tv_cell, text=start_col + c_i * totInterest + m_i)

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

        try:
            self.treemodel.append(measure)
        except:
            print "Could not append {0}".format(measure)

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
