#!/usr/bin/env python
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

from matplotlib import pyplot as plt
from matplotlib import patches as plt_patches
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas

#
# INTERNAL DEPENDENCIES
#

import scanomatic.gui.generic.view_generic as view_generic
import scanomatic.imageAnalysis.imageFixture as imageFixture

#
# STATIC GLOBALS
#

PADDING_LARGE = view_generic.PADDING_LARGE
PADDING_MEDIUM = view_generic.PADDING_MEDIUM
PADDING_SMALL = view_generic.PADDING_SMALL
PADDING_NONE = view_generic.PADDING_NONE

#
# CLASSES
#


class Analysis(view_generic.Page):

    def __init__(self, controller, model, top=None, stage=None):

        super(Analysis, self).__init__(controller, model, top=top,
                                       stage=stage)

    def _default_top(self):

        widget = Analysis_Top_Root(self._controller, self._model)

        return widget

    def _default_stage(self):

        widget = Analysis_Stage_About(self._controller, self._model)

        return widget


class Analysis_Top_Root(view_generic.Top):

    def __init__(self, controller, model):

        super(Analysis_Top_Root, self).__init__(controller, model)

        button = gtk.Button()
        button.set_label(model["analysis-top-root-project_button-text"])
        button.set_sensitive(False)
        self.pack_start(button, False, False, PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "project")

        """
        button = gtk.Button()
        button.set_label(model["analysis-top-root-color_button-text"])
        self.pack_start(button, False, False, PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "colour")
        """

        button = gtk.Button()
        button.set_label(model["analysis-top-root-1st_pass-text"])
        self.pack_start(button, False, False, PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "1st_pass")

        button = gtk.Button()
        button.set_label(model["analysis-top-root-inspect-text"])
        self.pack_start(button, False, False, PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "inspect")

        button = gtk.Button()
        button.set_label(model["analysis-top-root-convert"])
        self.pack_start(button, expand=False, fill=False,
                        padding=PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "convert")

        button = gtk.Button()
        button.set_label(model["analysis-top-root-features"])
        self.pack_start(button, expand=False, fill=False,
                        padding=PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "extract")

        self.pack_start(gtk.VSeparator(), expand=False, fill=False,
                        padding=PADDING_SMALL)

        button = gtk.Button()
        button.set_label(model["analysis-top-root-tpu_button-text"])
        self.pack_start(button, False, False, PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "transparency")

        self.show_all()


class Analysis_Inspect_Top(view_generic.Top):

    def __init__(self, controller, model):

        super(Analysis_Inspect_Top, self).__init__(controller, model)

        label = gtk.Label(model['analysis-top-inspect-text'])
        self.pack_start(label, True, True, PADDING_SMALL)

        self.show_all()


class Analysis_First_Pass_Top(view_generic.Top):

    def __init__(self, controller, model):

        super(Analysis_First_Pass_Top, self).__init__(controller, model)

        self._start_button = view_generic.Start_Button(controller, model)
        self.pack_end(self._start_button, False, False, PADDING_LARGE)
        self.set_allow_next(False)

        self.show_all()

    def hide_button(self):

        for child in self.children():
            self.remove(child)

    def set_allow_next(self, val):

        self._start_button.set_sensitive(val)


class Analysis_Convert_Top(view_generic.Top):

    def __init__(self, controller, model):

        super(Analysis_Convert_Top, self).__init__(controller, model)

        self.pack_back_button(model['analysis-top-root_button-text'],
                              controller.set_abort, None)

        self.show_all()


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


class Analysis_Convert_Stage(gtk.VBox):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model
        super(Analysis_Convert_Stage, self).__init__(False, spacing=2)

        hbox = gtk.HBox(False, spacing=2)
        hbox.pack_start(gtk.Label(model['convert-xml-select-label']),
                        expand=True, fill=False)
        button = gtk.Button(label=model['convert-xml-select-button'])
        button.connect("clicked", self._selectXmlDialog)
        hbox.pack_start(button, expand=False, fill=False)

        self.pack_start(hbox, expand=False, fill=False, padding=4)

        frame = gtk.Frame(model['convert-xml-conversions'])
        self._conversions = gtk.VBox(False, spacing=2)
        frame.add(self._conversions)

        self.pack_start(frame, expand=True, fill=True, padding=4)

        frame = gtk.Frame(model['convert-xml-conversions-done'])
        self._conversionsDone = gtk.VBox(False, spacing=2)
        frame.add(self._conversionsDone)

        self.pack_start(frame, expand=True, fill=True, padding=4)

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


class Analysis_Inspect_Stage(gtk.VBox):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model
        tc = controller.get_top_controller()
        self._app_config = tc.config
        self._paths = tc.paths

        self._fixture_drawing = None

        super(Analysis_Inspect_Stage, self).__init__()

        self._project_title = gtk.Label(model['analysis-stage-inspect-not_selected'])
        self.pack_start(self._project_title, False, False, PADDING_LARGE)

        button = gtk.Button()
        button.set_label(model['analysis-stage-inspect-select_button'])
        button.connect("clicked", self._select_analysis)
        hbox = gtk.HBox(False, 0)
        hbox.pack_start(button, False, False, PADDING_SMALL)
        self.pack_start(hbox, False, False, PADDING_SMALL)

        self._warning = gtk.Label()
        self.pack_start(self._warning, False, False, PADDING_SMALL)

        scrolled_window = gtk.ScrolledWindow()
        self._display = gtk.VBox()
        scrolled_window.add_with_viewport(self._display)
        self.pack_start(scrolled_window, True, True, PADDING_SMALL)

        self.show_all()

    def _select_analysis(self, widget):

        m = self._model
        base_dir = self._paths.experiment_root
        file_names = view_generic.select_file(
            m['analysis-stage-inspect-analysis-popup'],
            multiple_files=False,
            file_filter=m['analysis-stage-inspect-file-filter'],
            start_in=base_dir)

        if len(file_names) > 0:
            run_file = file_names[0]
            self._project_title.set_text(run_file)
            self._controller.set_analysis(run_file)

    def set_project_name(self, project_name):

            self._project_title.set_text(project_name)

    def set_inconsistency_warning(self):

        w = self._controller.get_window()
        m = self._model
        self._warning.set_text(m['analysis-stage-inspect-warning'])
        view_generic.dialog(
            w, m['analysis-stage-inspect-warning'],
            d_type='error', yn_buttons=False)

    def _toggle_drawing(self, widget, *args):

        if self._fixture_drawing is not None and widget.get_active():
            self._fixture_drawing.toggle_view_state()

    def set_display(self, sm):

        d = self._display
        m = self._model
        p_title = m['analysis-stage-inspect-plate-title']
        p_button = m['analysis-stage-inspect-plate-bad']
        p_no_button = m['analysis-stage-inspect-plate-nohistory']

        for child in d.children():
            d.remove(child)

        hd = gtk.HBox(False, 0)
        d.pack_start(hd, False, False, PADDING_MEDIUM)

        if sm is None:

            label = gtk.Label()
            label.set_markup(m['analysis-stage-inspect-error'])
            hd.pack_start(label, True, True, PADDING_MEDIUM)

        else:

            #ADD DRAWING
            fixture = imageFixture.Image(
                self._paths.experiment_local_fixturename,
                fixture_directory=sm['experiment-dir'])

            vbox = gtk.VBox()
            label = gtk.Label(m['analysis-stage-inspect-plate-drawing'])
            hbox = gtk.HBox()
            self._fixture_drawing = view_generic.Fixture_Drawing(
                fixture, width=300, height=400)
            self._fd_op1 = gtk.RadioButton(
                label=self._fixture_drawing.get_view_state())
            self._fd_op2 = gtk.RadioButton(
                group=self._fd_op1,
                label=self._fixture_drawing.get_other_state())

            vbox.pack_start(label, False, False, PADDING_SMALL)
            hbox.pack_start(self._fd_op1, False, False, PADDING_MEDIUM)
            hbox.pack_start(self._fd_op2, False, False, PADDING_MEDIUM)
            vbox.pack_start(hbox, False, False, PADDING_SMALL)
            vbox.pack_start(self._fixture_drawing, False, False, PADDING_SMALL)
            self._fd_op1.connect('clicked', self._toggle_drawing)
            self._fd_op2.connect('clicked', self._toggle_drawing)
            hd.pack_start(vbox, False, False, PADDING_MEDIUM)

            if sm['pinnings'] is not None:
                #ADD THE PLATES
                for i, plate in enumerate(sm['pinnings']):

                    if plate:

                        vbox = gtk.VBox()
                        label = gtk.Label(p_title.format(
                            i + 1, sm['plate-names'][i]))
                        vbox.pack_start(label, False, False, PADDING_SMALL)
                        image = gtk.Image()
                        image.set_from_file(sm['grid-images'][i])
                        vbox.pack_start(image, True, True, PADDING_SMALL)
                        button = gtk.Button()

                        if (sm['gridding-in-history'] is None or
                                sm['gridding-in-history'][i] is None):

                            button.set_label(p_no_button)
                            button.set_sensitive(False)

                        else:

                            button.set_label(p_button)
                            button.connect("clicked", self._verify_bad, i)

                        vbox.pack_start(button, False, False, PADDING_SMALL)
                        hd.pack_start(vbox, True, True, PADDING_MEDIUM)

            hbox = gtk.HBox(False, 0)
            button = gtk.Button(m['analysis-stage-inspect-upload-button'])
            button.connect('clicked', self._controller.launch_filezilla)
            hbox.pack_end(button, False, False, PADDING_NONE)
            d.pack_start(hbox, False, False, PADDING_SMALL)

        d.show_all()

    def warn_remove_failed(self):

        w = self._controller.get_window()
        m = self._model
        view_generic.dialog(
            w, m['analysis-stage-inspect-plate-remove-warn'],
            d_type='error', yn_buttons=False)

    def _verify_bad(self, widget, plate):

        w = self._controller.get_window()
        m = self._model
        if view_generic.dialog(
                w, m['analysis-stage-inspect-plate-yn'].format(plate),
                d_type="info", yn_buttons=True):

            widget.set_sensitive(False)
            if self._controller.remove_grid(plate):

                widget.set_label(m['analysis-stage-inspect-plate-gone'])
            else:
                widget.set_label(m['analysis-stage-inspect-plate-nohistory'])


class Analysis_Stage_About(gtk.Label):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model

        super(Analysis_Stage_About, self).__init__()

        self.set_justify(gtk.JUSTIFY_LEFT)
        self.set_markup(model['analysis-stage-about-text'])

        self.show()


class Analysis_Stage_First_Pass_Running(gtk.VBox):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model
        self._specific_model = controller.get_specific_model()
        specific_model = self._specific_model

        super(Analysis_Stage_First_Pass_Running, self).__init__(False, 0)

        label = gtk.Label()
        label.set_markup(model['analysis-stage-first-running-intro'].format(
            specific_model['meta-data']['Prefix']))

        self.pack_start(label, False, False, PADDING_LARGE)

        """
        self._progress = gtk.ProgressBar()
        self.pack_start(self._progress, False, False, PADDING_LARGE)

        self._errors = gtk.Label()
        self.pack_start(self._errors, False, False, PADDING_LARGE)
        """

        self.show_all()

    def update(self):

        pass

        """
        sm = self._specific_model
        m = self._model

        self._progress.set_fraction(sm['run-position'])

        if sm['run-complete']:
            self._progress.set_text(m['analysis-stage-first-running-complete'])
        else:
            self._progress.set_text(m['analysis-stage-first-running-working'])

        if sm['run-error'] is not None:

            self._errors.set_text(sm['run-error'])
            self._errors.show()

        else:

            self._errors.hide()
        """


class Analysis_Stage_First_Pass(gtk.VBox):

    ID_LENGTHS = 4
    ID_CTRL_LENGTH = 3

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model
        self._specific_model = controller.get_specific_model()
        specific_model = self._specific_model

        super(Analysis_Stage_First_Pass, self).__init__(False, 0)

        label = gtk.Label()
        label.set_markup(model['analysis-stage-first-title'])
        self.pack_start(label, False, False, PADDING_SMALL)

        #WHERE
        frame = gtk.Frame(model['analysis-stage-first-where'])
        vbox = gtk.VBox(False, 0)
        frame.add(vbox)
        self.pack_start(frame, False, False, PADDING_SMALL)

        ##BUTTON TO SELECT DIR
        button = gtk.Button()
        button.connect("clicked", controller.set_output_dir)
        hbox = gtk.HBox(False, 0)
        hbox.pack_start(button, False, False, PADDING_SMALL)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)
        button.set_label(model['analysis-stage-first-dir'])

        ##INFO ON DIRECTORY
        hbox = gtk.HBox(False, 0)
        label = gtk.Label(model['analysis-stage-first-dir-title'])
        hbox.pack_start(label, False, False, PADDING_MEDIUM)
        self._dir_label = gtk.Label("")
        hbox.pack_start(self._dir_label, True, True, PADDING_SMALL)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)

        ##INFO ON FILE
        hbox = gtk.HBox(False, 0)
        label = gtk.Label(model['analysis-stage-first-file'])
        hbox.pack_start(label, False, False, PADDING_MEDIUM)
        self._file_entry = gtk.Entry()
        self._file_entry.set_width_chars(55)
        self._file_entry.connect(
            "focus-out-event",
            controller.update_model, 'output-file')
        hbox.pack_start(self._file_entry, False, False, PADDING_SMALL)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)

        ##USE FIXTURE-CONF IN DIRECTORY
        hbox = gtk.HBox(False, 0)
        self._local_fixture = gtk.CheckButton(
            label=model['analysis-stage-first-local_fixture'])
        hbox.pack_start(self._local_fixture, False, False, PADDING_SMALL)
        self._local_fixture.set_sensitive(False)
        self._local_fixture.set_active(False)
        self._local_fixture.connect(
            "toggled",
            controller.set_local_fixture, 'local-fixture')
        self.pack_start(hbox, False, False, PADDING_SMALL)

        #FILES
        scrolled = gtk.ScrolledWindow()
        self.pack_start(scrolled, True, True, PADDING_MEDIUM)

        ##SELECTED FILE LIST
        if specific_model['image-list-model'] is None:
            treemodel = gtk.ListStore(str)
            specific_model['image-list-model'] = treemodel
        else:
            treemodel = specific_model['image-list-model']

        self._treeview = gtk.TreeView(treemodel)

        tv_cell = gtk.CellRendererText()
        tv_column = gtk.TreeViewColumn(
            model['analysis-stage-first-column-title'],
            tv_cell, text=0)
        self._treeview.append_column(tv_column)
        self._treeview.set_reorderable(True)
        self._treeview.connect(
            'key_press_event',
            controller.handle_keypress)
        self._treeview.get_selection().set_mode(gtk.SELECTION_MULTIPLE)
        self._treeview.set_rubber_banding(True)
        scrolled.add_with_viewport(self._treeview)
        scrolled.set_size_request(-1, 100)

        #META-DATA
        frame = gtk.Frame(model['analysis-stage-first-meta'])
        self.pack_start(frame, False, False, PADDING_MEDIUM)
        vbox = gtk.VBox(False, 0)
        frame.add(vbox)
        ##THE REST...
        hbox = gtk.HBox(False, 0)
        vbox.pack_start(hbox, False, False, PADDING_MEDIUM)
        table = gtk.Table(rows=3, columns=2, homogeneous=False)
        table.set_col_spacings(PADDING_MEDIUM)
        hbox.pack_start(table, False, False, PADDING_SMALL)
        ##PREFIX
        label = gtk.Label(model['analysis-stage-first-meta-prefix'])
        label.set_alignment(0, 0.5)
        self._prefix = gtk.Entry()
        self._prefix.connect(
            'focus-out-event',
            controller.update_model, 'prefix')
        table.attach(label, 0, 1, 0, 1)
        hbox = gtk.HBox(False, 0)
        hbox.pack_start(self._prefix, False, False, PADDING_NONE)
        table.attach(hbox, 1, 2, 0, 1)
        ##IDENTIFIER
        label = gtk.Label(model['analysis-stage-first-meta-id'])
        label.set_alignment(0, 0.5)

        self.project_id = gtk.Entry(self.ID_LENGTHS)
        self.project_id.connect("changed", self.set_id_val,
                                controller.ID_PROJECT)
        self.scan_layout_id = gtk.Entry(self.ID_LENGTHS)
        self.scan_layout_id.connect("changed", self.set_id_val,
                                    controller.ID_LAYOUT)
        self.id_control = gtk.Entry(self.ID_CTRL_LENGTH)
        self.id_control.connect("changed", self.set_id_val,
                                controller.ID_CONTROL)
        self.id_control_warning = gtk.Image()

        self.project_id.set_sensitive(False)
        self.scan_layout_id.set_sensitive(False)
        self.id_control.set_sensitive(False)

        self.set_id_val(None, None)

        hbox = gtk.HBox(False, 0)
        hbox.pack_start(self.project_id, False, False, PADDING_NONE)
        hbox.pack_start(self.scan_layout_id, False, False, PADDING_SMALL)
        hbox.pack_start(self.id_control, False, False, PADDING_SMALL)
        hbox.pack_start(self.id_control_warning, False, False, PADDING_SMALL)

        table.attach(label, 0, 1, 1, 2)
        table.attach(hbox, 1, 2, 1, 2)
        ##DESCRIPTION
        label = gtk.Label(model['analysis-stage-first-meta-desc'])
        label.set_alignment(0, 0.5)
        self._project_desc = gtk.Entry()
        self._project_desc.connect(
            "focus-out-event",
            controller.update_model, 'desc')
        self._project_desc.set_width_chars(55)
        table.attach(label, 0, 1, 2, 3)
        table.attach(self._project_desc, 1, 2, 2, 3)

        #FIXTURE AND SCANNER
        frame = gtk.Frame(model['analysis-stage-first-fixture_scanner'])
        self.pack_start(frame, False, False, PADDING_MEDIUM)
        hbox = gtk.HBox(False, 0)
        frame.add(hbox)
        label = gtk.Label(model['analysis-stage-first-scanner'])
        hbox.pack_start(label, False, False, PADDING_SMALL)
        self._scanner = gtk.Entry()
        self._scanner.connect(
            "focus-out-event", controller.update_model, 'scanner')
        hbox.pack_start(self._scanner, False, False, PADDING_MEDIUM)
        label = gtk.Label(model['analysis-stage-first-fixture'])
        hbox.pack_start(label, False, False, PADDING_SMALL)
        self._fixture = view_generic.get_fixtures_combo()
        view_generic.set_fixtures_combo(
            self._fixture,
            controller.get_top_controller().fixtures)

        self._fixture.connect(
            "changed", controller.update_model, None, 'fixture')
        hbox.pack_start(self._fixture, False, False, PADDING_SMALL)

        #PINNING
        frame = gtk.Frame(model['analysis-stage-first-plates'])
        self.pack_start(frame, False, False, PADDING_MEDIUM)
        vbox = gtk.VBox()
        frame.add(vbox)
        hbox = gtk.HBox()
        label = gtk.Label(model['analysis-stage-first-plates-number'])
        hbox.pack_start(label, False, False, PADDING_MEDIUM)
        self._plates = gtk.Entry(1)
        self._plates.set_text("0")
        self._plates.set_width_chars(2)
        self._plates.connect('changed', self._update_number_of_plates)
        hbox.pack_start(self._plates, False, False, PADDING_MEDIUM)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)
        self._pm_box = gtk.HBox()
        vbox.pack_start(self._pm_box, False, True, PADDING_SMALL)

    def set_id_val(self, entry, eType):
        """Callback for id-entry stuff."""

        if entry is not None:
            curVal = entry.get_text()
            curValUpper = curVal.upper()

            if (curVal != curValUpper and
                (eType == self._controller.ID_PROJECT or
                 eType == self._controller.ID_LAYOUT)):

                entry.set_text(curValUpper)

            if (eType == self._controller.ID_PROJECT and
                    len(curVal) == self.ID_LENGTHS):

                self.scan_layout_id.grab_focus()

            elif (eType == self._controller.ID_LAYOUT and
                  len(curVal) == self.ID_LENGTHS):

                self.id_control.grab_focus()

        ref_num = self._controller.get_ctrl_id_num(
            self.project_id.get_text(),
            self.scan_layout_id.get_text())

        if (eType == self._controller.ID_PROJECT or
                eType == self._controller.ID_LAYOUT):

            self._controller.update_model(entry, None, eType)

        if (entry is None and self.project_id.get_text() != "" and
                self.scan_layout_id.get_text() != "" and
                self.id_control.get_text() == ""):

            self.id_control.set_text(str(ref_num))

        try:
            ctrl_num = int(self.id_control.get_text())
        except:
            ctrl_num = 0

        if ref_num == ctrl_num:

            self.id_control_warning.set_from_stock(
                gtk.STOCK_APPLY,
                gtk.ICON_SIZE_SMALL_TOOLBAR)

            self.id_control_warning.set_tooltip_text("")

        else:

            self.id_control_warning.set_from_stock(
                gtk.STOCK_STOP,
                gtk.ICON_SIZE_SMALL_TOOLBAR)

            self.id_control_warning.set_tooltip_text(
                self._model['analysis-stage-first-id-warn'])

    def _update_number_of_plates(self, widget):

        t = widget.get_text()
        if t == "":
            return

        try:
            i = int(t)
        except:
            self._plates.set_text("0")
            return

        self._controller.set_new_plates(i)

    def set_pinning(self, sensitive=True):

        pinnings_list = self._specific_model['meta-data']['Pinning Matrices']

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

            elif len(children) > len(pinnings_list):

                for p in xrange(len(children) - len(pinnings_list)):
                    box.remove(children[-1 - p])

                children = box.children()

            for i, child in enumerate(children):

                child.set_sensitive(sensitive)
                child.set_pinning(pinnings_list[i])

        box.show_all()

    def delete_selection(self):

        sel = self._treeview.get_selection()
        result = sel.get_selected_rows()

        if result is not None:

            model, pathlist = result

            if pathlist is not None:

                for i in sorted(pathlist, reverse=True):
                    del model[i]

    def update_local_fixture(self, has_fixture):

        self._local_fixture.set_sensitive(has_fixture)
        self._local_fixture.set_active(has_fixture)
        self.project_id.set_sensitive(True)
        self.scan_layout_id.set_sensitive(True)
        self.id_control.set_sensitive(True)
        self.id_control.set_text("")
        self.set_id_val(None, None)

    def update(self):

        sm = self._specific_model
        self._dir_label.set_text(sm['output-directory'])
        self._file_entry.set_text(sm['output-file'])
        if sm['meta-data'] is not None:
            md = sm['meta-data']
            self._project_desc.set_text(md['Description'])
            self.project_id.set_text(md['Project ID'])
            try:
                self.scan_layout_id.set_text(md['Scanner Layout ID'])
            except:
                pass
            self._prefix.set_text(md['Prefix'])
            try:
                self._scanner.set_text(md['Scanner'])
            except:
                md['Scanner'] = 'Unknown'
                self._scanner.set_text(md['Scanner'])

            view_generic.set_fixtures_active(self._fixture, name=md['Fixture'])
            try:
                self._plates.set_text(str(len(md['Pinning Matrices'])))
            except:
                self._plates.set_text("0")


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
