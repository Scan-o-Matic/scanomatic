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
import scanomatic.imageAnalysis.imageFixture as image_fixture

#
# STATIC GLOBALS
#

from scanomatic.gui.generic.view_generic import \
    PADDING_SMALL, PADDING_MEDIUM, PADDING_LARGE, PADDING_NONE

#
# CLASSES
#


class Analysis_Inspect_Top(view_generic.Top):

    def __init__(self, controller, model):

        super(Analysis_Inspect_Top, self).__init__(controller, model)

        label = gtk.Label(model['analysis-top-inspect-text'])
        self.pack_start(label, True, True, PADDING_SMALL)

        self.show_all()


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
            fixture = image_fixture.Image(
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
