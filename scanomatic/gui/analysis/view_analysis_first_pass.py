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
    PADDING_SMALL, PADDING_MEDIUM, PADDING_LARGE, PADDING_NONE

#
# CLASSES
#


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
