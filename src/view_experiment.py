#!/usr/bin/env python
"""The GTK-GUI view for experiments"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import pygtk
pygtk.require('2.0')
import gtk
"""
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
import matplotlib.image as plt_img
import matplotlib.pyplot as plt
import matplotlib.text as plt_text
import matplotlib.patches as plt_patches
"""

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

class Experiment_View(Page):

    def __init__(self, controller, model, top=None, stage=None):

        super(Experiment_View, self).__init__(controller, model,
            top=top, stage=stage)

    def _default_stage(self):

        return Stage_About(self._controller, self._model)

    def _default_top(self):

        return Top_Root(self._controller, self._model)


class Top_Root(Top):

    def __init__(self, controller, model):

        super(Top_Root, self).__init__(controller, model)

        button = gtk.Button()
        button.set_label(model['mode-selection-top-project'])
        button.connect("clicked", controller.set_mode, 'project')
        self.pack_start(button, False, False, PADDING_MEDIUM)

        button = gtk.Button()
        button.set_label(model['mode-selection-top-gray'])
        button.connect("clicked", controller.set_mode, 'gray')
        self.pack_start(button, False, False, PADDING_MEDIUM)

        button = gtk.Button()
        button.set_label(model['mode-selection-top-color'])
        button.connect("clicked", controller.set_mode, 'color')
        self.pack_start(button, False, False, PADDING_MEDIUM)

class Stage_About(gtk.Label):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model

        super(Stage_About, self).__init__()

        self.set_justify(gtk.JUSTIFY_LEFT)
        self.set_markup(model['project-stage-about-text'])


class Top_Project_Setup(Top):

    def __init__(self, controller, model, specific_model):

        super(Top_Project_Setup, self).__init__(controller, model)

        self._specific_model = specific_model

        self._start_button = Start_Button(controller, model)
        self.pack_end(self._start_button, False, False, PADDING_LARGE)
        self.set_allow_next(False)

        self.show_all()

    def set_allow_next(self, val):

        self._start_button.set_sensitive(val)
        

class Top_Project_Running(Top):

    def __init__(self, controller, model, specific_model):

        super(Top_Project_Running, self).__init__(controller, model)

        self._specific_model = specific_model

        self.show_all()


class Top_One(Top):

    def __init__(self, controller, model, specific_model):

        super(Top_One, self).__init__(controller, model)

        self._specific_model = specific_model

        self.show_all()


class Stage_One(gtk.VBox):

    def __init__(self, controller, model, specific_model):

        self._controller = controller
        self._model = model
        self._specific_model = specific_model

        super(Stage_One, self).__init__(0, False)

        self.gtk_handlers = dict()

        #INTRO
        label = gtk.Label()
        label.set_markup(model[controller.get_model_intro_key()])
        self.pack_start(label, False, False, PADDING_LARGE)

        #FIXTURE AND SCANNER
        frame = gtk.Frame(model['one-stage-fixture_scanner'])
        self.pack_start(frame, False, False, PADDING_MEDIUM)
        hbox = gtk.HBox(False, 0)
        frame.add(hbox)
        label = gtk.Label(model['one-stage-scanner'])
        hbox.pack_start(label, False, False, PADDING_SMALL)
        self.scanner = gtk.combo_box_new_text()
        self.gtk_handlers['scanner-changed'] = self.scanner.connect(
            "changed", controller.set_new_scanner)
        hbox.pack_start(self.scanner, False, False, PADDING_MEDIUM)
        label = gtk.Label(model['one-stage-fixture'])
        hbox.pack_start(label, False, False, PADDING_SMALL)
        self.fixture = gtk.combo_box_new_text()
        self.gtk_handlers['fixture-changed'] = self.fixture.connect(
            "changed", controller.set_new_fixture)
        hbox.pack_start(self.fixture, False, False, PADDING_SMALL)

        #RUN-TYPES
        frame = gtk.Frame(model['one-stage-run-frame'])
        self.pack_start(frame, False, False, PADDING_MEDIUM)
        vbox = gtk.VBox(False, 0)
        hbox = gtk.HBox(False, 0)
        vbox.pack_start(hbox, False, False, PADDING_SMALL)
        self._button_1 = gtk.Button(label=model['one-stage-power-up'])
        self._button_1_command = ['power-up']
        self._button_1.connect('clicked', controller.set_run,
            self._button_1_command)
        hbox.pack_start(self._button_1, False, False, PADDING_MEDIUM)
        self._button_2 = gtk.Button(label=model['one-stage-run-all'])
        self._button_2_command = ['run-all']
        self._button_2.connect('clicked', controller.set_run,
            self._button_2_command)
        hbox.pack_start(self._button_2, False, False, PADDING_MEDIUM)
        frame.add(vbox)

        #PROGRESS
        frame = gtk.Frame(model['one-stage-progress'])
        self.pack_start(frame, False, False, PADDING_MEDIUM)
        vbox = gtk.VBox(False, 0)
        frame.add(vbox)
        label = gtk.Label(model['one-stage-progress-colors'])
        vbox.pack_start(label, False, False, PADDING_MEDIUM)
        hbox = gtk.HBox(False, 0)
        vbox.pack_start(hbox, False, False, PADDING_MEDIUM)
        self._progress_on = gtk.Label()
        self._progress_on.set_markup(model['one-stage-progress-not-run'].format(
            model['one-stage-progress-power-on']))
        hbox.pack_start(self._progress_on, False, False, PADDING_SMALL)
        self._progress_scan = gtk.Label()
        self._progress_scan.set_markup(model['one-stage-progress-not-run'].format(
            model['one-stage-progress-scan']))
        hbox.pack_start(self._progress_scan, False, False, PADDING_SMALL)
        self._progress_off = gtk.Label()
        self._progress_off.set_markup(model['one-stage-progress-not-run'].format(
            model['one-stage-progress-power-off']))
        hbox.pack_start(self._progress_off, False, False, PADDING_SMALL)
        self._progress_analysis = gtk.Label()
        self._progress_analysis.set_markup(model['one-stage-progress-not-run'].format(
            model['one-stage-progress-analysis']))
        hbox.pack_start(self._progress_analysis, False, False, PADDING_SMALL)
        self._progress_done = gtk.Label()
        self._progress_done.set_markup(model['one-stage-progress-not-run'].format(
            model['one-stage-progress-done']))
        hbox.pack_start(self._progress_done, False, False, PADDING_SMALL)

        self.set_fixtures()
        self.set_scanners()
        self.set_run_stage('not-ready')


        self.show_all()

    def set_progress(self, stage, completed=False, surpass=None,
        failed=None):

        m = self._model

        if completed:
            progress = m['one-stage-progress-completed']
        else:
            progress = m['one-stage-progress-running']

        if surpass:
            progress = m['one-stage-progress-pass']
        elif surpass == False:
            progress = m['one-stage-progress-not-run']
        if failed:
            progress = m['one-stage-progress-error']

        if stage == 'on':
            widget = self._progress_on
            text = m['one-stage-progress-power-on']
        elif stage == 'scan':
            widget = self._progress_scan
            text = m['one-stage-progress-scan']
        elif stage == 'off':
            widget = self._progress_off
            text = m['one-stage-progress-power-off']
        elif stage == 'analysis':
            widget = self._progress_analysis
            text = m['one-stage-progress-analysis']
        elif stage == 'done':
            widget = self._progress_done
            text = m['one-stage-progress-done']

        widget.set_markup(progress.format(text))
    
    def set_run_stage(self, stage):

        m = self._model

        if stage in ('running', 'not-ready'):

            self._button_1.set_sensitive(False)
            self._button_2.set_sensitive(False)

        if stage == 'ready':

            self._button_1.set_sensitive(True)
            self._button_2.set_sensitive(True)

        elif stage == 'started':

            self.fixture.set_sensitive(False)
            self.scanner.set_sensitive(False)

        elif stage == 'power-on':

            self._button_1.set_label(m['one-stage-scan'])
            self._button_1_command[0] = 'scan'
            self._button_2.set_label(m['one-stage-complete'])
            self._button_2_command[0] = 'complete'

    def set_fixtures(self):

        fixtures = self._controller.get_top_controller().fixtures.get_names()
        m = self._model

        widget_model = self.fixture.get_model()

        if len(widget_model) == 0:
            self.fixture.append_text(m['one-stage-no-fixture'])


        for row in widget_model:
            if row[0] not in fixtures and row[0] != m['one-stage-no-fixture']:
                widget_model.remove(row.iter)
            fixtures = [fix for fix in fixtures if fix != row[0]]

        for f in sorted(fixtures):
            self.fixture.append_text(f)
        
    def set_scanners(self):

        scanners = self._controller.get_top_controller().scanners.get_names(available=True)
        for s in self.scanner:
            if s not in scanners:
                self.scanner.remove(s)
            scanners = [sc for sc in scanners if sc != s]

        for s in scanners:
            self.scanner.append_text(s)

    def update_scanner(self):

        scanner = self._specific_model['scanner']
        self.scanner.handler_block(self.gtk_handlers['scanner-changed'])
        if scanner is None:

            self.scanner.set_active(-1)

        else:

            widget_model = self.scanner.get_model()
            for i, row in enumerate(widget_model):

                if row[0] == scanner:
                    self.scanner.set_active(i)
                    break

        self.scanner.handler_unblock(self.gtk_handlers['scanner-changed'])

    def force_no_fixture(self):

        self.fixture.set_active(0)
        self.fixture.set_sensitive(False)

class Stage_Project_Running(gtk.VBox):

    def __init__(self, controller, model, specific_model):

        self._controller = controller
        self._model = model
        self._specific_model = specific_model

        super(Stage_Project_Running, self).__init__(0, False)

        label = gtk.Label()
        label.set_markup(self._model['project-running'].format(
            self._specific_model['experiment-prefix']))
        self.pack_start(label, False, False, PADDING_LARGE)

        self.show_all()

class Stage_Project_Setup(gtk.VBox):

    def __init__(self, controller, model, specific_model=None):

        self._controller = controller
        self._model = model

        if specific_model is None:
            specific_model = controller.get_specific_model()

        self._specific_model = specific_model

        sm = self._specific_model
        self.gtk_handlers = dict()

        super(Stage_Project_Setup, self).__init__(0, False)

        #METADATA
        frame = gtk.Frame(model['project-stage-meta'])
        self.pack_start(frame, False, False, PADDING_MEDIUM)
        vbox = gtk.VBox(False, 0)
        frame.add(vbox)
        ##FOLDER
        hbox = gtk.HBox(False, 0)
        self.project_root = gtk.Label(specific_model['experiments-root'])
        hbox.pack_start(self.project_root, True, True, PADDING_MEDIUM)
        button = gtk.Button(label=model['project-stage-select_root'])
        button.connect("clicked", controller.set_project_root)
        hbox.pack_end(button, False, False, PADDING_SMALL)
        vbox.pack_start(hbox, False, False, PADDING_MEDIUM)
        ##THE REST...
        hbox = gtk.HBox(False, 0)
        vbox.pack_start(hbox, False, False, PADDING_MEDIUM)
        table = gtk.Table(rows=3, columns=2, homogeneous=False) 
        table.set_col_spacings(PADDING_MEDIUM)
        hbox.pack_start(table, False, False, PADDING_SMALL)
        ##PREFIX
        label = gtk.Label(model['project-stage-prefix'])
        label.set_alignment(0, 0.5)
        self.prefix = gtk.Entry()
        self.gtk_handlers['prefix'] = self.prefix.connect('changed', 
            controller.check_prefix_dupe)
        table.attach(label, 0, 1, 0, 1)
        self.warn_image = gtk.Image()
        self.set_prefix_status(False)
        hbox = gtk.HBox(False, 0)
        hbox.pack_start(self.prefix, False, False, PADDING_NONE)
        hbox.pack_start(self.warn_image, False, False, PADDING_NONE)
        table.attach(hbox, 1, 2, 0, 1)
        ##IDENTIFIER
        label = gtk.Label(model['project-stage-planning-id'])
        label.set_alignment(0, 0.5)
        self.project_id = gtk.Entry()
        self.project_id.connect("focus-out-event", controller.set_project_id)
        hbox = gtk.HBox(False, 0)
        hbox.pack_start(self.project_id, False, False, PADDING_NONE)
        table.attach(label, 0, 1, 1, 2)
        table.attach(hbox, 1, 2, 1, 2)
        ##DESCRIPTION
        label = gtk.Label(model['project-stage-desc'])
        label.set_alignment(0, 0.5)
        self.project_desc = gtk.Entry()
        self.project_desc.connect("focus-out-event", 
            controller.set_project_description)
        self.project_desc.set_width_chars(55)
        table.attach(label, 0, 1, 2, 3)
        table.attach(self.project_desc, 1, 2, 2, 3)

        #FIXTURE AND SCANNER
        frame = gtk.Frame(model['project-stage-fixture_scanner'])
        self.pack_start(frame, False, False, PADDING_MEDIUM)
        hbox = gtk.HBox(False, 0)
        frame.add(hbox)
        label = gtk.Label(model['project-stage-scanner'])
        hbox.pack_start(label, False, False, PADDING_SMALL)
        self.scanner = gtk.combo_box_new_text()
        self.gtk_handlers['scanner-changed'] = self.scanner.connect(
            "changed", controller.set_new_scanner)
        hbox.pack_start(self.scanner, False, False, PADDING_MEDIUM)
        label = gtk.Label(model['project-stage-fixture'])
        hbox.pack_start(label, False, False, PADDING_SMALL)
        self.fixture = gtk.combo_box_new_text()
        self.gtk_handlers['fixture-changed'] = self.fixture.connect(
            "changed", controller.set_new_fixture)
        hbox.pack_start(self.fixture, False, False, PADDING_SMALL)

        #TIME, INTERVAL, SCANS
        frame = gtk.Frame(model['project-stage-time_settings'])
        hbox = gtk.HBox(0, False)
        frame.add(hbox)
        table = gtk.Table(rows=3, columns=2, homogeneous=False)
        table.set_col_spacings(PADDING_MEDIUM)
        hbox.pack_start(table, False, False, PADDING_SMALL)
        self.pack_start(frame, False, False, PADDING_MEDIUM)
        ##DURATION
        label = gtk.Label(model['project-stage-duration'])
        label.set_alignment(0, 0.5)
        table.attach(label, 0, 1, 0, 1)
        self.project_duration = gtk.Entry()
        self.project_duration.set_text(self._get_human_duration('duration'))
        self.gtk_handlers['duration-changed'] = self.project_duration.connect("changed", 
            controller.check_experiment_duration,
            "duration")
        self.gtk_handlers['duration-exit'] = self.project_duration.connect(
            "focus-out-event",
            self._set_duration_value, 'duration')
        self.duration_warning = gtk.Image()
        hbox = gtk.HBox(False, 0)
        hbox.pack_start(self.project_duration, False, False, PADDING_SMALL)
        hbox.pack_start(self.duration_warning, False, False, PADDING_SMALL)
        table.attach(hbox, 1, 2, 0, 1)
        ##INTERVAL
        label = gtk.Label(model['project-stage-interval'])
        label.set_alignment(0, 0.5)
        table.attach(label, 0, 1, 1, 2)
        self.project_interval = gtk.Entry()
        self.project_interval.set_text(self._get_human_duration('interval'))
        self.gtk_handlers['interval-changed'] = self.project_interval.connect("changed", 
            controller.check_experiment_duration,
            "interval")
        self.gtk_handlers['interval-exit'] = self.project_interval.connect(
            "focus-out-event",
            self._set_duration_value, 'interval')
        self.interval_warning = gtk.Image()
        hbox = gtk.HBox(False, 0)
        hbox.pack_start(self.project_interval, False, False, PADDING_SMALL)
        hbox.pack_start(self.interval_warning, False, False, PADDING_SMALL)
        table.attach(hbox, 1, 2, 1, 2)
        ##SCANS
        label = gtk.Label(model['project-stage-scans'])
        label.set_alignment(0, 0.5)
        table.attach(label, 0, 1, 2, 3)
        self.project_scans = gtk.Entry()
        self.project_scans.set_text(self._get_human_duration('scans'))
        self.gtk_handlers['scans-changed'] = self.project_scans.connect("changed", 
            controller.check_experiment_duration,
            "scans")
        self.gtk_handlers['scans-exit'] = self.project_scans.connect(
            "focus-out-event",
            self._set_duration_value, 'scans')
        self.scans_warning = gtk.Image()
        hbox = gtk.HBox(False, 0)
        hbox.pack_start(self.project_scans, False, False, PADDING_SMALL)
        hbox.pack_start(self.scans_warning, False, False, PADDING_SMALL)
        table.attach(hbox, 1, 2, 2, 3)

        #PINNING
        frame = gtk.Frame(model['project-stage-plates'])
        self.pack_start(frame, False, False, PADDING_MEDIUM)
        vbox = gtk.VBox()
        frame.add(vbox)
        self.pm_box = gtk.HBox()
        vbox.pack_start(self.pm_box, False, True, PADDING_SMALL)
        #self.keep_gridding.clicked()

        self.set_fixtures()
        self.set_scanners()


    def _get_human_duration(self, w_type):

        sm = self._specific_model
        m = self._model

        if w_type == "duration":
            t = sm['duration']
            td = int(t / 24)
            t = t % 24
            th = int(t)
            tm = int((t-th)*60)

            return m['project-stage-duration-format'].format(td, th, tm)

        elif w_type == "scans":

            return m['project-stage-scans-format'].format(sm['scans'])

        else:

            return m['project-stage-interval-format'].format(sm['interval'])

    def _get_warn_image(self, w_type):

        if w_type == "duration":

            cur_img = self.duration_warning

        elif w_type == "scans":

            cur_img = self.scans_warning

        else:

            cur_img = self.interval_warning

        return cur_img

    def warn_scanner_claim_fail(self):

        self.scanner.handler_block(self.gtk_handlers['scanner-changed'])
        
        self.scanner.set_active(-1)

        dialog(self._controller._window,
            self._model['project-stage-scanner-claim-fail'],
            d_type="warning", yn_buttons=False)

        self.scanner.handler_unblock(self.gtk_handlers['scanner-changed'])

    def update_experiment_root(self):

        t = self._specific_model['experiments-root']
        self.project_root.set_text(t is None and "" or t)

    def set_duration_warning(self, w_type):

        cur_img = self._get_warn_image(w_type)
        cur_img.set_from_stock(gtk.STOCK_DIALOG_WARNING,
            gtk.ICON_SIZE_SMALL_TOOLBAR)
        cur_img.set_tooltip_text(self._model['project-stage-duration-warn'])

    def remove_duration_warning(self, w_type):

        cur_img = self._get_warn_image(w_type)
        cur_img.set_from_stock(gtk.STOCK_APPLY,
            gtk.ICON_SIZE_SMALL_TOOLBAR)
        cur_img.set_tooltip_text(self._model['project-stage-duration-ok'])

    def set_other_duration_values(self):

        for duration_name in self._specific_model['duration-settings-order'][0:2]:

            self._set_duration_value(None, None, duration_name)

    def _set_duration_value(self, widget, event, duration_name):

        if widget is None:
            if duration_name == "interval":
                widget = self.project_interval
            elif duration_name == 'scans':
                widget = self.project_scans
            elif duration_name == 'duration':
                widget = self.project_duration

        if widget is None:
            return

        widget.handler_block(self.gtk_handlers[duration_name+"-changed"])

        widget.set_text(self._get_human_duration(duration_name))
        self.remove_duration_warning(duration_name)
        widget.handler_unblock(self.gtk_handlers[duration_name+"-changed"])

    def set_prefix_status(self, is_ok):

        m = self._model

        if is_ok:

            self.warn_image.set_from_stock(gtk.STOCK_APPLY,
                    gtk.ICON_SIZE_SMALL_TOOLBAR)
            self.warn_image.set_tooltip_text(m['project-stage-prefix-ok'])

        else:

            self.warn_image.set_from_stock(gtk.STOCK_DIALOG_WARNING,
                    gtk.ICON_SIZE_SMALL_TOOLBAR)
            self.warn_image.set_tooltip_text(m['project-stage-prefix-warn'])

    def set_fixtures(self):

        fixtures = self._controller.get_top_controller().fixtures.get_names()
        m = self._model

        widget_model = self.fixture.get_model()

        for row in widget_model:
            if row[0] not in fixtures:
                widget_model.remove(row.iter)
            fixtures = [fix for fix in fixtures if fix != row[0]]

        for f in sorted(fixtures):
            self.fixture.append_text(f)

    def set_scanners(self):

        scanners = self._controller.get_top_controller().scanners.get_names(available=True)
        widget_model = self.scanner.get_model()

        for row in widget_model:

            if row[0] not in scanners:
                widget_model.remove(row.iter)

            scanners = [sc for sc in scanners if sc != row[0]]

        for s in sorted(scanners):
            self.scanner.append_text(s)

    def set_pinning(self, pinnings_list=None, sensitive=True):

        if pinnings_list is None:
            pinnings_list = self._specific_model['pinnings-list']

        box = self.pm_box

        children = box.children()

        if pinnings_list is not None:

            if len(children) < len(pinnings_list):

               for p in xrange(len(pinnings_list) - len(children)):

                    box.pack_start(Pinning(
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

