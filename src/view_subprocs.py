#!/usr/bin/env python
"""The GTK-GUI view for subprocs"""
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
import gobject
import os
import time
import inspect

#
# INTERNAL DEPENDENCIES
#

from src.view_generic import *
from src.gui.subprocs.event.event import Event

#
# STATIC GLOBALS
#

"""Gotten from view_generic instead
PADDING_LARGE = 10
PADDING_MEDIUM = 4
PADDING_SMALL = 2
"""

#
# METHODS
#


def whoCalled(fn):

    def wrapped(*args, **kwargs):
        frames = []
        frame = inspect.currentframe().f_back
        while frame.f_back:
            frames.append(inspect.getframeinfo(frame)[2])
            frame = frame.f_back
        frames.append(inspect.getframeinfo(frame)[2])

        print "===\n{0}\n{1}\n{2}\nCalled by {3}\n____".format(
            fn, args, kwargs, ">".join(frames[::-1]))

        fn(*args, **kwargs)

    return wrapped

#
# CLASSES
#


class _Running_Frame(gtk.Frame):

    def __init__(self, subproc_controller, proc, model, init_info_msg=None):

        super(_Running_Frame, self).__init__(model['process-unknown'])

        self._subprocs = subproc_controller

        self._subprocs.add_event(Event(
            proc.set_callback_prefix, self._set_title, None))
        self._subprocs.add_event(Event(
            proc.set_callback_total, self._set_total_iterations, None))

        self._model = model
        self._proc = proc
        self._no_more_action = False
        self._first_is_done = False

        self._current_iteration = -1
        self._total_iterations = None

        vbox = gtk.VBox(False, 0)

        #Progress bar area
        self._progress = gtk.ProgressBar()
        vbox.pack_start(self._progress, False, False, PADDING_SMALL)

        #Info area
        self._info_area = gtk.HBox(False, 0)
        if init_info_msg is not None:
            self._info = gtk.Label(model[init_info_msg])
        else:
            self._info = gtk.Label("")
        self._info_area.pack_start(self._info, True, True, PADDING_LARGE)

        vbox.pack_start(self._info_area, False, False, PADDING_MEDIUM)

        self.add(vbox)
        self.show_all()

    def _set_title(self, proc, title):

        self.set_label(title)

    def add_button(self, b, pack_start=True):

        if pack_start:
            pack = self._info_area.pack_start
        else:
            pack = self._info_area.pack_end

        pack(b, False, False, PADDING_MEDIUM)

    def _set_progress_bar(self, proc, frac):

        m = self._model
        cur = self._current_iteration

        if frac < 0 or frac is None:
            frac = 0
        elif frac > 1:
            frac = 1

        self._progress.set_fraction(frac)

        if cur > 1:

            self._first_is_done = True

            dt = time.time() - self._proc.get_start_time()
            eta = (1 - frac) * dt / frac

            if eta < 0:
                eta = 0

            self._progress.set_text(
                m['running-eta'].format(
                    eta / 3600.0))

        else:

            self._progress.set_text(
                m['running-elapsed'].format(
                    (time.time() - self._proc.get_start_time()) / 60.0))

    def _set_total_iterations(self, proc, total):

        self._total_iterations = total

    def _set_info_iteration_text(self, proc, cur):

        m = self._model
        self._current_iteration = cur

        if self._total_iterations is not None and cur > 0:
            self._info.set_text(m['running-progress'].format(
                cur, self._total_iterations))

    def _set_done_state(self, proc, is_alive):

        m = self._model
        if is_alive is False:

            if proc.get_exit_code() != 0:
                self._info.set_text(m['running-done'])
            else:
                self._info.set_text(m['running-done-error'])
            self._progress.set_fraction(1.0)
            self._progress.set_text("")
            self._no_more_action = True

        else:

            self._subprocs.add_event(Event(
                proc.set_callback_progress, self._set_progress_bar, None))
            self._subprocs.add_event(Event(
                proc.set_callback_current,
                self._set_info_iteration_text, None))

    def update(self):

        proc = self._proc

        if self._no_more_action is False:

            self._subprocs.add_event(Event(
                proc.set_callback_is_alive, self._set_done_state, False,
                responseTimeOut=10))

        return self._no_more_action


class _Running_Analysis(_Running_Frame):

    def __init__(self, subproc_controller, proc, model, controller):

        super(_Running_Analysis, self).__init__(subproc_controller,
                                                proc, model,
                                                'running-analysis-running')

        self._buttons = []

        self._grid_button = gtk.Button(
            label=model['running-analysis-view-gridding'])

        self._buttons.append((self._grid_button,
                              controller.produce_inspect_gridding,
                              False))

        self._grid_button.set_sensitive(False)

        self._subprocs.add_event(Event(
            proc.set_callback_prefix, self.connect_buttons, None))

        self.add_button(self._grid_button)

    def connect_buttons(self, proc, prefix):

        for i in range(len(self._buttons))[::-1]:

            button, callback, sensitivity = self._buttons[i]

            button.connect("clicked", callback, prefix)
            button.set_sensitive(sensitivity)

            del self._buttons[i]

    def update(self):

        super(_Running_Analysis, self).update()
        self._grid_button.set_sensitive(self._first_is_done)


class Subprocs_View(gtk.Frame):

    def __init__(self, controller, model, specific_model):

        super(Subprocs_View, self).__init__(model['composite-stat-title'])

        self._model = model
        self._controller = controller
        self._specific_model = specific_model

        table = gtk.Table(rows=5, columns=2)
        table.set_col_spacings(PADDING_MEDIUM)
        #table.set_row_spacing(0, PADDING_MEDIUM)
        #table.set_row_spacing(3, PADDING_MEDIUM)
        self.add(table)

        #FREE SCANNERS
        label = gtk.Label(model['free-scanners'])
        label.set_alignment(0, 0.5)
        table.attach(label, 0, 1, 0, 1)

        self.scanners = gtk.Button()
        self.scanners.set_label('-')
        self.scanners.connect("clicked", controller.produce_free_scanners)
        self.scanners.set_alignment(0, 0.5)
        table.attach(self.scanners, 1, 2, 0, 1)

        #PROJECTS
        label = gtk.Label(model['live-projects'])
        label.set_alignment(0, 0.5)
        table.attach(label, 0, 1, 1, 2)

        self.projects = gtk.Button()
        self.projects.set_label('-')
        self.projects.connect("clicked", controller.produce_live_projects)
        table.attach(self.projects, 1, 2, 1, 2)

        #RUNNING EXPERIMENTS
        label = gtk.Label(model['running-experiments'])
        label.set_alignment(0, 0.5)
        table.attach(label, 0, 1, 2, 3)

        self.experiments = gtk.Button()
        self.experiments.set_label('-')
        self.experiments.connect("clicked",
                                 controller.produce_running_experiments)
        table.attach(self.experiments, 1, 2, 2, 3)

        #RUNNING ANALYSIS
        label = gtk.Label(model['running-analysis'])
        label.set_alignment(0, 0.5)
        table.attach(label, 0, 1, 3, 4)

        self.analysis = gtk.Button()
        self.analysis.set_label('-')
        self.analysis.connect("clicked", controller.produce_running_analysis)
        table.attach(self.analysis, 1, 2, 3, 4)

        #ERRORS AND WARNINGS
        label = gtk.Label(model['queue'])
        label.set_alignment(0, 0.5)
        table.attach(label, 0, 1, 4, 5)

        self.queue = gtk.Button()
        self.queue.set_label('-')
        table.attach(self.queue, 1, 2, 4, 5)

        self.show_all()

    def update(self):

        specific_model = self._specific_model

        self.projects.set_label(str(specific_model['live-projects']))
        self.scanners.set_label(str(specific_model['free-scanners']))

        self.experiments.set_label(str(specific_model['experiments'].count()))
        self.analysis.set_label(str(specific_model['analysises'].count()))
        self.queue.set_label(str(specific_model['queue'].count()))


class Running_Analysis(gtk.VBox):

    def __init__(self, controller, model, specific_model):

        super(Running_Analysis, self).__init__(False, 0)

        self._controller = controller
        self._subprocs = controller.get_top_controller().subprocs
        self._model = model
        self._specific_model = specific_model

        label = gtk.Label()
        label.set_markup(model['running-analysis-intro'])
        self.pack_start(label, False, False, PADDING_LARGE)

        self._stuff = gtk.VBox(False, 0)
        self.pack_start(self._stuff, False, False, PADDING_MEDIUM)

        for proc in specific_model['analysises']:

            self._stuff.pack_start(_Running_Analysis(self._subprocs,
                                                     proc, model, controller))

        self.show_all()
        gobject.timeout_add(6037, self.update)

    def update(self):

        for proc_frame in self._stuff.get_children():

            proc_frame.update()

        return True


class Running_Experiments(gtk.VBox):

    def __init__(self, controller, model, specific_model):

        super(Running_Experiments, self).__init__(False, 0)

        self._controller = controller
        self._subprocs = controller.get_top_controller().subprocs
        self._model = model
        self._specific_model = specific_model

        label = gtk.Label()
        label.set_markup(model['running-experiments-intro'])
        self.pack_start(label, False, False, PADDING_LARGE)

        self._stuff = gtk.VBox(False, 0)
        self.pack_start(self._stuff, False, False, PADDING_MEDIUM)
        for proc in specific_model['experiments']:

            self._stuff.pack_start(_Running_Frame(self._subprocs, proc, model,
                                                  None))

        self.show_all()
        gobject.timeout_add(6029, self.update)

    def _verify_stop(self, widget, proc):

        def _verify_sure(widget, b_yes):

            if widget.get_text().lower() == 'stop':
                b_yes.set_sensitive(True)
            else:
                b_yes.set_sensitive(False)

        m = self._model
        dialog = gtk.MessageDialog(
            self._controller.get_window(),
            gtk.DIALOG_DESTROY_WITH_PARENT,
            gtk.MESSAGE_WARNING, gtk.BUTTONS_NONE,
            "")

        dialog.add_button(gtk.STOCK_NO, False)
        b_yes = dialog.add_button(gtk.STOCK_YES, True)
        b_yes.set_sensitive(False)

        vbox = dialog.get_children()[0]
        hbox, bbox = vbox.get_children()
        im, vbox2 = hbox.get_children()
        vbox2.remove(vbox2.get_children()[1])
        label = vbox2.get_children()[0]
        label.set_markup(
            m['running-experiments-stop-warning'].format(
                proc['sm']['experiment-prefix']))

        entry = gtk.Entry()
        entry.connect("changed", _verify_sure, b_yes)
        vbox2.pack_start(entry, False, False, PADDING_SMALL)

        dialog.show_all()

        resp = dialog.run()

        dialog.destroy()

        if resp == 1:

            self._controller.stop_process(proc)
            proc['progress'] = proc['sm']['scans']
            widget.set_sensitive(False)
            widget.set_label(m['running-experiments-stopping'])
            self.update()

    def update(self):

        for proc_frame in self._stuff:

            proc_frame.update()

        return True


class Free_Scanners(gtk.VBox):

    def __init__(self, controller, model, specific_model):

        super(Free_Scanners, self).__init__(False, 0)

        self._controller = controller
        self._model = model
        self._specific_model = specific_model

        label = gtk.Label()
        label.set_markup(model['free-scanners-intro'])
        self.pack_start(label, False, False, PADDING_LARGE)

        frame = gtk.Frame(model['free-scanners-frame'])
        self.pack_start(frame, False, False, PADDING_LARGE)

        self._scanners = gtk.HBox(False, 0)

        frame.add(self._scanners)

        self.show_all()
        gobject.timeout_add(4349, self.update,
                            controller.get_top_controller().scanners)

    def update(self, scanners):

        for child in self._scanners:
            self._scanners.remove(child)

        for scanner in scanners.get_names():

            label = gtk.Label(scanner)
            self._scanners.pack_start(label, False, False, PADDING_LARGE)

        self._scanners.show_all()

        return True


class Errors_And_Warnings(gtk.VBox):

    def __init__(self, controller, model, specific_model):

        super(Errors_And_Warnings, self).__init__(False, 0)

        self._controller = controller
        self._model = model
        self._specific_model = specific_model

        label = gtk.Label()
        label.set_markup(model['collected-messages-intro'])
        self.pack_start(label, False, False, PADDING_LARGE)

        self.show_all()


class Live_Projects(gtk.ScrolledWindow):

    def __init__(self, controller, model, specific_model):

        super(Live_Projects, self).__init__()
        self._controller = controller
        self._model = model
        self._specific_model = specific_model

        #TITLE
        vbox = gtk.VBox(False, 0)
        label = gtk.Label()
        label.set_markup(model['project-progress-title'])
        vbox.pack_start(label, False, False, PADDING_LARGE)

        #PROJECTS CONAINER
        self._projects = gtk.VBox()
        vbox.pack_start(self._projects, False, False, PADDING_NONE)

        #FINALIZE
        self.add_with_viewport(vbox)
        self.show_all()

        self.update(controller._project_progress)

        gobject.timeout_add(6217, self.update,
                            controller._project_progress)

    def update(self, project_progress):

        m = self._model
        pstages = m['project-progress-stages']
        pstage_title = m['project-progress-stage-title']
        pstage_spacer = m['project-progress-stage-spacer']
        pstatus = m['project-progress-stage-status']
        pview = self._projects

        cur_status = project_progress.get_all_stages_status(as_text=True)
        cur_keys = cur_status.keys()

        def _update_button(button, new_text, i, prefix):

            if button.get_label() != new_text:
                if hasattr(button, '_current_signal'):
                    button.handler_disconnect(button._current_signal)

                b_signal = None

                button.set_label(new_text)
                if new_text not in pstatus[2:4]:  # Launch or Running
                    button.set_sensitive(False)
                else:
                    button.set_sensitive(True)

                if new_text == pstatus[2]:
                    if i == 1:
                        b_signal = button.connect(
                            "clicked",
                            self._controller.produce_launch_analysis,
                            prefix)
                    elif i == 2:
                            b_signal = button.connect(
                                "clicked",
                                self._controller.produce_inspect_gridding,
                                prefix)
                    elif i == 3:
                            b_signal = button.connect(
                                "clicked",
                                self._controller.produce_upload,
                                prefix)

                elif new_text == pstatus[3]:
                    if i == 0:
                        b_signal = button.connect(
                            "clicked",
                            self._controller.produce_running_experiments)
                    elif i == 1:
                        b_signal = button.connect(
                            "clicked",
                            self._controller.produce_running_analysis)
                    elif i == 2:
                            b_signal = button.connect(
                                "clicked",
                                self._controller.produce_inspect_gridding,
                                prefix)
                    elif i == 3:
                            b_signal = button.connect(
                                "clicked",
                                self._controller.produce_upload,
                                prefix)

                if b_signal:
                    button._current_signal = b_signal

        for view_proj in pview.get_children():

            pname = view_proj.get_name()
            if pname not in cur_keys:

                pview.remove(view_proj)

            else:

                for i in range(len(pstages)):
                    _update_button(view_proj.my_status[i],
                                   cur_status[pname][i], i,
                                   pname)

                del cur_status[pname]

        #APPEND NEW
        for k, val in cur_status.items():

            frame = gtk.Frame(k)
            frame.set_name(k)

            hbox = gtk.HBox()
            frame.add(hbox)

            frame.my_status = []

            for i in range(len(pstages)):

                vbox = gtk.VBox()
                label = gtk.Label()
                label.set_markup(pstage_title.format(pstages[i]))
                vbox.pack_start(label, False, False, PADDING_SMALL)

                button = gtk.Button()
                _update_button(button, val[i], i, k)

                frame.my_status.append(button)
                vbox.pack_start(button, False, False, PADDING_SMALL)

                hbox.pack_start(vbox, False, False, PADDING_LARGE)

                if i < len(pstages) - 1:
                    label = gtk.Label()
                    label.set_markup(pstage_spacer)
                    hbox.pack_start(label, False, False, PADDING_LARGE)

            button = gtk.Button()
            button.set_label(m['project-progress-manual_remove'])
            button.connect('clicked', self._ask_manual_remove, k)
            hbox.pack_end(button, False, False, PADDING_LARGE)

            pview.pack_start(frame, False, False, PADDING_MEDIUM)

        pview.show_all()
        return True

    def _ask_manual_remove(self, widget, prefix):

        m = self._model
        c = self._controller
        if dialog(c.get_window(),
                  m['project-progress-dialog'],
                  d_type='warning',
                  yn_buttons=True):

            c.remove_live_project(prefix)


class View_Gridding_Images(gtk.ScrolledWindow):

    def __init__(self, controller, model, specific_model):

        super(View_Gridding_Images, self).__init__()

        self._controller = controller
        self._model = model
        self._specific_model = specific_model

        sm = specific_model

        plate = 1
        file_pattern = "grid___origin_plate_{0}.svg"
        file_path = os.sep.join(
            (sm['analysis-project-log_file_dir'],
                sm['analysis-project-output-path'],
                file_pattern))

        hbox = gtk.HBox(False, 0)
        self.add_with_viewport(hbox)

        while True:

            if os.path.isfile(file_path.format(plate)):
                vbox = gtk.VBox(False, 0)
                label = gtk.Label()
                label.set_markup(model['view-plate-pattern'].format(plate))
                vbox.pack_start(label, False, False, PADDING_LARGE)
                image = gtk.Image()
                image.set_from_file(file_path.format(plate))
                vbox.pack_start(image, False, False, PADDING_LARGE)
                hbox.pack_start(vbox, False, False, PADDING_LARGE)
            else:
                break

            plate += 1

        self.show_all()
