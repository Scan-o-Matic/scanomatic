#!/usr/bin/env python
"""The GTK-GUI view for the general layout"""
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
#import gobject

#
# INTERNAL DEPENDENCIES
#

from generic.view_generic import *
import scanomatic.io.paths as paths

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


class Splash(gtk.Window):

    def __init__(self, program_path):

        super(Splash, self).__init__()

        vbox = gtk.VBox(False, 0)
        self.add(vbox)

        image = gtk.Image()
        image.set_from_file(paths.Paths().logo)
        vbox.pack_start(image, False, False, PADDING_NONE)

        label = gtk.Label("Loading...")
        vbox.pack_start(label, False, False, PADDING_NONE)

        self._keep_alive = True

        self.show_all()
        width, height = self.get_size()
        self.move((gtk.gdk.screen_width() - width) / 2, (gtk.gdk.screen_height() - height) / 2)

    def main_is_loaded(self):

        self._keep_alive = False
        self.hide()
        self.destroy()


class Main_Window(gtk.Window):

    def __init__(self, controller=None, model=None):

        super(Main_Window, self).__init__()

        self.set_default_size(800, 600)
        self.move(0, 0)
        self._model = model
        self._controller = controller

        #SECTIONING MAIN AREA
        hbox = gtk.HBox(False, 0)
        self.add(hbox)
        self._panel = gtk.VBox(False, 0)
        self._stats_area = None
        hbox.pack_start(self._panel, False, False, PADDING_SMALL)
        self._content_notebook = gtk.Notebook()
        hbox.pack_end(self._content_notebook, True, True, PADDING_SMALL)
        self.logo = gtk.Image()
        self.logo.set_from_file(controller.paths.logo)
        hbox.pack_end(self.logo, True, True, PADDING_SMALL)

        if model is not None:
            self.populate_panel()
            self.set_window_properties()

        self.connect("delete_event", self._win_close_event)

    def _win_close_event(self, widget, *args, **kwargs):

        return self._controller.ask_quit()

    def set_current_page(self, val=-1):

        self._content_notebook.set_current_page(val)

    def set_model(self, model):

        self._model = model
        self.populate_panel()

    def set_controller(self, controller):

        self._controller = controller
        self.populate_panel()

    def set_window_properties(self):

        m = self._model
        self.set_title(m['window-title'])

    def populate_panel(self):

        if self._model is None or self._controller is None:
            return None
        else:
            return self._populate_panel()

    def _populate_panel(self):

        panel = self._panel
        #CLEANING UP THE UGLY WAY
        while len(panel.children()) > 0:

            panel.remove(panel.children()[0])

        m = self._model
        c = self._controller

        #ADDING BUTTONS AREA
        frame = gtk.Frame(m['panel-actions-title'])
        panel.pack_start(frame, False, False, PADDING_SMALL)
        vbox = gtk.VBox(False, 0)
        frame.add(vbox)

        button = gtk.Button()
        button.set_label(m['panel-actions-experiment'])
        button.connect("clicked", c.add_contents, 'experiment')
        vbox.pack_start(button, False, False, PADDING_MEDIUM)

        button = gtk.Button()
        button.set_label(m['panel-actions-analysis'])
        button.connect("clicked", c.add_contents, 'analysis')
        vbox.pack_start(button, False, False, PADDING_MEDIUM)

        button = gtk.Button()
        button.set_label(m['panel-actions-qc'])
        button.connect("clicked", c.add_contents, 'qc')
        vbox.pack_start(button, False, False, PADDING_MEDIUM)

        vbox.pack_start(gtk.HSeparator(), False, False, PADDING_MEDIUM)

        button = gtk.Button()
        button.set_label(m['panel-actions-calibration'])
        button.connect("clicked", c.add_contents, 'calibration')
        vbox.pack_start(button, False, False, PADDING_MEDIUM)

        button = gtk.Button()
        button.set_label(m['panel-actions-config'])
        button.connect("clicked", c.add_contents, 'config')
        vbox.pack_start(button, False, False, PADDING_MEDIUM)

        button = gtk.Button()
        button.set_label(m['panel-actions-quit'])
        button.connect("clicked", c.ask_quit, 'quit')
        vbox.pack_start(button, False, False, PADDING_MEDIUM)

        if self._stats_area is not None:
            panel.pack_start(self._stats_area, False, False, PADDING_LARGE)

        panel.show_all()

    def populate_stats_area(self, stats_widget):

        self._stats_area = stats_widget

    def show_notebook_or_logo(self):

        if self._content_notebook.get_n_pages() > 0:

            self._content_notebook.show_all()
            self.logo.hide()

        else:

            self._content_notebook.hide()
            self.logo.show()

    def add_notebook_page(self, page, title_text, specific_controller):

        button = gtk.Button()
        im = gtk.Image()
        im.set_from_stock(gtk.STOCK_CLOSE, gtk.ICON_SIZE_SMALL_TOOLBAR)
        button.add(im)
        button.connect("clicked", self._controller.remove_contents,
                       specific_controller)

        label = gtk.Label(title_text)

        title = gtk.HBox(False, 0)
        title.pack_start(label, True, True, PADDING_SMALL)
        title.pack_end(button, False, False, PADDING_SMALL)

        title.show_all()

        self._content_notebook.append_page(page, title)

        self.show_notebook_or_logo()

    def remove_notebook_page(self, widget):

        tag_label = widget.get_parent()
        n = self._content_notebook
        for i, page in enumerate(n.children()):

            if tag_label == n.get_tab_label(page):

                n.remove_page(i)
                break

        self.show_notebook_or_logo()
