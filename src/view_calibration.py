#!/usr/bin/env python
"""The GTK-GUI view for the general layout"""
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
import matplotlib.image as plt_img
import matplotlib.pyplot as plt
import matplotlib.text as plt_text
import matplotlib.patches as plt_patches
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

class Calibration_View(Page):

    def __init__(self, controller, model, top=None, stage=None):

        super(Calibration_View, self).__init__(controller, model,
            top=top, stage=stage)

    def _default_stage(self):

        return Stage_About(self._controller, self._model)

    def _default_top(self):

        return Top_Root(self._controller, self._model)


class Top_Root(Top):

    def __init__(self, controller, model):

        super(Top_Root, self).__init__(controller, model)

        button = gtk.Button()
        button.set_label(model['mode-selection-top-fixture'])
        button.connect("clicked", controller.set_mode, 'fixture')
        self.pack_start(button, False, False, PADDING_MEDIUM)

        button = gtk.Button()
        button.set_label(model['mode-selection-top-poly'])
        button.connect("clicked", controller.set_mode, 'poly')
        self.pack_start(button, False, False, PADDING_MEDIUM)

class Stage_About(gtk.Label):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model

        super(Stage_About, self).__init__()

        self.set_justify(gtk.JUSTIFY_LEFT)
        self.set_markup(model['calibration-stage-about-text'])

        self.show()

class Fixture_Select_Top(Top):

    def __init__(self, controller, model, specific_model):

        super(Fixture_Select_Top, self).__init__(controller, model)
        self._specific_model = specific_model

        self._next_button = Top_Next_Button(controller, model, specific_model,
            model['fixture-select-next'], controller.set_view_stage, 
            'marker-calibration')
        self.set_allow_next(False)

        self.pack_end(self._next_button, False, False, PADDING_SMALL)
        self.show_all()

    def set_allow_next(self, val):

        self._next_button.set_sensitive(val)

class Fixture_Select_Stage(gtk.VBox):

    def __init__(self, controller, model, specific_model):

        super(Fixture_Select_Stage, self).__init__(0, False)
        self._controller = controller
        self._model = model
        self._specific_model = specific_model

        #TITLE
        label = gtk.Label()
        label.set_markup(model['fixture-select-title'])
        self.pack_start(label, False, False, PADDING_LARGE)

        #EDIT BUTTON
        self.edit_fixture = gtk.RadioButton(group=None, 
            label=model['fixture-select-radio-edit'])
        self.edit_fixture.connect("clicked",
            self.toggle_new_fixture, False)
        self.pack_start(self.edit_fixture, False, False, PADDING_SMALL)

        #CURRENT FIXTURES
        self.fixtures = gtk.ListStore(str)
        self.treeview = gtk.TreeView(self.fixtures)
        tv_cell = gtk.CellRendererText()
        tv_column = gtk.TreeViewColumn(
            model['fixture-select-column-header'],
            tv_cell, text=0)

        self.treeview.append_column(tv_column)
        self.selection = self.treeview.get_selection()
        self.selection_signal = self.selection.connect('changed', 
            controller.check_fixture_select, False)
 
        fixtures = controller.get_top_controller().fixtures.names()
        for f in sorted(fixtures):
            self.fixtures.append([f])

        self.pack_start(self.treeview, False, False, PADDING_LARGE)

        #NEW BUTTON
        self.new_fixture = gtk.RadioButton(group=self.edit_fixture,
            label=model['fixture-select-radio-new'])
        self.new_fixture.connect("clicked",
            self.toggle_new_fixture, True)
        self.pack_start(self.new_fixture, False, False, PADDING_SMALL)

        #NEW NAME
        hbox = gtk.HBox(False, 0)
        self.new_name = gtk.Entry()
        self.new_name.connect("changed", controller.check_fixture_select, True)
        self.new_name.set_sensitive(False)
        hbox.pack_start(self.new_name, True, True, PADDING_MEDIUM)
        self.name_warning = gtk.Image()
        hbox.pack_end(self.name_warning, False, False, PADDING_SMALL)

        self.pack_start(hbox, False, False, PADDING_SMALL)        

        self.show_all()

    def toggle_new_fixture(self, widget, is_new):

        if widget.get_active():
            self.treeview.set_sensitive(is_new==False)
            self.new_name.set_sensitive(is_new)

            self._controller.check_fixture_select(None, is_new)

    def set_bad_name_warning(self, warn):

        if warn == False:

            self.name_warning.set_from_stock(gtk.STOCK_APPLY,
                    gtk.ICON_SIZE_SMALL_TOOLBAR)
            self.name_warning.set_tooltip_text(
                self._model['fixture-select-new-name-ok'])
            self.name_warning.show()

        elif warn == True:

            self.name_warning.set_from_stock(gtk.STOCK_DIALOG_WARNING,
                    gtk.ICON_SIZE_SMALL_TOOLBAR)
            self.name_warning.set_tooltip_text(
                self._model['fixture-select-new-name-duplicate'])
            self.name_warning.show()

        else:

            self.name_warning.hide()


class Fixture_Marker_Calibration_Top(Top):

    def __init__(self, controller, model, specific_model):

        super(Fixture_Marker_Calibration_Top, self).__init__(controller, model)
        self._specific_model = specific_model

        self._next_button = Top_Next_Button(controller, model, specific_model,
            model['fixture-calibration-next'], controller.set_view_stage, 
            'segmentation')
        self.set_allow_next(False)

        self.pack_end(self._next_button, False, False, PADDING_SMALL)
        self.show_all()

    def set_allow_next(self, val):

        self._next_button.set_sensitive(val)


class Fixture_Marker_Calibration_Stage(gtk.VBox):

    def __init__(self, controller, model, specific_model):

        super(Fixture_Marker_Calibration_Stage, self).__init__(0, False)
        self._controller = controller
        self._model = model
        self._specific_model = specific_model

        fixture_callbacks={'button_press_event': controller.mouse_press,
            'button_release_event': controller.mouse_release,
            'button_notify_event': controller.mouse_move}

        self.fixture_image = Fixture_Image(self._specific_model,
            event_callbacks=fixture_callbacks)

        self.pack_start(self.fixture_image.get_canvas())

class Fixture_Image(object):

    def __init__(self, model, event_callbacks=None, full_size=False):

        """
        event_callbacks     If None (default) the canvas won't
        emit events. If a dictionary it will connect the dict's
        keys as events with the corresponding values as callbacks.
        See matplotlib.canvas for valid mpl_connect-events (but
        in short these might be of interest: 'button_press_event',
        'button_release_event', 'motion_notify_event'

        full_size   If False (default) the canvas will be scaled to
        fit a reasonable area, if true the canvas will be placed in
        a scrollwindow.
        """
        self._model = model 
        self._im_overlays = dict()
        image_size=(300, 200)

        self.image_fig = plt.Figure(figsize=image_size, dpi=150)
        self.image_ax = self.image_fig.add_subplot(111)
        image_canvas = FigureCanvas(self.image_fig)

        self.image_ax.get_xaxis().set_visible(False)
        self.image_ax.get_yaxis().set_visible(False)

        if event_callbacks is not None:
            for event, callback in event_callbacks.items():
                self.image_fig.canvas.mpl_connect(event, callback)

        if 'im' in model and model['im'] is not None:
            self.load_from_array()
        elif 'im-path' in model and model['im-path'] is not None:
            self.load_from_path()
        elif 'im-not-loaded' in model and model['im-not-loaded'] is not None:
            self.set_not_loaded_text()

    def load_from_array(self):

        model = self._model
        self.image_ax.imshow(model['im'])
        self._im = im
        self.clear_overlays()
        self.image_fig.canvas.draw()

    def load_from_path(self):

        model = self._model
        im = plt.imread(model['im-path'])
        if im is not None:
            self.load_from_array(im)
        else:
            model['im-path'] = None

    def set_not_loaded_text(self): 

        model = self._model

        self._set_text(text=model['im-not-loaded'], x=0.5, y=0.5,
            overlay_key='no-im')

    def _set_text(self, text, x, y, overlay_key, alpha=0.75,
            color='#001166'):

        if overlay_key in self._im_overlays.keys():

            t = self._im_overlays[overlay_key]
            t.set_x(x)
            t.set_y(y)
            t.set_text(text)

        else:

            self._im_overlays[overlay_key] = plt_text.Text(x=x, y=y,
                text=text, alpha=alpha,
                horizontalalignment='center', family='serif',
                verticalalignment='center', size='large',
                weight='bold', color=color)

            self.image_ax.add_artist(self._im_overlays[overlay_key])

    def _set_rect(self, coords, overlay_key, color='#228822', lw=2,
            alpha=0.5):

        x, y = map(min, zip(*coords))
        w, h = [a-b for a,b in zip(map(sum, zip(*coords)), (x,y))]

        if overlay_key in self._im_overalys.keys():

            rect = self._im_overlays[overlay_key]
            rect.set_xy((x, y))
            rect.set_width(w)
            rect.set_height(h)

        else:

            rect = plt_patches.Rectangle((x, y), w, h, color=color,
                lw=lw, alpha=alpha, fill=False)
            rect.get_axes()
            rect.get_transform()
            self.image_ax.add_patch(rect)
            self._im_overlays[overlay_key] = rect

    def clear_overlays(self):

        for overlay in self._im_overlays:

            self.image_ax.remove(overlay)

        self._im_overlays = dict()

    def clear_overlay(self, overlay):

        self.image_ax.remove(self._im_overlays[overlay])
        del self._im_overlays[overlay]

    def clear_overlay_markers(self):

        model = self._model

        for m in xrange(len(model['marker-positions'])):

            overlay = "marker_{0}".format(m)
            self.clear_overlay(overlay)

    def clear_plate_overlay(self, plate_index):

        plate_patch_overlay = "plate_{0}_rect".format(plate_index)
        plate_text_overlay = "plate_{0}_text".format(plate_index)

        self.clear_overlays(plate_patch_overlay)
        self.clear_overlays(plate_text_overlay)
        
    def set_marker_overlays(self):

        model = self._model

        for i, (x, y) in enumerate(model['marker-positions']):

            self._set_text('x', x, y, 'marker_{0}'.format(i),
                alpha=0.5)

    def set_plate_overlay(self, plate_index):

        plate_patch_overlay = "plate_{0}_rect".format(plate_index)
        plate_text_overlay = "plate_{0}_text".format(plate_index)

        model = self._model
        plate = model['plate-coords'][plate_index]
        center_x, center_y = [p/2.0 for p in map(sum, zip(*plate))]

        self._set_text(plate_index, center_x, center_y,
            plate_text_overlay, alpha=0.5)
        
        self._set_rect(plate, plate_patch_overlay)

    def get_canvas(self):

        return self.image_fig.canvas
