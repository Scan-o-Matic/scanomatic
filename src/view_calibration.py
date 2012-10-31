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
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
import matplotlib.image as plt_img
import matplotlib.pyplot as plt
import matplotlib.text as plt_text
import matplotlib.patches as plt_patches


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

            if is_new:
                self.new_name.grab_focus()
            else:
                self.treeview.grab_focus()

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

        #TITLE
        label = gtk.Label()
        label.set_markup(model['fixture-calibration-title'])
        self.pack_start(label, False, False, PADDING_SMALL)

        #SELECT IMAGE
        hbox = gtk.HBox(0, False)
        self.im_path = gtk.Label()
        hbox.pack_start(self.im_path, True, True, PADDING_SMALL)
        button = gtk.Button()
        button.set_label(model['fixture-calibration-select-im'])
        button.connect("clicked", controller.set_image_path)
        hbox.pack_end(button, False, False, PADDING_SMALL)
        self.pack_start(hbox, False, False, PADDING_SMALL)

        #IMAGE DISPLAY
        """fixture_callbacks={'button_press_event': controller.mouse_press,
            'button_release_event': controller.mouse_release,
            'motion_notify_event': controller.mouse_move}
        """
        fixture_callbacks = None
        self.fixture_image = Fixture_Image(self._specific_model,
            event_callbacks=fixture_callbacks)
        self.fixture_image.set_marker_overlays()
        self.pack_start(self.fixture_image.get_canvas(), True, True,
            PADDING_SMALL)

        #MARKERS
        hbox = gtk.HBox(0, False)
        label = gtk.Label()
        label.set_text(model['fixture-calibration-marker-number'])
        hbox.pack_start(label, False, False, PADDING_SMALL)
        self.number_of_markers = gtk.Entry()
        self.number_of_markers.connect("changed",
            controller.set_number_of_markers)
        hbox.pack_start(self.number_of_markers, False, False, PADDING_MEDIUM)
        self.run_detect = gtk.Button(
            label=model['fixture-calibration-marker-detect'])
        self.run_detect.set_sensitive(False)
        self.run_detect.connect("clicked", controller.run_marker_detect)
        hbox.pack_start(self.run_detect, False, False, PADDING_SMALL)
        self.pack_start(hbox, False, False, PADDING_SMALL)

    def set_new_image(self):

        sm = self._specific_model

        if sm['im-path'] is not None:
            self.im_path.set_text(sm['im-path'])
            self.fixture_image.load_from_path()
        else:
            self.fixture_image.load_from_array()

    def set_markers(self):

        self.fixture_image.set_marker_overlays()

    def check_allow_marker_detection(self):

        sm = self._specific_model

        self.run_detect.set_sensitive(sm['im'] is not None and
            sm['markers'] >= 3)
        

class Fixture_Segmentation_Top(Top):

    def __init__(self, controller, model, specific_model):

        super(Fixture_Segmentation_Top, self).__init__(controller, model)
        self._specific_model = specific_model

        self._next_button = Top_Next_Button(controller, model, specific_model,
            model['fixture-segmentation-next'], controller.set_view_stage, 
            'save')
        self.set_allow_next(False)

        self.pack_end(self._next_button, False, False, PADDING_SMALL)
        self.show_all()

    def set_allow_next(self, val):

        self._next_button.set_sensitive(val)


class Fixture_Segmentation_Stage(gtk.VBox):

    def __init__(self, controller, model, specific_model):

        super(Fixture_Segmentation_Stage, self).__init__(0, False)
        self._controller = controller
        self._model = model
        self._specific_model = specific_model

        #TITLE
        label = gtk.Label()
        label.set_markup(model['fixture-segmentation-title'])
        self.pack_start(label, False, False, PADDING_SMALL)

        #MAKING TWO VBOXES
        hbox = gtk.HBox(False, 0)
        self.pack_start(hbox, True, True, PADDING_SMALL)
        left_side = gtk.VBox(False, 0)
        right_side = gtk.VBox(False, 0)
        hbox.pack_start(left_side, True, True, PADDING_SMALL)
        hbox.pack_end(right_side, False, False, PADDING_SMALL)

        #IMAGE DISPLAY - LEFT
        fixture_callbacks={'button_press_event': controller.mouse_press,
            'button_release_event': controller.mouse_release,
            'motion_notify_event': controller.mouse_move}
        self.fixture_image = Fixture_Image(self._specific_model,
            event_callbacks=fixture_callbacks)
        self.fixture_image.set_marker_overlays()
        left_side.pack_start(self.fixture_image.get_canvas(), True, True,
            PADDING_SMALL)

        #SEGMENTATION SETTINGS - RIGHT

        ##Gray Scale
        self.has_grayscale = gtk.CheckButton(
            label=model['fixture-segmentation-gs'])
        self.has_grayscale.set_active(specific_model['grayscale-exists'])
        self.has_grayscale.connect("clicked", controller.toggle_grayscale)
        right_side.pack_start(self.has_grayscale, False, False, PADDING_SMALL)

        ##Number of plates
        hbox = gtk.HBox(False, 0)
        label = gtk.Label()
        label.set_text(model['fixture-segmentation-plates'])
        hbox.pack_start(label, False, False, PADDING_SMALL)
        self.number_of_plates = gtk.Entry(1)
        self.number_of_plates.connect("changed", controller.set_number_of_plates)
        hbox.pack_start(self.number_of_plates, False, False, PADDING_SMALL)
        right_side.pack_start(hbox, False, False, PADDING_SMALL)

        ##What area are you working with
        self.segments = gtk.ListStore(str, str, str)
        self.treeview = gtk.TreeView(self.segments)
        self.treeview.connect('key_press_event',
            controller.handle_keypress)
        tv_cell = gtk.CellRendererText()
        tv_column = gtk.TreeViewColumn(
            model['fixture-segmentation-column-header-segment'],
            tv_cell, text=0)
        tv_cell = gtk.CellRendererText()
        self.treeview.append_column(tv_column)
        tv_column = gtk.TreeViewColumn(
            model['fixture-segmentation-column-header-ok'],
            tv_cell, text=1)
        self.treeview.append_column(tv_column)
        self.selection = self.treeview.get_selection()
        self.selection_signal = self.selection.connect('changed', 
            controller.set_active_segment)
        right_side.pack_start(self.treeview, True, True, PADDING_SMALL)
        
        self.set_segments_in_list()
        self.fixture_image.set_plate_overlays()

    def set_segments_in_list(self):

        sm = self._specific_model
        m = self._model

        store = self.segments

        #Grayscale
        if sm['grayscale-exists']:

            if not(len(store) > 0 and store[0][2] == 'G'):

                store.insert(0, (m['fixture-segmentation-grayscale'], 
                    m['fixture-segmentation-nok'], 'G'))

        else:

            for r in store:

                label, ok_nok, segment_type = r
                if segment_type == 'G':

                    store.remove(r.iter)

        #Plates
        plates = len(sm['plate-coords'])
        found_plates = 0
        for r in store:

            label, ok_nok, segment_type = r 
            
            if segment_type[0] == 'P':

                found_plates += 1

            if found_plates > plates:

                store.remove(r.iter)

        if found_plates < plates:

            for plate_index in xrange(found_plates + 1, plates + 1):

                store.append(
                    (m['fixture-segmentation-plate'].format(plate_index),
                    m['fixture-segmentation-nok'], 'P{0}'.format(plate_index)))

        self.set_ok_nok()

    def set_ok_nok(self):

        store = self.segments
        m =  self._model
        sm = self._specific_model
        all_rows = list()

        for r in store:

            label, ok_nok, segment_type = r

            if segment_type == 'G':
                if len(sm['grayscale-coords']) == 2 and \
                        sm['grayscale-sources'] is not None and \
                        None not in sm['grayscale-coords']:

                    row_ok = True

                else:

                    row_ok = False
            else:

                try:
                    plate = int(segment_type[-1]) - 1

                except:

                    plate = None

                if plate is not None and sm['plate-coords'][plate] is not None \
                        and None not in sm['plate-coords'][plate]:

                    row_ok = True

                else:

                    row_ok = False

            all_rows.append(row_ok)

            if row_ok:
                r[1] = m['fixture-segmentation-ok']
            else:
                r[1] = m['fixture-segmentation-nok']


        self._controller.set_allow_save(sum(all_rows) == len(all_rows))

    def update_segment(self, segment_name, scale=1.0):

        if segment_name == 'G':
           self.fixture_image.set_grayscale_overlay(scale=scale)
        else:
            try:
                plate = int(segment_name[-1])
            except:
                plate = None

            if plate is not None:
                self.fixture_image.set_plate_overlay(plate,
                    scale=scale)
 
        self.set_ok_nok()

    def draw_active_segment(self, scale=1.0):

        self.fixture_image.set_active_overlay(scale=scale)
        self.set_ok_nok()


class Fixture_Save_Top(Top):

    def __init__(self, controller, model, specific_model):

        super(Fixture_Save_Top, self).__init__(controller, model)
        self._specific_model = specific_model

        self.show_all()


class Fixture_Save_Stage(gtk.VBox):

    def __init__(self, controller, model, specific_model):

        super(Fixture_Save_Stage, self).__init__(0, False)
        self._controller = controller
        self._model = model
        self._specific_model = specific_model

        label = gtk.Label()
        label.set_markup(model['fixture-save-title'])
        self.pack_start(label, True, True, PADDING_SMALL)

        self.show_all()

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
        image_size=(200, 300)

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
        self.image_ax.imshow(model['im'], cmap=plt.cm.Greys_r)
        self.clear_overlays()
        self.image_fig.canvas.draw()

    def load_from_path(self):

        model = self._model
        im = plt.imread(model['im-path'])
        model['im'] = im
        if im is not None:
            self.load_from_array()
        else:
            model['im-path'] = None

    def set_not_loaded_text(self): 

        model = self._model

        self._set_text(text=model['im-not-loaded'], x=0.5, y=0.5,
            overlay_key='no-im')

        self.image_fig.canvas.draw()

    def _set_text(self, text, x, y, overlay_key, alpha=0.75,
            color='#004400', scale=1.0):

        if overlay_key in self._im_overlays.keys():

            t = self._im_overlays[overlay_key]
            t.set_x(x * scale)
            t.set_y(y * scale)
            t.set_text(text)

        else:

            self._im_overlays[overlay_key] = plt_text.Text(
                x=x * scale, y=y * scale,
                text=text, alpha=alpha,
                horizontalalignment='center', family='serif',
                verticalalignment='center', size='large',
                weight='bold', color=color)

            self.image_ax.add_artist(self._im_overlays[overlay_key])

    def _set_rect(self, coords, overlay_key, color='#228822', lw=2,
            alpha=0.5, scale=1.0):

        """
        x, y = map(min, zip(*coords))
        w, h = [a-b for a,b in zip(map(sum, zip(*coords)), (x,y))]
        """
        x, y = coords[0]
        w = coords[1][0] - x
        h = coords[1][1] - y

        if overlay_key in self._im_overlays.keys():

            rect = self._im_overlays[overlay_key]
            rect.set_xy((x * scale, y * scale))
            rect.set_width(w * scale)
            rect.set_height(h * scale)

        else:

            rect = plt_patches.Rectangle((x * scale, y * scale),
                w * scale, h * scale, color=color,
                lw=lw, alpha=alpha, fill=False)
            rect.get_axes()
            rect.get_transform()
            self.image_ax.add_patch(rect)
            self._im_overlays[overlay_key] = rect

    def _set_circle(self, x, y, overlay_key, alpha=0.75,
            color='#771100', radius=136, scale=1.0):

        if overlay_key in self._im_overlays.keys():

            circ = self._im_overlays[overlay_key]
            #circ.set_xy()
            circ.set_radius(radius)
            circ.set_edgecolor(color)
            circ.set_alpha(alpha)

        else:

            circ = plt_patches.Circle((x*scale, y*scale), lw=2,
                color=color, radius=radius, alpha=alpha, fill=False)
            circ.get_axes()
            circ.get_transform()
            self.image_ax.add_patch(circ)
            self._im_overlays[overlay_key] = circ

    def clear_overlays(self):

        for overlay in self._im_overlays:

            self._im_overlays[overlay].remove()

        self._im_overlays = dict()
        self.image_fig.canvas.draw()

    def clear_overlay(self, overlay):

        if overlay in self._im_overlays:
            self._im_overlays[overlay].remove()
            del self._im_overlays[overlay]

    def clear_overlay_markers(self):

        model = self._model

        for m in xrange(len(model['marker-positions'])):

            overlay = "marker_{0}".format(m)
            self.clear_overlay(overlay)

        self.image_fig.canvas.draw()

    def clear_plate_overlay(self, plate_index):

        plate_patch_overlay = "plate_{0}_rect".format(plate_index)
        plate_text_overlay = "plate_{0}_text".format(plate_index)

        self.clear_overlay(plate_patch_overlay)
        self.clear_overlay(plate_text_overlay)

        self.image_fig.canvas.draw()
        
    def clear_grayscale_overlay(self):

        self.clear_overlay("grayscale_text")
        self.clear_overlay("grayscale_rect")
        self.image_fig.canvas.draw()

    def set_marker_overlays(self, scale=1.0):

        model = self._model

        for i, (x, y) in enumerate(model['marker-positions']):

            self._set_circle(x*model['im-original-scale'], 
                y*model['im-original-scale'], 'marker_{0}'.format(i),
                alpha=0.5,
                radius=int(136.0*model['im-original-scale']),
                scale=scale)

        self.image_fig.canvas.draw()

    def set_plate_overlay(self, plate_index, plate=None, scale=1.0):

        plate_patch_overlay = "plate_{0}_rect".format(plate_index)
        plate_text_overlay = "plate_{0}_text".format(plate_index)

        model = self._model

        if plate is None:
            plate = model['plate-coords'][plate_index - 1]

        if plate is not None and None not in plate:
            self._set_segment_overlay(str(plate_index), plate,
                plate_text_overlay, plate_patch_overlay, scale=scale)
        else:
            self.clear_plate_overlay(plate_index)

    def set_plate_overlays(self, scale=1.0):

        model = self._model
        for i, p in enumerate(model['plate-coords']):
            self.set_plate_overlay(i + 1, p, scale=scale)

    def set_grayscale_overlay(self, coords=None, scale=1.0):

        model = self._model
        if coords is None:
            coords = model['grayscale-coords']

        if coords is not None and None not in coords and len(coords) > 0:
            self._set_segment_overlay(model['grayscale-image-text'],
                coords, 'grayscale_text', 'grayscale_rect', scale=scale)
        else:
            self.clear_grayscale_overlay()

    def _set_segment_overlay(self, segment_text, coords, segment_text_key,
            segment_rect_key, scale=1.0):

        center_x, center_y = [p/2.0 for p in map(sum, zip(*coords))]

        self._set_text(segment_text, center_x, center_y,
            segment_text_key, alpha=0.5, scale=scale)
        
        self._set_rect(coords, segment_rect_key, scale=scale)

        self.image_fig.canvas.draw()

    def set_active_overlay(self, scale=1.0):

        model = self._model

        coords = (model['active-source'], model['active-target'])
        plate_index = model['active-segment'][-1]

        if plate_index == 'G':

            self.set_grayscale_overlay(coords=coords, scale=scale)

        else:
            try:
                plate_index = int(plate_index)
            except:
                plate_index = None

            if plate_index is not None:
                self.set_plate_overlay(plate_index, plate=coords, scale=scale) 

    def get_canvas(self):

        return self.image_fig.canvas
