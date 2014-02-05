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
import numpy as np
import types
from matplotlib import pyplot as plt
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas

#
# INTERNAL DEPENDENCIES
#

#import src.resource_scanner as resource_scanner
import scanomatic.imageAnalysis.grayscale as grayscale
import scanomatic.io.logger as logger

#
# STATIC GLOBALS
#

PADDING_LARGE = 10
PADDING_MEDIUM = 4
PADDING_SMALL = 2
PADDING_NONE = 0

#
# FUNCTIONS
#


def select_dir(title, start_in=None):

    d = gtk.FileChooserDialog(
        title=title,
        action=gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER,
        buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                 gtk.STOCK_APPLY, gtk.RESPONSE_APPLY))

    if start_in is not None:
        d.set_current_folder(start_in)

    res = d.run()
    file_list = d.get_filename()
    d.destroy()

    if res == gtk.RESPONSE_APPLY:

        return file_list

    else:

        return None


def select_file(title, multiple_files=False, file_filter=None,
                start_in=None):

    d = gtk.FileChooserDialog(
        title=title,
        action=gtk.FILE_CHOOSER_ACTION_OPEN,
        buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                 gtk.STOCK_APPLY, gtk.RESPONSE_APPLY))

    d.set_select_multiple(multiple_files)

    if start_in is not None:
        d.set_current_folder(start_in)

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


def save_file(title, multiple_files=False, file_filter=None,
              start_in=None):

    d = gtk.FileChooserDialog(
        title=title,
        action=gtk.FILE_CHOOSER_ACTION_SAVE,
        buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                 gtk.STOCK_APPLY, gtk.RESPONSE_APPLY))

    d.set_select_multiple(multiple_files)

    if start_in is not None:
        d.set_current_folder(start_in)

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


def overwrite(text, file_name, window):

    dialog = gtk.MessageDialog(
        window, gtk.DIALOG_DESTROY_WITH_PARENT,
        gtk.MESSAGE_WARNING, gtk.BUTTONS_NONE,
        text.format(file_name))

    dialog.add_button(gtk.STOCK_NO, False)
    dialog.add_button(gtk.STOCK_YES, True)

    dialog.show_all()

    resp = dialog.run()

    dialog.destroy()

    return resp


def dialog(window, text, d_type="info", yn_buttons=False):

    d_types = {'info': gtk.MESSAGE_INFO, 'error': gtk.MESSAGE_ERROR,
               'warning': gtk.MESSAGE_WARNING,
               'question': gtk.MESSAGE_QUESTION,
               'other': gtk.MESSAGE_OTHER}

    if d_type in d_types.keys():

        d_type = d_types[d_type]

    else:

        d_type = gtk.MESSAGE_INFO

    d = gtk.MessageDialog(window, gtk.DIALOG_DESTROY_WITH_PARENT,
                          gtk.MESSAGE_INFO, gtk.BUTTONS_NONE,
                          text)

    if yn_buttons:

        d.add_button(gtk.STOCK_YES, True)
        d.add_button(gtk.STOCK_NO, False)

    else:

        d.add_button(gtk.STOCK_OK, -1)

    result = d.run()

    d.destroy()

    return result


def claim_a_scanner_dialog(window, text, image_path, scanners):

    scanners.update()
    scanner_names = scanners.get_names()

    dialog = gtk.MessageDialog(window, gtk.DIALOG_DESTROY_WITH_PARENT,
                               gtk.MESSAGE_INFO, gtk.BUTTONS_NONE,
                               text)

    for i, s in enumerate(scanner_names):
        dialog.add_button(s, i)

    dialog.add_button(gtk.STOCK_CANCEL, -1)

    img = gtk.Image()
    img.set_from_file(image_path)
    dialog.set_image(img)
    dialog.show_all()

    resp = dialog.run()

    dialog.destroy()

    if resp >= 0:
        return scanner_names[resp]
    else:
        return None


def get_grayscale_combo():

    def _get_text(self):
        row = self.get_active()
        if row >= 0:
            m = self.get_model()
            return m[row][0]
        else:
            return None

    def _set_activeGrayscale(self, gsName):

        m = self.get_model()
        for rowIndex, row in enumerate(m):
            if row[0] == gsName:
                self.set_active(rowIndex)
                return

        self.set_active(-1)

    def _addGrayscale(self, gsName):

        m = self.get_model()
        for rowIndex, row in enumerate(m):
            if row[0] == gsName:
                self.set_active(rowIndex)
                return

        m.append((gsName,))
        self.set_active(len(m) - 1)

    gs = gtk.combo_box_new_text()
    gs.get_text = types.MethodType(_get_text, gs)
    gs.set_activeGrayscale = types.MethodType(_set_activeGrayscale, gs)
    gs.addGrayscale = types.MethodType(_addGrayscale, gs)

    model = gs.get_model()

    for key in grayscale.GRAYSCALES:
        model.append([key])

    if len(grayscale.GRAYSCALES) > 0:
        gs.set_active(0)

    return gs


def get_fixtures_combo():

    def _get_text(self):
        row = self.get_active()
        if row >= 0:
            m = self.get_model()
            return m[row][0]
        else:
            return None

    fixture = gtk.combo_box_new_text()
    fixture.get_text = types.MethodType(_get_text, fixture)

    return fixture


def set_fixtures_combo(widget, fixtures, default_text=None):

    fixture_names = fixtures.get_names()
    widget_model = widget.get_model()
    active_row = widget.get_active()

    if active_row >= 0:
        active_option = widget_model[active_row][0]
    else:
        active_option = None

    if len(widget_model) == 0 and default_text is not None:
        #widget.append_text(default_text)
        widget_model.append([default_text])
        fixture_names.append(default_text)

    for row in widget_model:
        if row[0] not in fixture_names:
            widget_model.remove(row.iter)
        else:
            fixture_names = [fix for fix in fixture_names if fix != row[0]]

    for f in sorted(fixture_names):
        #widget.append_text(f)
        widget_model.append([f])

    if active_option is not None:
        for i, row in enumerate(widget_model):
            if row[0] == active_option:
                widget.set_active(i)
                break


def set_fixtures_active(widget, name=None):

    for i, row in enumerate(widget.get_model()):

        if row[0] == name:

            widget.set_active(i)
            return True

    widget.set_active(-1)
    return False
#
# CLASSES
#


class Fixture_Drawing(gtk.DrawingArea):

    HEIGHT = 1
    WIDTH = 0
    PADDING = 0.01
    VIEW_STATE = ('Image', 'Scanner')
    BACKGROUND = (0.75, 0.75, 0.95, 0.9)
    ERROR_BACKGROUND = (0.9, 0.1, 0.1, 1.0)
    ERROR_STROKE = (1, 1, 1, 1)

    def __init__(self, fixture, width=None, height=None,
                 scanner_view=False):

        super(Fixture_Drawing, self).__init__()
        self.connect("expose_event", self.expose)

        self._fixture = fixture
        self._scanner_view = scanner_view

        self._plate_fill_rgba = tuple(np.array((242, 122, 17, 255)) / 255.0)
        self._gs_fill_rgba = (0.5, 0.5, 0.5, 0.9)
        self._plate_stroke_rgba = (1, 1, 1, 0.5)
        self._gs_stroke_rgba = (1, 1, 1, 0.5)
        self._text_rgba = (1, 1, 1, 0.9)
        self._good_fixture = None

        self._data_width = 0
        self._data_height = 0

        self._logger = logger.Logger("Fixture Drawing")

        self._set_data()

        if width is not None and height is not None:
            self.set_size_request(width, height)

    def get_fixture_status(self):

        return self._good_fixture

    def _set_data(self):

        self._plates = np.array(self._fixture.get_plates('fixture'))
        self._grayscale = np.array(self._fixture['fixture']['grayscale_area'])

        try:
            self._data_height = max(self._plates[:, :, self.HEIGHT].max(),
                                    self._grayscale[:, self.HEIGHT].max())

            self._data_width = max(self._plates[:, :, self.WIDTH].max(),
                                   self._grayscale[:, self.WIDTH].max())
        except IndexError:
            self._good_fixture = False
            return

        print "SCANNER VIEW", self._scanner_view
        if self._scanner_view:
            self._flipflip(xflip=True, yflip=True)
        else:
            self._flipflip(xflip=False, yflip=True)

        self._good_fixture = True

    def _flipflip(self, xflip=True, yflip=False):
        """This doesn't flip the Y-axis when toggling since the 'normal' way to
        display an image is with inverted y-axis, it does it the first time when
        loading the image."""

        if xflip:
            self._grayscale[:, self.WIDTH] = \
                self._data_width - self._grayscale[:, self.WIDTH]
            self._plates[:, :, self.WIDTH] = \
                self._data_width - self._plates[:, :, self.WIDTH]
        if yflip:
            self._grayscale[:, self.HEIGHT] = \
                self._data_height - self._grayscale[:, self.HEIGHT]
            self._plates[:, :, self.HEIGHT] = \
                self._data_height - self._plates[:, :, self.HEIGHT]

    def _get_pos(self, d1, d2):

        new_pos = (d1 / self._data_width * self._cr_active_w +
                   self._cr_padding_w, d2 / self._data_height *
                   self._cr_active_h + self._cr_padding_h)

        return new_pos

    def _draw_bg(self, cr, w, h):

        cr.set_source_rgba(*self.BACKGROUND)
        cr.rectangle(0, 0, w, h)
        cr.fill()

    def _draw_error(self, cr, w, h):

        cr.set_source_rgba(*self.ERROR_BACKGROUND)
        cr.rectangle(0, 0, w, h)
        cr.fill()

        cr.set_source_rgba(*self.ERROR_STROKE)
        cr.move_to(0, 0)
        cr.line_to(w, h)
        cr.stroke()

        cr.move_to(w, 0)
        cr.line_to(0, h)
        cr.stroke()

    def _draw_rect(self, cr, positions, stroke_rgba=None, stroke_width=0.5, fill_rgba=None):

        #self._logger.info("Will draw {0}".format(positions))

        cr.move_to(*self._get_pos(*positions[0]))

        cr.line_to(*self._get_pos(*positions.diagonal()))
        cr.line_to(*self._get_pos(*positions[1]))
        cr.line_to(*self._get_pos(*positions[::-1].diagonal()))

        cr.close_path()

        if stroke_rgba is not None:
            cr.set_line_width(stroke_width)
            cr.set_source_rgba(*stroke_rgba)
            if fill_rgba is not None:
                cr.stroke_preserve()
            else:
                cr.stroke()

        if fill_rgba is not None:
            cr.set_source_rgba(*fill_rgba)
            cr.fill()

    def _draw_text(self, cr, positions, text, text_rgba=None,
                   fsize=20):

        if text_rgba is None:
            text_rgba = self._text_rgba

        rect_x, rect_y = self._get_pos(*positions.mean(axis=0))
        text = str(text)

        cr.set_font_size(fsize)

        fascent, fdescent, fheight, fxadvance, fyadvance = cr.font_extents()
        xbearing, ybearing, width, height, xadvance, yadvance = (
            cr.text_extents(text))

        cr.move_to(rect_x - xbearing - width / 2,
                   rect_y - fdescent + fheight / 2)

        cr.set_source_rgba(*text_rgba)
        cr.show_text(text)

    def toggle_view_state(self):

        self._flipflip()
        self._scanner_view = self._scanner_view is False
        self.window.clear()
        self.expose(self, None)
        return self.get_view_state()

    def set_view_state(self, wstate):

        wstate = wstate.capitalize()

        try:
            scanner_view = self.VIEW_STATE.index(wstate)
        except:
            self._logger.error(
                'Unknown view state {0}, accepted are {1}'.format(
                wstate, self.VIEW_STATE))
            return False

        if scanner_view != self._scanner_view:
            self.toggle_view_state()
        else:
            self._logger.warning('Already in state {0}'.format(wstate))

        return True

    def get_view_state(self):

        return self.VIEW_STATE[self._scanner_view]

    def get_other_state(self):

        return self.VIEW_STATE[self._scanner_view is False]

    def expose(self, widget, event):

        cr = widget.window.cairo_create()
        rect = self.get_allocation()

        w = rect.width
        h = rect.height

        #CALCULATE CURRENT PARAMETERS
        self._cr_padding_w = w * self.PADDING
        self._cr_padding_h = h * self.PADDING
        self._cr_active_w = w - 2 * self._cr_padding_w
        self._cr_active_h = h - 2 * self._cr_padding_h

        if self._grayscale is None or len(self._plates) == 0:
            self._draw_error(cr, w, h)
            return

        #DRAWING
        self._draw_bg(cr, w, h)

        self._draw_rect(cr, self._grayscale, stroke_rgba=self._gs_stroke_rgba,
                        fill_rgba=self._gs_fill_rgba)

        for i, plate in enumerate(self._plates):

            self._draw_rect(cr, plate, stroke_rgba=self._plate_stroke_rgba,
                            fill_rgba=self._plate_fill_rgba)

            self._draw_text(cr, plate, i + 1, fsize=h / 10)


class Grayscale_Display(object):

    GS_MARKER = np.array([[-1, -0.5], [1, -0.5], [1, 0.5], [-1, 0.5],
                          [-1, -0.5]])

    def __init__(self, gtkBox=gtk.VBox, imFigSize=(150, 250),
                 graphFigSize=(150, 250), dpi=150):

        self._gtkBox = gtkBox(False, 0)

        self._imFig = plt.Figure(figsize=imFigSize, dpi=dpi)
        self._imAx = self._imFig.add_subplot(111)
        FigureCanvas(self._imFig)
        self._imAx.get_xaxis().set_visible(False)
        self._imAx.get_yaxis().set_visible(False)

        self._graphFig = plt.Figure(figsize=graphFigSize, dpi=dpi)
        self._graphAx = self._graphFig.add_subplot(111)
        FigureCanvas(self._graphFig)
        self._graphAx.get_xaxis().set_visible(False)
        self._graphAx.get_yaxis().set_visible(False)

        self._gtkBox.pack_start(self._imFig.canvas, True, True, PADDING_SMALL)
        self._gtkBox.pack_start(self._graphFig.canvas, True, True,
                                PADDING_SMALL)

    @property
    def widget(self):

        return self._gtkBox

    def update(self):

        self._imFig.canvas.draw()
        self._graphFig.canvas.draw()

    def set(self, gsAnalysis=None, sourceValues=None, targetValues=None,
            im=None):

        self._imAx.cla()
        self._graphAx.cla()

        if gsAnalysis is None:
            self.update()
            return

        if im is None:
            im = gsAnalysis.image

        self._imAx.imshow(im,
                          interpolation='nearest', cmap=plt.cm.gray)

        Xposition, Ypositions = gsAnalysis.get_sampling_positions()
        if sourceValues is None:
            sourceValues = gsAnalysis.get_source_values()
        if targetValues is None:
            targetValues = gsAnalysis.get_target_values()

        if Ypositions is not None:
            X = np.ones(len(Ypositions)) * Xposition
            self._imAx.plot(X, Ypositions, marker=self.GS_MARKER, ls='None',
                            mew=1, mec='b', mfc='None', ms=2)

            self._graphAx.plot(
                sourceValues,
                targetValues,
                marker='o', ls='-', color='b', mec='b', mfc='None', mew=1,
                ms=3)

            self._graphAx.set_xlabel("Pixel value", size='xx-small')
            self._graphAx.set_ylabel("Target value", size='xx-small')
            self._graphAx.set_xlim(0, 255)

        self.update()


class Start_Button(gtk.Button):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model

        super(Start_Button, self).__init__(stock=gtk.STOCK_EXECUTE)

        al = self.get_children()[0]
        hbox = al.get_children()[0]
        im, l = hbox.get_children()

        l.set_text(model['start-text'])

        self.connect("clicked", controller.start)


class Pinning(gtk.VBox):

    def __init__(self, controller, model, project_veiw,
                 plate_number, pinning=None):

            self._model = model
            self._controller = controller
            self._project_view = project_veiw

            super(Pinning, self).__init__()

            label = gtk.Label(model['plate-label'].format(plate_number))

            self.pack_start(label, False, False, PADDING_SMALL)

            if pinning is None:

                pinning = model['pinning-default']

            self.dropbox = gtk.combo_box_new_text()

            def_key = 0
            ref_matrices = model['pinning-matrices-reversed']
            for i, m in enumerate(sorted(model['pinning-matrices'].keys())):

                self.dropbox.append_text(m)

                if pinning in ref_matrices and ref_matrices[pinning] == m:
                    def_key = i

            self.dropbox.set_active(def_key)

            self.dropbox_signal = self.dropbox.connect(
                "changed", self._controller.set_pinning, plate_number)

            self.pack_start(self.dropbox, True, True, PADDING_SMALL)

    def set_sensitive(self, val):

        if val is None:
            val = False

        self.dropbox.set_sensitive(val)

    def set_pinning(self, pinning):

        orig_key = self.dropbox.get_active()
        new_key = -2
        pinning_text = self._model['pinning-matrices-reversed'][pinning]

        for i, m in enumerate(sorted(self._model['pinning-matrices'].keys())):

            if pinning_text == m:
                new_key = i

        if new_key != orig_key:

            self.dropbox.handler_block(self.dropbox_signal)
            self.dropbox.set_active(new_key)
            self.dropbox.handler_unblock(self.dropbox_signal)


class Page(gtk.VBox):

    def __init__(self, controller, model, top=None, stage=None):

        super(Page, self).__init__(False, 0)

        self._controller = controller
        self._model = model

        self.set_top(top)
        self.set_stage(stage)

    def _remove_child(self, pos=0):

        children = self.get_children()

        if len(children) - pos > 0:

            self.remove(children[pos])

    def get_controller(self):

        return self._controller

    def set_controller(self, c):

        self._controller = c

    def set_top(self, widget=None):

        if widget is None:

            widget = self._default_top()

        self._top = widget
        self._remove_child(pos=0)
        self.pack_start(widget, False, True, PADDING_LARGE)

    def _default_top(self):

        return gtk.VBox()

    def get_top(self):

        return self._top

    def set_stage(self, widget=None):

        if widget is None:

            widget = self._default_stage()

        self._stage = widget
        self._remove_child(pos=1)
        self.pack_end(widget, True, True, 10)
        widget.show_all()

    def _default_stage(self):

        return gtk.VBox()

    def get_stage(self):

        return self._stage


class Top(gtk.HBox):

    def __init__(self, controller, model):

        super(Top, self).__init__(False, 0)

        self._controller = controller
        self._model = model

    def pack_back_button(self, label, callback, message):

        button = gtk.Button(stock=gtk.STOCK_GO_BACK)
        al = button.get_children()[0]
        hbox = al.get_children()[0]
        im, l = hbox.get_children()
        l.set_text(label)
        button.connect("clicked", callback, message)
        self.pack_start(button, False, False, PADDING_SMALL)
        button.show()
        return button


class Top_Next_Button(gtk.Button):

    def __init__(self, controller, model, specific_model, label_text,
                 callback, stage_signal_text):

        self._controller = controller
        self._model = model
        self._specific_model = specific_model

        super(Top_Next_Button, self).__init__(stock=gtk.STOCK_GO_FORWARD)

        al = self.get_children()[0]
        hbox = al.get_children()[0]
        im, l = hbox.get_children()

        l.set_text(label_text)
        hbox.remove(im)
        hbox.remove(l)
        hbox.pack_start(l, False, False, PADDING_SMALL)
        hbox.pack_end(im, False, False, PADDING_SMALL)

        self.connect("clicked", callback,
                     stage_signal_text, specific_model)
