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

from matplotlib import pyplot as plt
#from matplotlib import patches as plt_patches
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas

#
# INTERNAL DEPENDENCIES
#

import scanomatic.dataProcessing.phenotyper as phenotyper

#import scanomatic.gui.generic.view_generic as view_generic

#
# CLASSES
#


class Main_Window(gtk.Window):

    def __init__(self, controller, model):

        super(Main_Window, self).__init__()

        self.set_default_size(800, 600)
        self.set_border_width(5)
        self.set_title(model['app-title'])
        self.move(10, 10)

        self._controller = controller
        self._model = model
        self.add(QC_Stage(model, controller))

        self.connect("delete_event", self._win_close_event)

    def _win_close_event(self, widget, *args, **kwargs):

        v = (self._model['unsaved'] and self._ask_quit() or False)
        if not v:
            gtk.main_quit()
        return v

    def _ask_quit(self):

        dialog = gtk.MessageDialog(
            flags=gtk.DIALOG_DESTROY_WITH_PARENT,
            type=gtk.MESSAGE_WARNING,
            buttons=gtk.BUTTONS_YES_NO,
            message_format=self._model["quit-unsaved"])

        ret = dialog.run() != gtk.RESPONSE_YES
        dialog.destroy()
        return ret

    def get_stage(self):

        return self.child


class QC_Stage(gtk.VBox):

    COLOR_THiS = 0
    COLOR_ALL = 1
    COLOR_FIXED = 2

    def __init__(self, model, controller):

        super(QC_Stage, self).__init__(False, spacing=2)

        self._model = model
        self._controller = controller

        self._widgets_require_data = SensitivityGroup()
        self._widgets_require_fixed_color = SensitivityGroup()
        self._widgets_require_selection = SensitivityGroup()
        self._widgets_require_removed = SensitivityGroup()
        self._widgets_require_subplates = SensitivityGroup()
        self._widgets_require_references = SensitivityGroup()
        self._widgets_require_norm = SensitivityGroup()

        self._widgets_require_data.addDependetGroup(
            self._widgets_require_fixed_color)
        self._widgets_require_data.addDependetGroup(
            self._widgets_require_selection)
        self._widgets_require_data.addDependetGroup(
            self._widgets_require_removed)
        self._widgets_require_data.addDependetGroup(
            self._widgets_require_references)
        self._widgets_require_data.addDependetGroup(
            self._widgets_require_subplates)
        self._widgets_require_subplates.addDependetGroup(
            self._widgets_require_norm)

        #
        # LOAD BUTTONS
        #

        hbox = gtk.HBox(False, spacing=2)

        self._buttonLoadData = gtk.Button(self._model['button-load-data'])
        self._buttonLoadData.connect("clicked", self._loadPlate)
        self._buttonLoadMetaData = gtk.Button(self._model['button-load-meta'])
        self._buttonLoadMetaData.connect("clicked", self._loadMetaData)

        hbox.pack_start(self._buttonLoadData, expand=False, fill=False)
        hbox.pack_start(self._buttonLoadMetaData, expand=False, fill=False)
        self.pack_start(hbox, expand=False, fill=False)

        #
        # PLATE SELECTOR
        #

        hbox = gtk.HBox(False, spacing=2)

        self._plateSelectionAdjustment = gtk.Adjustment(0, 0, 0, 1, 1, 0)
        self._plateSelector = gtk.SpinButton(self._plateSelectionAdjustment,
                                             0, 0)
        self._plateSelectionAdjustment.connect("value_changed",
                                               self._loadPlate)
        self._plateSelector.set_wrap(True)
        self._plateSelector.set_snap_to_ticks(True)

        self._widgets_require_data.append(self._plateSelector)

        hbox.pack_start(gtk.Label(self._model['label-plate']), expand=False,
                        fill=False)
        hbox.pack_start(self._plateSelector, expand=False, fill=False)

        #
        #PHENOTYPE SELECTOR
        #
        self._phenotypeSelector = gtk.combo_box_new_text()

        self._phenotypeName2Key = {}
        for k in sorted(phenotyper.Phenotyper.NAMES_OF_PHENOTYPES.keys()):

            n = phenotyper.Phenotyper.NAMES_OF_PHENOTYPES[k]
            self._phenotypeSelector.append_text(n)
            self._phenotypeName2Key[n] = k

        self._phenotypeSelector.connect(
            "changed", self._newPhenotype)

        self._widgets_require_data.add(self._phenotypeSelector)

        hbox.pack_start(self._phenotypeSelector, expand=False, fill=False)
        self.pack_start(hbox, expand=False, fill=False)

        #
        #HEATMAP
        #

        plate_figure = plt.Figure(figsize=(40, 40), dpi=150)
        plate_figure.add_axes()
        plate_figure_ax = plate_figure.gca()
        plate_figure_ax.set_axis_off()
        self._plate_image_canvas = FigureCanvas(plate_figure)

        self._plate_image_canvas.mpl_connect('button_press_event',
                                             self._mousePress)
        self._plate_image_canvas.mpl_connect('button_release_event',
                                             self._mouseRelease)
        self._plate_image_canvas.mpl_connect('motion_notify_event',
                                             self._mouseHover)

        self.set_events(gtk.gdk.KEY_PRESS_MASK | gtk.gdk.KEY_RELEASE_MASK)
        self.connect('key_press_event', self._pressKey)
        self.connect('key_release_event', self._releaseKey)

        self._HeatMapInfo = gtk.Label("")

        self._plateSaveImage = gtk.Button(self._model['plate-save'])
        self._plateSaveImage.connect("clicked", self._saveImage)
        self._widgets_require_data.append(self._plateSaveImage)

        hbox = gtk.HBox(False, spacing=2)
        self.pack_start(hbox, expand=True, fill=True)
        vbox = gtk.VBox(False, spacing=2)
        hbox.pack_start(vbox, expand=True, fill=True)
        vbox.pack_start(self._plate_image_canvas, expand=True, fill=True)
        hbox2 = gtk.HBox(False, spacing=2)
        hbox2.pack_start(self._plateSaveImage, expand=False, fill=False)
        hbox2.pack_start(self._HeatMapInfo, expand=True, fill=True)
        vbox.pack_start(hbox2, expand=False, fill=False)

        #
        # HEATMAP COLOR VALUES
        #

        plateActionVB = gtk.VBox(False, spacing=2)
        hbox.pack_start(plateActionVB, expand=False, fill=False)
        frame = gtk.Frame(self._model['colors'])
        vbox2 = gtk.VBox(False, spacing=2)
        frame.add(vbox2)
        plateActionVB.pack_start(frame, expand=False, fill=False)

        self._colorsThisPlate = gtk.RadioButton(
            group=None, label=self._model['color-one-plate'])
        self._colorsAllPlate = gtk.RadioButton(
            group=self._colorsThisPlate, label=self._model['color-all-plates'])
        self._colorsFixed = gtk.RadioButton(
            group=self._colorsThisPlate, label=self._model['color-fixed'])
        self._colorMin = gtk.Entry()
        self._colorMin.set_size_request(0, -1)
        self._colorMax = gtk.Entry()
        self._colorMax.set_size_request(0, -1)
        self._colorFixUpdate = gtk.Button(self._model['color-fixed-update'])
        self._colorsThisPlate.set_active(True)
        self._colorsThisPlate.connect("toggled", self._setColors,
                                      self.COLOR_THiS)
        self._colorsAllPlate.connect("toggled", self._setColors,
                                     self.COLOR_ALL)
        self._colorsFixed.connect("toggled", self._setColors, self.COLOR_FIXED)
        self._colorFixUpdate.connect("clicked", self._setColors,
                                     self.COLOR_FIXED)

        self._widgets_require_data.append(self._colorsThisPlate)
        self._widgets_require_data.append(self._colorsAllPlate)
        self._widgets_require_data.append(self._colorsFixed)

        self._widgets_require_fixed_color.append(self._colorMin)
        self._widgets_require_fixed_color.append(self._colorMax)
        self._widgets_require_fixed_color.append(self._colorFixUpdate)

        vbox2.pack_start(self._colorsThisPlate, expand=False, fill=False)
        vbox2.pack_start(self._colorsAllPlate, expand=False, fill=False)
        vbox2.pack_start(self._colorsFixed, expand=False, fill=False)
        hbox2 = gtk.HBox(False, spacing=2)
        vbox2.pack_start(hbox2, expand=False, fill=False)
        hbox2.pack_start(self._colorMin, expand=True, fill=True)
        hbox2.pack_start(gtk.Label("-"), expand=False, fill=False)
        hbox2.pack_start(self._colorMax, expand=True, fill=True)
        hbox2.pack_start(self._colorFixUpdate, expand=False, fill=False)

        #
        # Value Selection Ranges
        #

        frame = gtk.Frame(self._model['multi-select-phenotype'])
        plateActionVB.pack_start(frame, expand=False, fill=False)
        vbox2 = gtk.VBox(False, spacing=2)
        frame.add(vbox2)

        self._lowerBoundAdjustment = gtk.Adjustment(0, 0, 1, 0.01, 0.01, 1)
        self._higherBoundAdjustment = gtk.Adjustment(0, 0, 1, 0.01, 0.01, 1)
        self._lowerBoundWidget = gtk.HScale(self._lowerBoundAdjustment)
        self._higherBoundWidget = gtk.HScale(self._higherBoundAdjustment)
        self._lowerBoundAdjustment.connect("value_changed",
                                           self._updateBounds)
        self._higherBoundAdjustment.connect("value_changed",
                                            self._updateBounds)

        self._updatingBounds = False
        #self._updateBounds()

        self._widgets_require_data.append(self._lowerBoundWidget)
        self._widgets_require_data.append(self._higherBoundWidget)

        hbox2 = gtk.HBox(False, spacing=2)
        vbox2.pack_start(hbox2, expand=False, fill=False)
        hbox2.pack_start(gtk.Label(self._model['multi-sel-lower']),
                         expand=False, fill=False)
        hbox2.pack_start(self._lowerBoundWidget, expand=True, fill=True)
        hbox2 = gtk.HBox(False, spacing=2)
        vbox2.pack_start(hbox2, expand=False, fill=False)
        hbox2.pack_start(gtk.Label(self._model['multi-sel-higher']),
                         expand=False, fill=False)
        hbox2.pack_start(self._higherBoundWidget, expand=True, fill=True)

        #
        # Subplate selection
        #

        self._subplate_0_0 = gtk.CheckButton(self._model['subplate-0-0'])
        self._subplate_0_1 = gtk.CheckButton(self._model['subplate-0-1'])
        self._subplate_1_0 = gtk.CheckButton(self._model['subplate-1-0'])
        self._subplate_1_1 = gtk.CheckButton(self._model['subplate-1-1'])

        self._subplate_0_0.connect("toggled", self._subplateSelect, (0, 0))
        self._subplate_0_1.connect("toggled", self._subplateSelect, (0, 1))
        self._subplate_1_0.connect("toggled", self._subplateSelect, (1, 0))
        self._subplate_1_1.connect("toggled", self._subplateSelect, (1, 1))

        self._widgets_require_data.add(self._subplate_1_1)
        self._widgets_require_data.add(self._subplate_1_0)
        self._widgets_require_data.add(self._subplate_0_1)
        self._widgets_require_data.add(self._subplate_0_0)

        frame = gtk.Frame(self._model['subplate-selection'])
        vbox2 = gtk.VBox(False, spacing=2)
        frame.add(vbox2)
        plateActionVB.pack_start(frame, expand=False, fill=False)
        hbox2 = gtk.HBox(False, spacing=2)
        vbox2.pack_start(hbox2, expand=False, fill=False)
        hbox2.pack_start(self._subplate_0_0, expand=False, fill=False)
        hbox2.pack_start(self._subplate_0_1, expand=False, fill=False)
        hbox2 = gtk.HBox(False, spacing=2)
        vbox2.pack_start(hbox2, expand=False, fill=False)
        hbox2.pack_start(self._subplate_1_0, expand=False, fill=False)
        hbox2.pack_start(self._subplate_1_1, expand=False, fill=False)

        #
        #Buttons Working on selected curves
        #

        self._buttonUnSelect = gtk.Button(self._model['unselect'])
        self._buttonRemoveCurvesPhenotype = gtk.Button(
            self._model['removeCurvesPhenotype'])
        self._buttonRemoveCurvesAllPhenotypes = gtk.Button(
            self._model['removeCurvesAllPhenotypes'])

        self._buttonUnSelect.connect("clicked", self._unselect)
        self._buttonRemoveCurvesPhenotype.connect("clicked",
                                                  self._removeCurvesPhenotype)
        self._buttonRemoveCurvesAllPhenotypes.connect(
            "clicked", self._removeCurvesAllPhenotype)

        self._widgets_require_selection.append(self._buttonUnSelect)
        self._widgets_require_selection.append(
            self._buttonRemoveCurvesPhenotype)
        self._widgets_require_selection.append(
            self._buttonRemoveCurvesAllPhenotypes)

        self._undoButton = gtk.Button(self._model['undo'])
        self._undoButton.connect("clicked", self._undoRemove)

        self._widgets_require_removed.append(self._undoButton)

        frame = gtk.Frame(self._model['selections-section'])
        vbox2 = gtk.VBox(False, spacing=2)
        frame.add(vbox2)
        plateActionVB.pack_start(frame, expand=False, fill=False)
        vbox2.pack_start(self._buttonUnSelect, expand=False, fill=False)
        vbox2.pack_start(gtk.HSeparator(), expand=False, fill=False, padding=4)
        vbox2.pack_start(self._buttonRemoveCurvesPhenotype, expand=False,
                         fill=False)
        vbox2.pack_start(self._buttonRemoveCurvesAllPhenotypes,
                         expand=False, fill=False)
        vbox2.pack_start(gtk.HSeparator(), expand=False, fill=False, padding=4)
        vbox2.pack_start(self._undoButton, expand=False, fill=False)

        #
        # Curve Display
        #

        self._curve_figure = plt.Figure(figsize=(40, 40), dpi=150)
        self._curve_figure.add_axes()
        self._curve_figure_ax = self._curve_figure.gca()
        self._curve_figure_ax.set_axis_off()
        self._curve_image_canvas = FigureCanvas(self._curve_figure)

        self._curveSaveImage = gtk.Button(self._model['curve-save'])
        self._curveSaveImage.connect("clicked", self._saveImage)

        self._widgets_require_selection.append(self._curveSaveImage)

        hbox = gtk.HBox(False, spacing=2)
        vbox = gtk.VBox(False, spacing=2)
        hbox.pack_start(vbox, expand=True, fill=True)
        self.pack_start(hbox, expand=True, fill=True)
        vbox.pack_start(self._curve_image_canvas, expand=True, fill=True)
        hbox2 = gtk.HBox(False, spacing=2)
        vbox.pack_start(hbox2, expand=False, fill=False)
        hbox2.pack_start(self._curveSaveImage, expand=False, fill=False)

        #
        # Normalize
        #

        self._setRefButton = gtk.Button(self._model['references'])
        self._setRefButton.connect("clicked", self._setReferences)

        self._widgets_require_subplates.append(self._setRefButton)

        self._normalize = gtk.Button(self._model['normalize'])
        self._normalize.connect("clicked", self._doNormalize)

        self._widgets_require_references.append(self._normalize)

        vbox = gtk.VBox(False, spacing=2)
        hbox.pack_start(vbox, expand=False, fill=False)
        frame = gtk.Frame(self._model['frame-normalize'])
        vbox.pack_start(frame, expand=False, fill=False)
        vbox2 = gtk.VBox(False, spacing=2)
        frame.add(vbox2)
        vbox2.pack_start(self._setRefButton, expand=False, fill=False)
        vbox2.pack_start(gtk.HSeparator(), expand=False, fill=False, padding=4)
        vbox2.pack_start(self._normalize, expand=False, fill=False)

        #
        # Save
        #

        self._overwriteAbsolute = gtk.Button(self._model['save-absolute'])
        self._overwriteAbsolute.connect("clicked", self._saveAbsolute)

        self._widgets_require_data.append(self._overwriteAbsolute)

        self._saveNormed = gtk.Button(self._model['save-relative'])
        self._saveNormed.connect("clicked", self._saveRelative)

        self._widgets_require_norm.append(self._saveNormed)

        self._curSelection = None
        self._multiSelecting = False

        vbox.pack_start(self._overwriteAbsolute, expand=False, fill=False,
                        padding=4)

        vbox.pack_start(self._saveNormed, expand=False, fill=False,
                        padding=4)

        self._widgets_require_data.sensitive = False

        self.show_all()

    @property
    def colorMin(self):

        try:
            val = float(self._colorMin.get_text())
        except:
            val = None
        return val

    @property
    def colorMax(self):

        try:
            val = float(self._colorMax.get_text())
        except:
            val = None
        return val

    def _subplateSelect(self, widget, subPlate):

        self._model['subplateSelected'][subPlate] = widget.get_active()
        self._controller.setSubPlateSelection(subPlate)

        self._curve_figure_ax.cla()

        self._controller.plotData(self._curve_figure)
        self._drawSelectionsDataSeries()
        self._widgets_require_subplates.sensitive = \
            self._model['subplateSelected'].any()

    def _saveAbsolute(self, widget):

        dialog = gtk.FileChooserDialog(
            title=self._model['saveTo'],
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_SAVE,
                     gtk.RESPONSE_OK))
        dialog.set_action(gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER)

        savePath = (dialog.run() == gtk.RESPONSE_OK and dialog.get_filename()
                    or None)
        dialog.destroy()

        if (savePath is not None and self._controller.saveAbsolute(savePath)):
            self._HeatMapInfo.set_text(self._model['saved-absolute'].format(
                savePath))

    def _saveRelative(self, widget):

        dialog = gtk.FileChooserDialog(
            title=self._model['saveTo'],
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_SAVE,
                     gtk.RESPONSE_OK))
        dialog.set_action(gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER)

        savePath = (dialog.run() == gtk.RESPONSE_OK and dialog.get_filename()
                    or None)

        if (savePath is not None and self._controller.saveNormed(savePath)):
            self._HeatMapInfo.set_text(self._model['saved-normed'].format(
                savePath))

    def _setReferences(self, widget):

        self._model['reference-positions'] = self._model[
            'subplateSelected'].copy()
        self._HeatMapInfo.set_text(self._model['set-references'])
        self._widgets_require_references.sensitive = True

    def _doNormalize(self, widget):

        self._controller.normalize()
        self._widgets_require_norm.sensitive = True

    def _undoRemove(self, wiget):

        self._widgets_require_removed.sensitive = self._controller.undoLast()

    def _saveImage(self, widget):

        dialog = gtk.FileChooserDialog(
            title=self._model['saveTo'],
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_SAVE,
                     gtk.RESPONSE_OK))
        dialog.set_action(gtk.FILE_CHOOSER_ACTION_SAVE)

        fname = (dialog.run() == gtk.RESPONSE_OK and dialog.get_filename()
                 or None)

        dialog.destroy()
        if fname is not None:
            fig = (widget is self._plateSaveImage and
                   self._plate_image_canvas.figure
                   or self._curve_figure)

            fig.savefig(fname)

            self._HeatMapInfo.set_text(self._model['image-saved'].format(
                fname))

    def _setColors(self, widget=None, colorSetting=None):

        if (colorSetting is None):

            if self._colorsThisPlate.get_active():
                widget = self._colorsThisPlate
                colorSetting = self.COLOR_THiS
            elif self._colorsAllPlate.get_active():
                widget = self._colorsAllPlate
                colorSetting = self.COLOR_ALL
            else:
                widget = self._colorFixUpdate
                colorSetting = self.COLOR_FIXED

        if colorSetting == self.COLOR_FIXED and widget == self._colorFixUpdate:

            self._widgets_require_fixed_color.sensitive = True
            if self._colorMin.get_text() == "":
                self._colorMin.set_text("0")

        elif widget.get_active():
            self._widgets_require_fixed_color.sensitive = \
                colorSetting == self.COLOR_FIXED

        """
        self._plate_image_canvas.figure.gca().cla()
        self._model['selection_patches'] = None
        """
        self._controller.plotHeatmap(self._plate_image_canvas.figure,
                                     colorSetting)
        self._drawSelectionsDataSeries()

    """Not valid controller doesn't need to inform
    def addSelection(self, pos):

        if pos not in self._model['selection_patches']:
            self._model['selection_patches'][pos] = plt_patches.Rectangle(
                pos, 1, 1, ec='k', fill=True, lw=1,
                hatch='o')
    """

    """ Not in use
    def _remove_patch(self, patch):

        try:
            i = self._plate_figure_ax.patches.index(patch)
            if i >= 0:
                self._plate_figure_ax.patches[i].remove()
        except ValueError:
            pass
    """

    """Not valid controller doesn't need to inform
    def removeSelection(self, pos):

        if pos in self._model['selection_patches']:

            self._remove_patch(
                self._model['selection_patches'][pos])

            del self._model['selection_patches'][pos]

    """

    def _loadMetaData(self, widget):

        dialog = gtk.FileChooserDialog(
            title=self._model['meta-data-files'],
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                     gtk.STOCK_OPEN, gtk.RESPONSE_OK))

        dialog.set_action(gtk.FILE_CHOOSER_ACTION_OPEN)
        dialog.set_select_multiple(True)

        pathMetaData = (dialog.run() == gtk.RESPONSE_OK and dialog.get_filename()
                        or None)

        dialog.destroy()

        if (pathMetaData is not None and
                self._controller.loadMetaData(pathMetaData)):

            self._HeatMapInfo.set_text(self._model["meta-data-loaded"])

    def _loadPlate(self, widget=None):

        if widget is None:

            self._updateBounds()
            self._widgets_require_data.sensitive = False

        else:

            if widget is self._buttonLoadData:

                dialog = gtk.FileChooserDialog(
                    title=self._model['load-data-dir'],
                    buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                             gtk.STOCK_OPEN, gtk.RESPONSE_OK))

                dialog.set_action(gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER)

                dataPath = (dialog.run() == gtk.RESPONSE_OK and
                            dialog.get_filename() or None)

                dialog.destroy()

                if dataPath is not None:
                    if (self._controller.loadPhenotypes(dataPath)):

                        if self._model['phenotype'] is None:
                            self._phenotypeSelector.set_active(0)

                        self._widgets_require_data.sensitive = True
                        self._plateSelectionAdjustment.set_lower(1)
                        self._plateSelectionAdjustment.set_upper(
                            self._model['phenotyper'].shape[0])
                        self._plateSelectionAdjustment.set_value(1)
                        self._setColors()
                        if self._model['plate'] == 0:
                            self._newPhenotype()
                        else:
                            self._model['plate'] = 0

            elif widget is self._plateSelectionAdjustment:

                self._model['plate'] = \
                    int(self._plateSelectionAdjustment.get_value() - 1)
                self._newPhenotype()

    def plotNoData(self, fig, msg="No Data Loaded"):

        fig.gca().cla()
        fig.text(0.25, 0.5, msg)

    def _setBoundaries(self):

        lower, upper, minVal, maxVal = self._controller.getRecommendedFilter()
        self._updatingBounds = True
        self._lowerBoundAdjustment.set_lower(minVal)
        self._lowerBoundAdjustment.set_upper(maxVal)

        self._higherBoundAdjustment.set_lower(minVal)
        self._higherBoundAdjustment.set_upper(maxVal)

        step = (maxVal - minVal) / 250.0

        self._lowerBoundAdjustment.set_page_increment(step)
        self._higherBoundAdjustment.set_page_increment(step)

        self._lowerBoundAdjustment.set_step_increment(step)
        self._higherBoundAdjustment.set_step_increment(step)

        self._lowerBoundAdjustment.set_value(lower)

        self._updatingBounds = False

        self._higherBoundAdjustment.set_value(upper)

    def _updateBounds(self, widget=None):

        if not self._updatingBounds:

            self._updatingBounds = True

            if widget is self._lowerBoundAdjustment:

                if (widget.get_value() >=
                        self._higherBoundAdjustment.get_value()):

                    self._higherBoundAdjustment.set_value(widget.get_value())

            elif widget is self._higherBoundAdjustment:

                if (widget.get_value() <=
                        self._lowerBoundAdjustment.get_value()):

                    self._lowerBoundAdjustment.set_value(widget.get_value())

            self._controller.setSelection(
                self._lowerBoundAdjustment.get_value(),
                self._higherBoundAdjustment.get_value())

            self._updatingBounds = False

    def _mouseHover(self, event, *args):

        if event.xdata is not None and event.ydata is not None:

            if self._model['meta-data'] is None:
                self._HeatMapInfo.set_text(self._model['hover-position'].format(
                    int(round(event.xdata)), int(round(event.ydata))))
            else:
                print "Meta?"

    def _mousePress(self, event, *args, **kwargs):

        if not(None in (event.xdata, event.ydata)):
            self._curSelection = tuple(
                np.round([event.ydata, event.xdata]).tolist())

        else:

            self._curSelection = None

    def _mouseRelease(self, event, *args):

        if not(None in (event.xdata, event.ydata)):
            curSelection = tuple(np.round([event.ydata, event.xdata]).tolist())
            if (curSelection == self._curSelection):

                if self._multiSelecting is False:
                    self._unselect()
                    self._controller.setSelected(curSelection, True)

                """Artist seems not to work in GTK, using data seris hack atm

                    isSel = True
                else:
                    isSel = self._controller.toggleSelection(curSelection)

                if (isSel and curSelection not in
                        self._model['selection_patches']):

                    p = plt_patches.Rectangle(
                        [v - 0.5 for v in curSelection], 1, 1,
                        transform=self._plate_figure_ax.transData,
                        axes=self._plate_figure_ax,
                        color='k', fill=True, lw=1)
                        #hatch='o')

                    self._plate_figure_ax.add_artist(p)
                    self._plate_figure_ax.draw_artist(p)
                    self._model['selection_patches'][curSelection] = True

                elif (not(isSel) and curSelection in
                      self._model['selection_patches']):

                    self._remove_patch(
                        self._model['selection_patches'][curSelection])
                    del self._model['selection_patches'][curSelection]
                """

                self._drawSelectionsDataSeries()

                self._curve_figure_ax.cla()
                self._controller.plotData(self._curve_figure)
                self._curve_image_canvas.draw()

                self._widgets_require_selection.sensitive = \
                    self._model['numberOfSelections'] > 0

    def _drawSelectionsDataSeries(self):

        ax = self._plate_image_canvas.figure.axes[0]

        data = zip(*self._model['selectionCoordinates'])

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if (len(data) == 2):

            Y, X = data

            if (self._model['selection_patches'] is None):

                self._model['selection_patches'] = ax.plot(
                    X, Y, mec='k', mew=1,
                    ms=1, ls="None",
                    marker='s', fillstyle='none')[0]

            else:

                self._model['selection_patches'].set_data(X, Y)

        elif (self._model['selection_patches'] is not None):

            self._model['selection_patches'].set_data([], [])

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        """
        zorder = 0
        for im in ax.images:
            if im.zorder > zorder:
                zorder = im.zorder

        for line in ax.lines:
            zorder += 1
            line.zorder = zorder
        """
        self._plate_image_canvas.draw()

    def _unselect(self, *args):

        """
        for sel in self._model['selection_patches'].values():
            self._remove_patch(sel)

        self._plate_image_canvas.draw()
        self._model['selection_patches'] = dict()
        """

        self._model['plate_selections'][...] = False

        self._widgets_require_selection.sensitive = False

        self._curve_figure_ax.cla()
        """
        self._model['selection_patches'] = None
        """
        self._drawSelectionsDataSeries()

    def _removeCurvesPhenotype(self, *args):

        self._controller.removeCurves(onlyCurrent=True)
        self._widgets_require_removed.sensitive = True

    def _removeCurvesAllPhenotype(self, *args):

        self._controller.removeCurves(onlyCurrent=False)
        self._widgets_require_removed.sensitive = True

    def _newPhenotype(self, widget=None, *args):

        if widget is not None:
            row = widget.get_active()
            model = widget.get_model()
            key = model[row][0]

            self._model['phenotype'] = self._phenotypeName2Key[key]

        if self._model['plate'] is not None:
            self._unselect()
            """
            self._plate_image_canvas.figure.gca().cla()
            self._model['selection_patches'] = None
            """
            self._controller.plotHeatmap(self._plate_image_canvas.figure)
            self._drawSelectionsDataSeries()
            self._setBoundaries()

    def _pressKey(self, widget, event):

        if event.state & gtk.gdk.CONTROL_MASK:
            self._multiSelecting = True
            self._HeatMapInfo.set_text(self._model['msg-multiSelecting'])

    def _releaseKey(self, widget, event):

        if event.state & gtk.gdk.CONTROL_MASK:
            self._multiSelecting = False
            self._HeatMapInfo.set_text("")
            self._controller.plotData(self._curve_figure)


class SensitivityGroup(object):

    def __init__(self, startingValue=True):
        """The Sensitivity group bundles a set of widgets that share
        sensitivity pattern. It also allows for dependent groups False
        cascading.

        Kwargs:

            startingValue (bool):   The starting state of the group.
                                    Defalt value is ``True``

        Group members are added with ``append`` or ``add`` methods.

        Dependent groups are added with ``addDependetGroup`` method

        Senistivity is set by e.g. ``SensitivityGroup.sensitive = False``,
        by using ``SensitivityGroup.toggle()`` or to cascade the setting
        of ``True`` use ``SensitivityGroup.cascadeTrue()``

        .. note::

            By setting senstivity to ``False`` all dependent groups
            are also set to ``False``.

        .. Example Usage::

            >>>button1 = gtk.Button()
            >>>entry1 = gtk.Entry()

            >>>group1 = SensitivityGroup()
            >>>group1.add(button1)
            >>>group1.add(entry1)

            >>>entry2 = gtk.Entry()

            >>>group2 = SensitivityGroup()
            >>>group1.addDependetGroup(group2)
            >>>group2.add(entry2)

            >>>#Toggling group1 will set all to False
            >>>group1.toggle()

            >>>#Setting group1 to True does not cascade:
            >>>group1.sensitive = True

            >>>print group2.sensitive
            False

        """
        self._lastValue = startingValue
        self._members = set()
        self._dependentGroups = set()

    def __keys__(self):

        return self._members

    def __getitem__(self, member):

        if member in self._members:
            return member.get_sensitive()
        else:
            raise KeyError

    def append(self, member):
        """See ``SensitivityGroup.add``"""

        self.add(member)

    def add(self, member):
        """Adds a new group member

        Args:

            member (gtk.Widget):    A new member of the sensitivity group

        .. note::

            Members must expose the methods ``member.get_sensitive()`` and
            ``member.set_senitive(bool)``

        .. note::

            The added member's sensitivity is set to the groups current
            sensitivity state.

        """
        if not(hasattr(member, "get_sensitive") and
               hasattr(member, "set_sensitive")):

            raise TypeError

        else:

            member.set_sensitive(self._lastValue)
            self._members.add(member)

    def _circularityCheck(self, group):
        """Searches dependent group hierarchy for existence of ``group``"""

        if group in self._dependentGroups:
            return True

        for dependentGroup in self._dependentGroups:
            if dependentGroup._circularityCheck():
                return True

        return False

    def addDependetGroup(self, group):
        """Dependet groups are groups that inherits the False
        but no the True of current group

        Raises:

            ValueError: If the current group can be found as depending
                        on the group to be added.

        """

        if group._circularityCheck(self):

            raise ValueError

        else:

            self._dependentGroups.add(group)

    def cascadeTrue(self):
        """Method enforces setting sensitivity to ``True`` for the group's
        members and all members of dependent groups"""

        self.sensitive = True
        for g in self._dependentGroups:
            g.cascadeTrue()

    def toggle(self):
        """Helper method for toggling the state of the sensitivity group"""
        self.sensitive != self._lastValue

    @property
    def sensitive(self):

        return self._lastValue

    @sensitive.setter
    def sensitive(self, value):

        for member in self._members:
            member.set_sensitive(value)

        if value is False:
            for group in self._dependentGroups:
                group.sensitive = value
