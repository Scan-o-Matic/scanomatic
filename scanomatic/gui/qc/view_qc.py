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
import pango

import numpy as np

from matplotlib import pyplot as plt
#from matplotlib import patches as plt_patches
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas

#
# INTERNAL DEPENDENCIES
#

import scanomatic.dataProcessing.phenotyper as phenotyper
import scanomatic.io.logger as logger

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

        self._logger = logger.Logger("QC Stage")

        isDebug = model['debug-mode']

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

        self._widgets_require_data.add(self._buttonLoadMetaData)

        #
        # PLATE SELECTOR
        #

        hbox = gtk.HBox(False, spacing=2)

        self._plateSelectionAdjustment = gtk.Adjustment(0, 0, 0, 1, 1, 0)
        self._plateSelector = gtk.SpinButton(self._plateSelectionAdjustment,
                                             0, 0)
        self._plateSelector.connect("value_changed", self._loadPlate)
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
        #MAIN BOXES
        #

        vboxLeft = gtk.VBox(False, spacing=2)
        vboxRight = gtk.VBox(False, spacing=2)
        hbox = gtk.HBox(False, spacing=2)
        self.pack_start(hbox, expand=True, fill=True)
        hbox.pack_start(vboxLeft, expand=True, fill=True)
        hbox.pack_start(vboxRight, expand=False, fill=False)

        #
        #HEATMAP
        #

        self._plate_figure = plt.Figure(figsize=(40, 40), dpi=150)
        self._plate_figure.add_axes()
        self._plate_figure_ax = self._plate_figure.gca()
        self._plate_figure_ax.set_axis_off()
        self._plate_image_canvas = FigureCanvas(self._plate_figure)

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
        self._HeatMapInfo.set_justify(gtk.JUSTIFY_CENTER)
        self._HeatMapInfo.set_ellipsize(pango.ELLIPSIZE_MIDDLE)

        self._plateSaveImage = gtk.Button(self._model['plate-save'])
        self._plateSaveImage.connect("clicked", self._saveImage)
        self._widgets_require_data.append(self._plateSaveImage)

        vboxLeft.pack_start(self._plate_image_canvas, expand=True, fill=True)
        hbox2 = gtk.HBox(False, spacing=2)
        hbox2.pack_start(self._plateSaveImage, expand=False, fill=False)
        hbox2.pack_start(self._HeatMapInfo, expand=True, fill=True)
        vboxLeft.pack_start(hbox2, expand=False, fill=False)

        #
        # HEATMAP COLOR VALUES
        #

        frame = gtk.Frame(self._model['colors'])
        vbox2 = gtk.VBox(False, spacing=2)
        frame.add(vbox2)
        vboxRight.pack_start(frame, expand=False, fill=False)

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
        #vboxRight.pack_start(frame, expand=False, fill=False)
        vbox2 = gtk.VBox(False, spacing=2)
        frame.add(vbox2)

        self._lowerBoundAdjustment = gtk.Adjustment(
            value=0, lower=0, upper=1)
        self._higherBoundAdjustment = gtk.Adjustment(
            value=0, lower=0, upper=1)
        self._lowerBoundWidget = gtk.HScale(self._lowerBoundAdjustment)
        self._higherBoundWidget = gtk.HScale(self._higherBoundAdjustment)
        """
        self._lowerBoundWidget.set_upper_stepper_sensitivity(
            gtk.SENSITIVITY_ON)
        self._lowerBoundWidget.set_lower_stepper_sensitivity(
            gtk.SENSITIVITY_ON)
        self._higherBoundWidget.set_upper_stepper_sensitivity(
            gtk.SENSITIVITY_ON)
        self._higherBoundWidget.set_lower_stepper_sensitivity(
            gtk.SENSITIVITY_ON)
        """
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

        self._badSelectorAdjustment = gtk.Adjustment(1, 1, 1, 1, 1, 0)
        self._badSelector = gtk.SpinButton(self._badSelectorAdjustment,
                                           0, 0)

        self._badSelector.connect("value_changed", self._selectBad)

        self._badSelector.set_wrap(True)
        self._badSelector.set_snap_to_ticks(True)

        self._widgets_require_data.add(self._badSelector)

        frame = gtk.Frame(self._model['selections-section'])
        vbox2 = gtk.VBox(False, spacing=2)
        frame.add(vbox2)
        vboxRight.pack_start(frame, expand=False, fill=False)
        vbox2.pack_start(self._buttonUnSelect, expand=False, fill=False)
        vbox2.pack_start(gtk.HSeparator(), expand=False, fill=False, padding=4)
        badHbox = gtk.HBox(False, spacing=2)
        badHbox.pack_start(gtk.Label(self._model['badness-label']),
                           expand=False, fill=False)
        badHbox.pack_start(self._badSelector, expand=False, fill=False)
        vbox2.pack_start(badHbox, expand=False, fill=False)
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

        vboxLeft.pack_start(self._curve_image_canvas, expand=True, fill=True)

        hbox = gtk.HBox(False, spacing=2)
        hbox.pack_start(self._curveSaveImage, expand=False, fill=False)
        vboxLeft.pack_start(hbox, expand=False, fill=False)

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
        vboxRight.pack_start(frame, expand=False, fill=False)
        hbox2 = gtk.HBox(False, spacing=2)
        vbox2.pack_start(hbox2, expand=False, fill=False)
        hbox2.pack_start(self._subplate_0_0, expand=False, fill=False)
        hbox2.pack_start(self._subplate_0_1, expand=False, fill=False)
        hbox2 = gtk.HBox(False, spacing=2)
        vbox2.pack_start(hbox2, expand=False, fill=False)
        hbox2.pack_start(self._subplate_1_0, expand=False, fill=False)
        hbox2.pack_start(self._subplate_1_1, expand=False, fill=False)

        #
        # Normalize
        #

        self._setRefButton = gtk.Button(self._model['references'])
        self._setRefButton.connect("clicked", self._setReferences)

        self._refPositionText = gtk.Label("")
        self._setReferences()

        self._widgets_require_subplates.append(self._setRefButton)

        self._normalize = gtk.Button(self._model['normalize'])
        self._normalize.connect("clicked", self._doNormalize)

        self._widgets_require_references.append(self._normalize)

        frame = gtk.Frame(self._model['frame-normalize'])
        vboxRight.pack_start(frame, expand=False, fill=False)

        vbox2 = gtk.VBox(False, spacing=2)
        frame.add(vbox2)
        vbox2.pack_start(self._setRefButton, expand=False, fill=False)
        vbox2.pack_start(self._refPositionText, expand=False, fill=False)
        vbox2.pack_start(gtk.HSeparator(), expand=False, fill=False, padding=4)

        if isDebug:

            self._toggleAlgInLog = gtk.CheckButton(
                self._model['norm-alg-in-log-text'])
            self._toggleAlgInLog.set_active(
                self._model['norm-alg-in-log'])
            self._toggleAlgInLog.connect("toggled",
                                         self._setToggleLogInAlg)
            self._widgets_require_references.add(self._toggleAlgInLog)
            vbox2.pack_start(self._toggleAlgInLog,
                             expand=False, fill=False)

            vbox2.pack_start(gtk.HSeparator(),
                             expand=False, fill=False, padding=4)

            hbox = gtk.HBox(False, spacing=2)
            hbox.pack_start(gtk.Label("n-sigma"), expand=False, fill=False)
            e = gtk.Entry()
            e.set_text(str(self._model['norm-outlier-k']))
            e.connect("changed", self._setNormK)
            e.set_size_request(0, -1)
            hbox.pack_start(e, expand=True, fill=True)
            self._widgets_require_references.add(e)

            vbox2.pack_start(hbox, expand=False, fill=False)

            hbox = gtk.HBox(False, spacing=2)
            hbox.pack_start(gtk.Label("Est. n filtered"),
                            expand=False, fill=False)
            e = gtk.Entry()
            e.set_text(str(self._model['norm-outlier-p']))
            e.connect("changed", self._setNormP)
            e.set_size_request(0, -1)
            hbox.pack_start(e, expand=True, fill=True)
            self._widgets_require_references.add(e)

            vbox2.pack_start(hbox, expand=False, fill=False)

            hbox = gtk.HBox(False, spacing=2)
            hbox.pack_start(gtk.Label("Max Iterations"),
                            expand=False, fill=False)
            e = gtk.Entry()
            e.set_text(str(self._model['norm-outlier-iterations']))
            e.connect("changed", self._setNormIterations)
            e.set_size_request(0, -1)
            hbox.pack_start(e, expand=True, fill=True)
            self._widgets_require_references.add(e)

            vbox2.pack_start(hbox, expand=False, fill=False)

            vbox2.pack_start(gtk.HSeparator(),
                             expand=False, fill=False, padding=4)

            self._toggleInitialValueUse = gtk.CheckButton(
                self._model['norm-use-initial-text'])
            self._toggleInitialValueUse.set_active(
                self._model['norm-use-initial-values'])
            self._toggleInitialValueUse.connect("toggled",
                                                self._setNormWithInitals)

            vbox2.pack_start(self._toggleInitialValueUse,
                             expand=False, fill=False)

            self._widgets_require_references.add(self._toggleInitialValueUse)

            hbox = gtk.HBox(False, spacing=2)
            hbox.pack_start(gtk.Label("Spline"),
                            expand=False, fill=False)
            e = gtk.Entry()
            e.set_text(str(self._model['norm-spline-seq']))
            e.connect("changed", self._setNormSplineSeq)
            e.set_size_request(0, -1)
            hbox.pack_start(e, expand=True, fill=True)
            self._widgets_require_references.add(e)

            vbox2.pack_start(hbox, expand=False, fill=False)

            vbox2.pack_start(gtk.HSeparator(),
                             expand=False, fill=False, padding=4)

            hbox = gtk.HBox(False, spacing=2)
            hbox.pack_start(gtk.Label("Median filt"), expand=False, fill=False)
            e = gtk.Entry()
            e.set_text(str(self._model['norm-outlier-fillSize']))
            e.connect("changed", self._setNormNanFillShape)

            e.set_size_request(0, -1)
            hbox.pack_start(e, expand=True, fill=True)
            self._widgets_require_references.add(e)

            vbox2.pack_start(hbox, expand=False, fill=False)
            hbox = gtk.HBox(False, spacing=2)
            hbox.pack_start(gtk.Label("Gauss Smooth"),
                            expand=False, fill=False)
            e = gtk.Entry()
            e.set_text(str(self._model['norm-smoothing']))
            e.connect("changed", self._setNormSmoothing)
            e.set_size_request(0, -1)
            hbox.pack_start(e, expand=True, fill=True)
            self._widgets_require_references.add(e)

            vbox2.pack_start(hbox, expand=False, fill=False)

        vbox2.pack_start(self._normalize, expand=False, fill=False)

        #
        # Save
        #

        self._savePhenoAbs = gtk.Button(self._model['save-absolute'])
        self._savePhenoAbs.connect("clicked", self._saveAbsolute)

        self._widgets_require_data.append(self._savePhenoAbs)

        self._saveNormed = gtk.Button(self._model['save-relative'])
        self._saveNormed.connect("clicked", self._saveRelative)

        self._widgets_require_norm.append(self._saveNormed)

        self._saveState = gtk.Button(self._model['save-state'])
        self._saveState.connect("clicked", self._saveCurState)

        self._widgets_require_removed.add(self._saveState)

        self._curSelection = None
        self._multiSelecting = False

        vboxRight.pack_start(self._savePhenoAbs, expand=False, fill=False,
                             padding=4)

        vboxRight.pack_start(self._saveNormed, expand=False, fill=False,
                             padding=4)

        vboxRight.pack_start(self._saveState, expand=False, fill=False,
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

    def _testEntryIsType(self, widget, dtype=float):
        """If widget has a value understandable as a float, it is returned
        else None is returned."""

        v = widget.get_text()

        if dtype is float:
            if "," in v:
                if "." not in v:
                    v = v.replace(",", ".")
                else:
                    v = v.replace(",", "")

        v = v.replace(" ", "")

        try:

            v = dtype(v)
            widget.set_icon_from_stock(gtk.ENTRY_ICON_SECONDARY, None)

        except ValueError:
            v = None
            widget.set_icon_from_stock(gtk.ENTRY_ICON_SECONDARY,
                                       gtk.STOCK_DIALOG_ERROR)

        return v

    def _setToggleLogInAlg(self, widget):

        self._model['norm-alg-in-log'] = widget.get_active()

    def _setNormSmoothing(self, widget):

        v = self._testEntryIsType(widget, dtype=float)
        if v is not None and v > 0:
            if self._model['norm-smoothing'] != v:
                self._model['norm-smoothing'] = v
                widget.set_text(str(v))
        elif widget.get_text().lower() in ["", "none"]:
            self._model['norm-smoothing'] = None
            widget.set_icon_from_stock(gtk.ENTRY_ICON_SECONDARY,
                                       None)
        else:
            widget.set_icon_from_stock(gtk.ENTRY_ICON_SECONDARY,
                                       gtk.STOCK_DIALOG_ERROR)

    def _setNormIterations(self, widget):

        v = self._testEntryIsType(widget, dtype=int)
        if v is not None and v >= 0:
            if self._model['norm-outlier-iterations'] != v:
                self._model['norm-outlier-iterations'] = v
                widget.set_text(str(v))
        else:
            widget.set_icon_from_stock(gtk.ENTRY_ICON_SECONDARY,
                                       gtk.STOCK_DIALOG_ERROR)

    def _setNormP(self, widget):

        v = self._testEntryIsType(widget, dtype=float)
        if v is not None:
            if self._model['norm-outlier-p'] != v:
                self._model['norm-outlier-p'] = v
                widget.set_text(str(v))

    def _setNormK(self, widget):

        v = self._testEntryIsType(widget, dtype=float)
        if v is not None:
            if self._model['norm-outlier-k'] != v:
                self._model['norm-outlier-k'] = v
                widget.set_text(str(v))
        elif widget.get_text().lower() in ["", "none"]:
            self._model['norm-outlier-k'] = None
            widget.set_icon_from_stock(gtk.ENTRY_ICON_SECONDARY,
                                       None)

    def _setNormWithInitals(self, widget):
        self._model['norm-use-initial-values'] = widget.get_active()

    def _setNormSplineSeq(self, widget):

        defaults = ('cubic', 'linear', 'nearest')
        v = widget.get_text().lower()
        if v == "":
            self._model['norm-spline-seq'] = defaults
            widget.set_icon_from_stock(gtk.ENTRY_ICON_SECONDARY, None)
            widget.set_text(str(defaults))

        else:

            v = tuple(i.strip(" '\"").lower() for
                      i in v.lstrip("(").rstrip(")").split(",")
                      if i != "")

            if all(i in defaults for i in v):

                if v != self._model['norm-spline-seq']:
                    self._model['norm-spline-seq'] = v
                    widget.set_text(str(v))
                    widget.set_icon_from_stock(gtk.ENTRY_ICON_SECONDARY, None)

            else:

                widget.set_icon_from_stock(gtk.ENTRY_ICON_SECONDARY,
                                           gtk.STOCK_DIALOG_ERROR)

    def _setNormNanFillShape(self, widget):

        v = widget.get_text().lower()
        if v == "" or v == "none":
            self._model['norm-outlier-fillSize'] = None
            widget.set_icon_from_stock(gtk.ENTRY_ICON_SECONDARY, None)
        else:
            try:
                t = eval(v)
            except (SyntaxError, NameError, TypeError):
                t = None

            if (isinstance(t, tuple) and len(t) == 2 and
                    all(isinstance(v, int) for v in t) and t[0] == t[1] and
                    t[0] % 2 == 1):

                self._model['norm-outlier-fillSize'] = t

                widget.set_text(str(self._model['norm-outlier-fillSize']))
                widget.set_icon_from_stock(gtk.ENTRY_ICON_SECONDARY, None)
            else:
                widget.set_icon_from_stock(gtk.ENTRY_ICON_SECONDARY,
                                           gtk.STOCK_DIALOG_ERROR)

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
        dialog.set_action(gtk.FILE_CHOOSER_ACTION_SAVE)
        dialog.set_current_folder(self._model['phenotyper-path'])

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
        dialog.set_action(gtk.FILE_CHOOSER_ACTION_SAVE)
        dialog.set_current_folder(self._model['phenotyper-path'])

        savePath = (dialog.run() == gtk.RESPONSE_OK and dialog.get_filename()
                    or None)

        dialog.destroy()

        if (savePath is not None and self._controller.saveNormed(savePath)):
            self._HeatMapInfo.set_text(self._model['saved-normed'].format(
                savePath))

    def _saveCurState(self, widget):

        self._controller.saveState()

    def _setReferences(self, widget=None):

        if widget is not None:
            self._controller.setReferencePositions()
            self._HeatMapInfo.set_text(self._model['set-references'])

        if (self._model['reference-positions'] is None):
            self._refPositionText.set_text(self._model['no-reference'])
        else:
            self._refPositionText.set_text(
                self._model['yes-reference'] +
                ", ".join([self._model['reference-offset-names'][
                    y * 2 + x] for y, x in
                    zip(*np.where(self._model['reference-positions'][
                        self._model['plate']]))]))

            self._widgets_require_references.sensitive = \
                self._model['reference-positions'][self._model['plate']].any()

    def _doNormalize(self, widget):

        normInfo = self._controller.normalize()

        if (normInfo['ref-CV-warning'].any() or
                normInfo['ref-usage-warning'].any()):

            msg = "\n\n".join(

                self._model['ref-bad-plate'].format(
                    pId + 1,
                    normInfo['ref-usage'][pId] * 100,
                    normInfo['ref-CV'][pId])

                for pId in range(normInfo['ref-CV'].size)

                if normInfo['ref-CV-warning'][pId] or
                normInfo['ref-usage-warning'][pId])

            if len(msg) > 0:

                d = gtk.MessageDialog(
                    type=gtk.MESSAGE_WARNING,
                    buttons=gtk.BUTTONS_OK,
                    message_format=self._model['ref-warning-head'] + msg)

                d.run()
                d.destroy()

        self._widgets_require_norm.sensitive = True
        self._HeatMapInfo.set_text(self._model['normalized-text'])
        if (self._model['phenotype'] != self._model['absPhenotype']):
            self._newPhenotype()
        else:

            normedName = self._model['normalized-phenotype'].format(
                phenotyper.Phenotyper.NAMES_OF_PHENOTYPES[
                    self._model['phenotype']])

            if normedName in self._phenotypeName2Key:
                #This switches view to the normalized version if exists
                for i, row in enumerate(self._phenotypeSelector.get_model()):

                    if row[0] == normedName:

                        self._phenotypeSelector.set_active(i)
                        break

    def updateAvailablePhenotypes(self):

        if (len(self._phenotypeSelector.get_model()) <
                len(phenotyper.Phenotyper.NAMES_OF_PHENOTYPES) +
                len(self._model['normalized-phenotype-names'])):

            wrapText = self._model['normalized-phenotype']
            offset = self._model['normalized-index-offset']

            for k in sorted(self._model['normalized-phenotype-names'].keys()):

                n = wrapText.format(
                    self._model['normalized-phenotype-names'][k])
                self._phenotypeSelector.append_text(n)
                self._phenotypeName2Key[n] = k + offset

    def _undoRemove(self, wiget):

        didSomething = self._controller.undoLast()
        if (not(didSomething)):

            dialog = gtk.MessageDialog(
                flags=gtk.DIALOG_DESTROY_WITH_PARENT,
                type=gtk.MESSAGE_WARNING,
                buttons=gtk.BUTTONS_YES_NO,
                message_format=self._model["unremove-text"])

            if dialog.run() == gtk.RESPONSE_YES:

                self._model['removed_filter'][...] = np.False_
                self._model['platesHaveUnsaved'][self._model['plate']] = False
                self._newPhenotype()

            dialog.destroy()

        self._widgets_require_removed.sensitive = \
            self._model['plate_has_removed']

    def _saveImage(self, widget):

        dialog = gtk.FileChooserDialog(
            title=self._model['saveTo'],
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_SAVE,
                     gtk.RESPONSE_OK))
        dialog.set_action(gtk.FILE_CHOOSER_ACTION_SAVE)
        dialog.set_current_folder(self._model['phenotyper-path'])

        fname = (dialog.run() == gtk.RESPONSE_OK and dialog.get_filename()
                 or None)

        dialog.destroy()
        if fname is not None:
            fig = (widget is self._plateSaveImage and
                   self._plate_figure
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

        self._controller.plotHeatmap(self._plate_figure,
                                     colorSetting)
        self._drawSelectionsDataSeries()

    def _loadMetaData(self, widget):

        dialog = gtk.FileChooserDialog(
            title=self._model['meta-data-files'],
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                     gtk.STOCK_OPEN, gtk.RESPONSE_OK))

        dialog.set_action(gtk.FILE_CHOOSER_ACTION_OPEN)
        dialog.set_select_multiple(True)
        dialog.set_current_folder(self._model['phenotyper-path'])

        pathMetaData = (dialog.run() == gtk.RESPONSE_OK and
                        dialog.get_filenames()
                        or None)

        dialog.destroy()

        if (pathMetaData is not None and
                self._controller.loadMetaData(pathMetaData)):

            self._HeatMapInfo.set_text(self._model["meta-data-loaded"])

    def _loadPlate(self, widget=None):

        if widget is None:
            self._setBoundaries()
            self._updateBounds()
            self._widgets_require_data.sensitive = False

        else:

            if widget is self._buttonLoadData:

                dialog = gtk.FileChooserDialog(
                    title=self._model['load-data-dir'],
                    buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                             gtk.STOCK_OPEN, gtk.RESPONSE_OK))

                dialog.set_action(gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER)
                dialog.set_current_folder(self._model['phenotyper-path'])

                dataPath = (dialog.run() == gtk.RESPONSE_OK and
                            dialog.get_filename() or None)

                dialog.destroy()

                if dataPath is not None:
                    if (self._controller.loadPhenotypes(dataPath)):

                        self._model['auto-selecting'] = True
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

                        s = self._model['plate_size']
                        if s is None or s < 1:
                            self._badSelectorAdjustment.set_upper(1)
                        else:
                            self._badSelectorAdjustment.set_upper(s)

                        self._badSelectorAdjustment.set_value(0)
                        self._selectBad(self._badSelector)
                        self._setReferences()
                        self._HeatMapInfo.set_text(
                            self._model['loaded-text'])

                    else:

                        self._HeatMapInfo.set_text(
                            self._model['load-fail-text'])

            elif widget is self._plateSelector:

                p = self._plateSelector.get_value_as_int() - 1

                if p != self._model['plate']:
                    self._model['plate'] = p

                    if self._model['meta-data'] is not None:

                        s = self._model['plate_size']
                        if s is None or s < 1:
                            self._badSelectorAdjustment.set_upper(1)
                        else:
                            self._badSelectorAdjustment.set_upper(s)

                        self._controller.guessBestColumn()

                    self._setReferences()
                    self._newPhenotype()
                    if (self._model['auto-selecting']):
                        self._badSelectorAdjustment.set_value(0)

    def plotNoData(self, fig, msg="No Data Loaded"):

        for ax in fig.axes:
            ax.cla()
        ax.text(0.25, 0.5, msg)
        if (fig == self._plate_figure):
            self._plate_image_canvas.draw()

    def _setBoundaries(self):

        lower, upper, minVal, maxVal = self._controller.getRecommendedFilter()
        self._updatingBounds = True
        self._lowerBoundAdjustment.set_lower(minVal)
        self._lowerBoundAdjustment.set_upper(maxVal)

        self._higherBoundAdjustment.set_lower(minVal)
        self._higherBoundAdjustment.set_upper(maxVal)

        """
        step = (maxVal - minVal) / 250.0
        print step
        self._lowerBoundAdjustment.set_page_size(step)
        self._higherBoundAdjustment.set_page_size(step)
        self._higherBoundAdjustment.set_step_increment(step)
        self._higherBoundAdjustment.set_page_increment(step)
        self._lowerBoundAdjustment.set_page_increment(step)
        self._higherBoundAdjustment.set_page_increment(step)

        self._lowerBoundAdjustment.set_step_increment(step)
        self._higherBoundAdjustment.set_step_increment(step)
        """
        self._lowerBoundAdjustment.set_value(lower)
        self._higherBoundAdjustment.set_value(upper)

        self._updatingBounds = False

    def _updateBounds(self, widget=None):

        if not self._updatingBounds:

            self._updatingBounds = True
            if widget is not None:
                print widget.get_step_increment()

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

            x = int(round(event.xdata))
            y = int(round(event.ydata))

            posText = self._model['hover-position'].format(y, x)

            if self._model['meta-data'] is None:
                self._HeatMapInfo.set_text(posText)
            else:

                posMD = self._model['meta-data'](self._model['plate'], y, x)

                if self._model['meta-data-info-column'] < len(posMD):
                    self._HeatMapInfo.set_text(
                        posText + "; " +
                        self._model['meta-data-info-columnName'] + ": " +
                        posMD[self._model['meta-data-info-column']])
                else:
                    self._HeatMapInfo.set_text(posText)

    def _mousePress(self, event, *args, **kwargs):

        if (not(None in (event.xdata, event.ydata)) and
                self._model['plate_exists']):
            self._curSelection = tuple(
                np.round([event.ydata, event.xdata]).tolist())

        else:

            self._curSelection = None

    def _mouseRelease(self, event, *args):

        if (not(None in (event.xdata, event.ydata)) and
                self._model['plate_exists']):
            curSelection = tuple(np.round([event.ydata, event.xdata]).tolist())
            if (curSelection == self._curSelection):

                self._model['auto-selecting'] = False

                if self._multiSelecting is False:
                    self._unselect()
                    self._controller.setSelected(curSelection, True)
                else:
                    self._controller.toggleSelection(curSelection)

                self._drawSelectionsDataSeries()

                if not self._multiSelecting:
                    self._curve_figure_ax.cla()
                    self._controller.plotData(self._curve_figure)
                    self._curve_image_canvas.draw()

                self._widgets_require_selection.sensitive = \
                    self._model['numberOfSelections'] > 0

    def _drawSelectionsDataSeries(self):

        ax = self._plate_figure.axes[0]
        ax.set_axis_off()

        data = zip(*self._model['selectionCoordinates'])

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if (len(data) == 2):

            Y, X = data

            if (self._model['selection_patches'] is None or len(ax.lines) == 0):

                self._model['selection_patches'] = ax.plot(
                    X, Y, mec='k', mew=1,
                    ms=1, ls="None",
                    marker='s', fillstyle='none')[0]

            else:

                self._model['selection_patches'].set_data(X, Y)

        elif (self._model['selection_patches'] is not None):

            #self._logger.info("Unknown selection {0}".format(data))

            self._model['selection_patches'].set_data([], [])

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        zorder = 0
        for im in ax.images:
            if im.zorder > zorder:
                zorder = im.zorder

        for line in ax.lines:
            zorder += 1
            line.zorder = zorder

        self._plate_image_canvas.draw()

    def _selectBad(self, widget):

        self._model['auto-selecting'] = True
        if self._model['phenotype'] not in self._model['badSortingPhenotypes']:
            pos = self._controller.getMostProbableBad(widget.get_value_as_int())
        else:
            pos = self._controller.getPhenotypeBad(widget.get_value_as_int())

        self._model['plate_selections'][...] = False

        if pos is None or len(pos) != 2 or len(pos[0]) == 0:
            n = int(widget.get_value())
            self._curve_figure_ax.cla()
            self._curve_figure_ax.text(0.25, 0.5,
                                       "Already Removed the #{0} worst".format(
                                       n))
            self._curve_image_canvas.draw()
            return None

        X, Y = pos

        self._model['plate_selections'][X, Y] = True
        self._curve_figure_ax.cla()
        self._controller.plotData(self._curve_figure)
        self._curve_image_canvas.draw()
        self._drawSelectionsDataSeries()
        self._widgets_require_selection.sensitive = True

    def _unselect(self, *args):

        self._model['plate_selections'][...] = False

        self._widgets_require_selection.sensitive = False

        self._curve_figure_ax.cla()
        self._drawSelectionsDataSeries()

    def _removeCurvesPhenotype(self, *args):

        self._controller.removeCurves(onlyCurrent=True)
        self._unselect()
        self._newPhenotype()
        if (self._model['auto-selecting']):
            self._badSelectorAdjustment.set_value(
                self._badSelectorAdjustment.get_value() + 1)

    def _removeCurvesAllPhenotype(self, *args):

        self._controller.removeCurves(onlyCurrent=False)
        self._unselect()
        self._newPhenotype()
        if (self._model['auto-selecting']):
            self._badSelectorAdjustment.set_value(
                self._badSelectorAdjustment.get_value() + 1)

    def _newPhenotype(self, widget=None, *args):

        if widget is not None:
            row = widget.get_active()
            model = widget.get_model()
            key = model[row][0]

            self._model['phenotype'] = self._phenotypeName2Key[key]

        if self._model['plate'] is not None:
            self._unselect()
            if self._controller.plotHeatmap(self._plate_figure):
                self._drawSelectionsDataSeries()
                self._setBoundaries()
                self._updateBounds()
            else:
                self._HeatMapInfo.set_text(self._model['phenotype-fail-text'])

        self._widgets_require_removed.sensitive = \
            self._model['plate_has_removed']

    def _pressKey(self, widget, event):

        ctrlKey = "CONTROL" in gtk.gdk.keyval_name(event.keyval).upper()

        if ctrlKey or event.state & gtk.gdk.CONTROL_MASK:
            self._multiSelecting = True
            self._HeatMapInfo.set_text(self._model['msg-multiSelecting'])
        else:
            keyName = gtk.gdk.keyval_name(event.keyval).upper()
            if (keyName == "D"):
                self._removeCurvesAllPhenotype()
            elif (keyName in ["N", "W", "D", "L"]):
                self._badSelectorAdjustment.set_value(
                    self._badSelectorAdjustment.get_value() + 1)
            elif (keyName in ["B", "S", "A", "H"]):
                self._badSelectorAdjustment.set_value(
                    self._badSelectorAdjustment.get_value() - 1)

    def _releaseKey(self, widget, event):

        ctrlKey = "CONTROL" in gtk.gdk.keyval_name(event.keyval).upper()

        if ctrlKey or event.state & gtk.gdk.CONTROL_MASK:
            self._multiSelecting = False
            self._HeatMapInfo.set_text("")
            self._curve_figure_ax.cla()
            self._controller.plotData(self._curve_figure)
            self._curve_image_canvas.draw()
        """
        else:

            keyName = gtk.gdk.keyval_name(event.keyval).upper()
        """


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
