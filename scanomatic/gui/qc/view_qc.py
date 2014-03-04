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
from matplotlib import patches as plt_patches
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas

#
# INTERNAL DEPENDENCIES
#

import scanomatic.gui.generic.view_generic as view_generic

#
# CLASSES
#


class QC_About(gtk.Label):

    def __init__(self, model, controller):

        self._controller = controller
        self._model = model

        super(QC_About, self).__init__()

        self.set_justify(gtk.JUSTIFY_LEFT)
        self.set_markup(model['about'])

        self.show()


class QC_Stage(gtk.VBox):

    def __init__(self, model, controller):

        super(QC_Stage, self).__init__(False, 0)

        self._model = model
        self._controller = controller

        #
        #PHENOTYPE SELECTOR
        #
        dropbox = gtk.combo_box_new_text()

        self._phenotypeName2Key = {}
        for k in sorted(self._model['phenotyper'].NAMES_OF_PHENOTYPES.keys()):

            n = self._model['phenotyper'].NAMES_OF_PHENOTYPES[k]
            dropbox.append_text(n)
            self._phenotypeName2Key[n] = k

        dropbox.connect(
            "changed", self._newPhenotype)

        dropbox.set_active(0)

        #
        #HEATMAP
        #
        self._plate_figure = plt.Figure(figsize=(40, 40), dpi=150)
        self._plate_figure.add_axes()
        self._plate_figure_ax = self._plate_figure.gca()

        self._plate_image_canvas = FigureCanvas(self._plate_figure)

        self._plate_image_canvas.mpl_connect('button_press_event',
                                             self._mousePress)
        self._plate_image_canvas.mpl_connect('button_release_event',
                                             self._mouseRelease)

        self._plate_image_canvas.mpl_connect('key_press_event',
                                             self._pressKey)

        self._plate_image_canvas.mpl_connect('key_release_event',
                                             self._releaseKey)

        self._HeatMapInfo = gtk.Label("")

        #
        # Curve Display
        #
        self._curve_figure = plt.Figure(figsize=(40, 40), dpi=150)
        self._curve_figure.add_axes()
        self._curve_figure_ax = self._curve_figure.gca()

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

        self._selectionButtons = (self._buttonUnSelect,
                                  self._buttonRemoveCurvesPhenotype,
                                  self._buttonRemoveCurvesAllPhenotypes)

        self._curSelection = None
        self._multiSelecting = False

    def _mousePress(self, event, *args, **args):

        if not(None in (event.xdata, event.ydata)):
            self._curSelection = np.floor([event.xdata, event.ydata]).tolist()

        else:

            self._curSelection = None

    def _mouseRelease(self, event, *args):

        if not(None in (event.xdata, event.ydata)):
            curSelection = np.floor([event.xdata, event.ydata]).tolist()
            if (curSelection == self._curSelection):

                if self._multiSelecting is False:
                    self._unselect()

                if self._controller.toggleSelection(curSelection):
                    self._model['selection_patches'][curSelection] = \
                        plt_patches.Rectangle(
                            curSelection, 1, 1, ec='k', fill=True, lw=1,
                            hatch='o')

                elif curSelection in self._model['selection_patches']:

                    self._plate_figure_ax.remove_patch(
                        self._model['selection_patches'][curSelection])

                    del self._model['selection_patches'][curSelection]

                if not self._multiSelecting:
                    self._controller.plotData(self._curve_figure)

                for b in self._selectionButtons:
                    b.set_sensitive(
                        len(self._model['selection_patches']) > 0)

    def _unselect(self, *args):

        for sel in self._model['selection_patches'].values():
            self._plate_figure_ax.remove_patch(sel)

        self._model['selection_patches'] = dict()

        for b in self._selectionButtons:
            b.set_sensitive(False)

        self._curve_figure_ax.cla()

    def _removeCurvesPhenotype(self, *args):

        self._controller.removeCurves(onlyCurrent=True)

    def _removeCurvesAllPhenotype(self, *args):

        self._controller.removeCurves(onlyCurrent=False)

    def _newPhenotype(self, widget, *args):

        row = widget.get_active()
        model = widget.get_model()
        key = model[row][0]

        self._model['phenotype'] = self._phenotypeName2Key[key]
        self._unselect()
        self._controller.plotHeatmap(self._plate_figure)

    def _pressKey(self, key):

        if "control" in key or "ctrl" in key:
            self._multiSelecting = True
            self._HeatMapInfo.set_text(self._model['multiselecting'])

    def _releaseKey(self, key):

        if "control" in key or "ctrl" in key:
            self._multiSelecting = False
            self._HeatMapInfo.set_text("")
            self._controller.plotData(self._curve_figure)
