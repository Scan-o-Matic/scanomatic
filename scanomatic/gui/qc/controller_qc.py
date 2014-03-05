"""The QC Controller"""
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

import gtk
import gobject

#
# INTERNAL DEPENDENCIES
#

#Own Model and View
import view_qc
import model_qc

#Generics
import scanomatic.gui.generc.controller_generic as controller_generic

#Resources
import scanomatic.io.paths as paths
import scanomatic.io.app_config as app_config
import scanomatic.io.logger as logger
import scanomatic.io.dataProcessing.phenotyper as phenotyper

#
# CLASSES
#


class Controller(controller_generic.Controller):

    def __init__(self, asApp=False, debug_mode=False):

        #PATHS NEED TO INIT BEFORE GUI
        self.paths = paths.Paths()

        if asApp:
            model = model_qc.NewModel.LoadAppModel()
            view = view_qc.Main_Window(controller=self, model=model)
        else:
            model = model_qc.NewModel.LoadStageModel()
            view = view_qc.QC_Stage(controller=self, model=model)

        super(Controller, self).__init__(None, view=view, model=model)
        self._logger = logger.Logger("Main Controller")

        #TODO: FIX new way
        """
        self._logger.SetDefaultOutputTarget(
            self.paths.log_main_out, catchStdOut=True, catchStdErr=True)
        if debug_mode is False:
            self.set_simple_logger()
        """

        self.config = app_config.Config(self.paths)

        self._view.show_notebook_or_logo()

        view.show_all()

    def loadPhenotypes(self, pathToDirectory):

        self._model['phenotyper'] = phenotyper.Phenotyper.LoadFromSate(
            pathToDirectory)

        self._model['plates'] = [
            i for i, p in enumerate(self._model['phenotyper']) if
            p is not None]

    def _plotNoData(self, fig, msg="No Data Loaded"):
        fig.clf()
        fig.text(0.1, 0.4, msg)

    def plotData(self, fig):

        #TODO: Change so it works on selected positoins

        plate = self._model['plate']

        if (self._model['phenotyper'] is None or
                self._model['phenotyper'][plate] is None):
            self._plotNoData(fig)
            return

        self._model['phenotyper'].plotACurve(
            (plate, ) + position,
            measure=self._model['phenotype'],
            plotRaw=self._model['showRaw'],
            plotSmooth=self._model['showSmooth'],
            plotRegLine=self._model['showGTregLine'],
            plotFit=self._model['showModelLine'],
            annotateGTpos=self._model['showGT'],
            annotateFit=self._model['showFitValue'],
            fig=fig,
            figClear=True)

    def getPhenotypes(self):

        return (self._model['phenotyper'] is None and dict() or
                self._model['phenotyper'].NAMES_OF_PHENOTYPES)

    def plotHeatmap(self, fig, newColorSpace=None):

        plate = self._model['plate']

        if (self._model['phenotyper'] is None or
                self._model['phenotyper'][plate] is None):
            self._plotNoData(fig)
            return

        if newColorSpace is not None:
            stage = self._view.get_stage()

            minVal = None
            maxVal = None

            if (newColorSpace == stage.COLOR_FIXED):

                minVal = stage.colorMin
                maxVal = stage.colorMax

            if minVal is None or maxVal is None:

                self._model['fixedColors'] = None

            else:

                self._model['fixedColors'] = (minVal, maxVal)

            self._model['colorsAll'] = newColorSpace == stage.COLOR_ALL

        #TODO: plot!

    def toggleSelection(self, pos):

        self._model["plate_selections"][pos] != \
            self._model["plate_selections"][pos]

        return self._model["plate_selections"][pos]

    def setSelected(self, pos, value):

        updated = value != self._model["plate_selections"][pos]
        self._model["plate_selections"][pos] = value
        return updated

    def removeCurves(self, onlyCurrent=False):

        #TODO: Flag removal filter
        pass

    def undoLast(self):

        #TODO: unflag or something
        return False

    def getRecommendedFilter(self):

        #TODO: Filter plate as first step of norm
        return 0, 1, 0, 1

    def loadData(self, path):

        #TODO: load a phenotyper
        pass

    def loadMetaData(self, path):

        #TODO: load metadata
        pass

    def saveAbsolute(self, path):

        #TODO: Save data to path
        return False

    def saveNormed(self, path):

        #TODO: Save data to path
        return False

    def setSubPlateSelection(self, platePos):

        selStatus = self._model['subplateSelected'][platePos]
        stage = self._view.get_stage()
        off1, off2 = platePos
        for id1, d1 in enumerate(self._model['plate_selections'][off2::2]):

            for id2, v in enumerate(d1[off2::2]):

                pos = (id1 * 2 + off1, id2 * 2 + off2)

                if self.setSelected(pos, selStatus):
                    if selStatus:
                        stage.addSelection(pos)
                    else:
                        stage.removeSelection(pos)

    def setSelection(self, lowerBound, higherBound):

        #TODO: Find all positions where not inside bounds
        #If need to toggle inform view
        stage = self._view.get_stage()
        outliers = []
        for pos in outliers:
            if self.setSelected(pos, True):
                stage.addSelection(pos)

    def Normalize(self):

        #TODO:

        pass
