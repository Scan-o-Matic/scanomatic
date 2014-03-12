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

#import gtk
#import gobject
from matplotlib import pyplot as plt
import numpy as np

#
# INTERNAL DEPENDENCIES
#

#Own Model and View
import view_qc
import model_qc

#Generics
import scanomatic.gui.generic.controller_generic as controller_generic

#Resources
import scanomatic.io.paths as paths
import scanomatic.io.app_config as app_config
import scanomatic.io.logger as logger
import scanomatic.dataProcessing.phenotyper as phenotyper

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

        view.show_all()

    def loadPhenotypes(self, pathToDirectory):

        try:
            p = phenotyper.Phenotyper.LoadFromSate(pathToDirectory)
        except IOError:
            p = None

        if (p is None or p.source is None or p.phenotypes is None or
                p.times is None or p.smoothData is None or p.shape[0] == 0):

            return False

        self._model['phenotyper'] = p

        self._model['plates'] = [
            i for i, p in enumerate(self._model['phenotyper']) if
            p is not None]

        return True

    def plotData(self, fig):

        plate = self._model['plate']

        if (self._model['phenotyper'] is None or
                self._model['phenotyper'][plate] is None):
            self._view.get_stage().plotNoData(fig)
            return

        for position in zip(*np.where(
                self._model['plate_selections'][
                    ..., self._model['phenotype']])):

            self._model['phenotyper'].plotACurve(
                (plate, ) + position,
                plotRaw=self._model['showRaw'],
                plotSmooth=self._model['showSmooth'],
                plotRegLine=self._model['showGTregLine'],
                plotFit=self._model['showModelLine'],
                annotateGTpos=self._model['showGT'],
                annotateFit=self._model['showFitValue'],
                annotatePosition=not(self._model['multiSelecting']),
                annotatePhenotypeValue=not(self._model['multiSelecting']),
                fig=fig,
                figClear=False,
                showFig=False)

    def getPhenotypes(self):

        return (self._model['phenotyper'] is None and dict() or
                self._model['phenotyper'].NAMES_OF_PHENOTYPES)

    def plotHeatmap(self, fig, newColorSpace=None):

        plate = self._model['plate']

        if (self._model['phenotyper'] is None or
                self._model['phenotyper'][plate] is None):
            self._view.get_stage().plotNoData(fig)
            return

        if newColorSpace is not None:
            stage = self._view.get_stage()

            minVal = None
            maxVal = None

            if (newColorSpace == stage.COLOR_FIXED):

                minVal = stage.colorMin
                maxVal = stage.colorMax

            if minVal is None or maxVal is None:

                self._model['fixedColors'] = (None, None)

            else:

                self._model['fixedColors'] = (minVal, maxVal)

            self._model['colorsAll'] = newColorSpace == stage.COLOR_ALL

        cm = plt.cm.RdBu_r
        cm.set_bad(color='#A0A0A0', alpha=1.0)

        self._model['phenotyper'].plotPlateHeatmap(
            plate,
            measure=self._model['phenotype'],
            useCommonValueAxis=self._model['colorsAll'],
            vmin=self._model['fixedColors'][0],
            vmax=self._model['fixedColors'][1],
            showColorBar=True,
            horizontalOrientation=True,
            cm=cm,
            titleText=None,
            hideAxis=False,
            fig=fig,
            showFig=False)

    def toggleSelection(self, pos):

        pos = tuple(pos)
        p = self._model["plate_selections"]
        p[pos] = p[pos] == False

        return p[pos][self._model['phenotype']]

    def setSelected(self, pos, value):

        pos = tuple(pos)
        p = self._model["plate_selections"]
        updated = value != p[pos]
        p[pos] = value
        return updated.any()

    def removeCurves(self, onlyCurrent=False):

        self._model['phenotyper'].add2RemoveFilter(
            plate=self._model['plate'],
            positionList=self._model['selectionCoordinates'],
            phenotype=onlyCurrent and self._model['phenotype'] or None)

    def undoLast(self):

        #TODO: unflag or something
        return False

    def getRecommendedFilter(self):

        #TODO: Filter plate as first step of norm
        return 0, 1, 0, 1

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
        #stage = self._view.get_stage()
        off1, off2 = platePos
        plateSels = self._model['plate_selections']

        if plateSels is None:
            return

        for id1, d1 in enumerate(plateSels[off2::2]):

            for id2, v in enumerate(d1[off2::2]):

                pos = (id1 * 2 + off1, id2 * 2 + off2)

                self.setSelected(pos, selStatus)

                """Not valid anymore
                if self.setSelected(pos, selStatus):
                    if selStatus:
                        stage.addSelection(pos)
                    else:
                        stage.removeSelection(pos)

                """

    def setSelection(self, lowerBound, higherBound):

        #TODO: Find all positions where not inside bounds
        #If need to toggle inform view
        #stage = self._view.get_stage()
        outliers = []
        for pos in outliers:
            self.setSelected(pos, True)
            """Not valid anymore
            if self.setSelected(pos, True):
                stage.addSelection(pos)
            """

    def Normalize(self):

        #TODO:

        pass
