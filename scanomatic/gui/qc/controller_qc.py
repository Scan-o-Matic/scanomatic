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
import scanomatic.io.meta_data as meta_data
import scanomatic.dataProcessing.phenotyper as phenotyper
import scanomatic.dataProcessing.subPlates as sub_plates
import scanomatic.dataProcessing.dataBridge as data_bridge
import scanomatic.dataProcessing.norm as norm

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
        self._model['phenotyper-path'] = pathToDirectory

        self._model['plates'] = [
            i for i, p in enumerate(self._model['phenotyper']) if
            p is not None]

        return True

    def saveState(self):

        self._model['phenotyper'].saveState(
            self._model['phenotyper-path'],
            askOverwrite=False)

        return True

    def plotData(self, fig):

        plate = self._model['plate']

        if (self._model['phenotyper'] is None or
                self._model['phenotyper'][plate] is None):
            self._view.get_stage().plotNoData(fig)
            return

        for position in self._model['selectionCoordinates']:

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

        data = None
        measure = self._model['absPhenotype']

        if (self._model['normalized-index-offset'] is not None and
                self._model['phenotype'] >=
                self._model['normalized-index-offset']):

            data = self._model['normalized-data']

        self._model['phenotyper'].plotPlateHeatmap(
            plate,
            measure=measure,
            data=data,
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

        return p[pos][self._model['absPhenotype']]

    def setSelected(self, pos, value):

        pos = tuple(pos)
        p = self._model["plate_selections"]
        updated = value != p[pos]
        p[pos] = value
        return updated.any()

    def removeCurves(self, onlyCurrent=False):

        self._model['phenotyper'].add2RemoveFilter(
            plate=self._model['plate'],
            positionList=self._model['selectionWhere'],
            phenotype=onlyCurrent and self._model['absPhenotype'] or None)

    def undoLast(self):

        #TODO: unflag or something
        return False

    def getRecommendedFilter(self):

        lb = self._model['visibleMin']
        ub = self._model['visibleMax']
        return lb, ub, lb, ub

    def getMostProbableBad(self, index=0):

        wFit = 0.25
        p = self._model['phenotyper']
        pl = self._model['plate']

        gt = p.phenotypes[pl][..., p.PHEN_GT_VALUE]
        gtErr = p.phenotypes[pl][..., p.PHEN_GT_ERR]
        curvFit = p.phenotypes[pl][..., p.PHEN_FIT_VALUE]

        gtBar = gt.ravel()[np.isfinite(gt.ravel())].mean()

        badness = (np.abs(gt - gtBar) / gtBar +
                   (gtErr) * 100 +
                   wFit * (1 - curvFit.clip(0, 1)) * 100)

        pos = np.where(badness == badness.ravel()[
            badness.ravel().argsort()[-index]])

        """DEBUG
        if pos is not None and len(pos) == 2 and len(pos[0]) >= 0:

            print np.abs(gt[pos[0], pos[1]] - gtBar) / gtBar
            print (gtErr[pos[0], pos[1]]) * 100
            print wFit * (1 - curvFit.clip(0, 1)[pos[0], pos[1]]) * 100
        """
        return pos

    def loadMetaData(self, paths):

        self._model['meta-data'] = meta_data.Meta_Data(
            self._model['plate_shapes'], *paths)

        self.guessBestColumn()

    def guessBestColumn(self):

        MD = self._model['meta-data']
        CR = MD[self._model['plate']]

        hRow = MD.getHeaderRow(self._model['plate'])

        if CR.full == MD.PLATE_PARTIAL:

            rows = [
                CR(*pos) for pos in [(0, 0), (1, 0), (0, 1), (1, 1)]]

            allowedCols = min(len(r) for r in rows if rows is not None)

        else:

            allowedCols = len(hRow)

        hRow = hRow[:allowedCols]

        for i, h in enumerate(hRow):

            if h.lower().rstrip("s") in ['strain', 'specie', 'organism',
                                         'construct', 'mutation', 'content']:

                self._model['meta-data-info-column'] = i
                return

        self._model['meta-data-info-column'] = 0

    def saveAbsolute(self, path):

        return self._model['phenotyper'].savePhenotypes(
            path,
            askOverwrite=False)

    def saveNormed(self, path):

        headers = tuple(
            self._model['normalized-phenotype-names'][i] for i
            in range(len(self._model['normalized-phenotype-names'])))

        return self._model['phenotyper'].savePhenotypes(
            path,
            data=self._model['normalized-data'],
            dataHeaders=headers,
            askOverwrite=False)

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

    def setSelection(self, lowerBound, higherBound):

        #TODO: Find all positions where not inside bounds
        #If need to toggle inform view
        #stage = self._view.get_stage()
        outliers = []
        for pos in outliers:
            self.setSelected(pos, True)

    def setReferencePositions(self):

        if (self._model['reference-positions'] is None):

            self._model['reference-positions'] = np.array([
                self._model['subplateSelected'].copy() for _ in
                self._model['phenotyper']])

        else:

            self._model['reference-positions'][self._model['plate']] = \
                self._model['subplateSelected'].copy()

    def normalize(self):

        normalizedPhenotypes = (phenotyper.Phenotyper.PHEN_GT_VALUE,)

        aCopy = []

        for p in self._model['phenotyper'].phenotypes:

            if isinstance(p, np.ma.masked_array):
                aCopy.append(p[..., normalizedPhenotypes].filled().copy())
            else:
                aCopy.append(p[..., normalizedPhenotypes].copy())

        phenotypes = data_bridge.Data_Bridge(np.array(aCopy))

        subSampler = sub_plates.SubPlates(
            phenotypes, kernels=self._model['reference-positions'])

        #If user has missed dubious positions they are filtered out
        for measure in normalizedPhenotypes:
            norm.applyOutlierFilter(subSampler, measure=measure)

        #Data array
        NA = norm.getControlPositionsArray(
            phenotypes,
            controlPositionKernel=subSampler.kernels)

        #Get smothened norm surface
        N = norm.getNormalisationSurfaceWithGridData(
            NA, useAccumulated=False, smoothing=2)

        #Get normed values
        ND = norm.normalisation(phenotypes, N, updateBridge=False)

        self._model['normalized-data'] = ND
        self._model['normalized-index-offset'] = len(
            phenotyper.Phenotyper.NAMES_OF_PHENOTYPES)
        self._model['normalized-phenotype-names'] = {
            i: name for i, (val, name) in enumerate(
                phenotyper.Phenotyper.NAMES_OF_PHENOTYPES.items())
            if val in normalizedPhenotypes}

        self._view.get_stage().updateAvailablePhenotypes()
