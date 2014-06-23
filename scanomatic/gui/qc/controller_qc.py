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
from scipy.optimize import leastsq
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter

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

    NORMALIZATION_PHENOTYPES = (phenotyper.Phenotyper.PHEN_GT_VALUE,)
    """
                            phenotyper.Phenotyper.PHEN_LAG,
                            phenotyper.Phenotyper.PHEN_YIELD)
    """

    def __init__(self, asApp=False, debugMode=False, parent=None):

        #PATHS NEED TO INIT BEFORE GUI
        self.paths = paths.Paths()

        if asApp:
            model = model_qc.Model.LoadAppModel()
            model['debug-mode'] = debugMode
            view = view_qc.Main_Window(controller=self, model=model)
        else:
            model = model_qc.Model.LoadStageModel()
            model['debug-mode'] = debugMode
            view = view_qc.QC_Dummy(controller=self, model=model)

        super(Controller, self).__init__(parent, view=view, model=model)
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

        self._logger.info("Loading {0}".format(pathToDirectory))
        try:
            p = phenotyper.Phenotyper.LoadFromState(pathToDirectory)
        except IOError:
            self._logger.error("Phenotyper reported IOError, bad path?")
            p = None

        if (p is None or p.source is None or p.phenotypes is None or
                p.times is None or p.smoothData is None or p.shape[0] == 0):

            self._logger.error("Project is corrupt or missing")
            return False

        self._model['reference-positions'] = None
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

        self._model['platesHaveUnsaved'][...] = False

        return True

    def plotData(self, fig):

        plate = self._model['plate']

        if (self._model['phenotyper'] is None or
                self._model['phenotyper'][plate] is None):

            self._logger.warning("No data for plate, why does it exist?")
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
                drawable=fig,
                clearDrawable=False)

    def getPhenotypes(self):

        return (self._model['phenotyper'] is None and dict() or
                self._model['phenotyper'].NAMES_OF_PHENOTYPES)

    def plotHeatmap(self, fig, newColorSpace=None):

        plate = self._model['plate']

        if (self._model['phenotyper'] is None or
                self._model['phenotyper'][plate] is None):
            self._view.get_stage().plotNoData(fig)
            return False

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

        if (data is not None and (data[plate] is None or data[plate].size == 0
                                  or data[plate].shape[-1] <= measure)
                or measure >= self._model['phenotyper'].nPhenotypesInData):

            self._logger.error(
                "Can't plot non-existing plate, maybe support for phenotype "
                " was added after extraction was done!")
            self._view.get_stage().plotNoData(fig)
            return False

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
        return True

    def toggleSelection(self, pos):

        pos = tuple(pos)
        p = self._model["plate_selections"]
        p[pos] = p[pos] is np.False_

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

        self._model['platesHaveUnsaved'][self._model['plate']] = True

    def undoLast(self):

        #TODO: unflag or something
        return False

    def getRecommendedFilter(self):

        lb = self._model['visibleMin']
        ub = self._model['visibleMax']
        return lb, ub, lb, ub

    def getBadness(self):

        wFit = 0.25
        p = self._model['phenotyper']
        pl = self._model['plate']

        gt = p.phenotypes[pl][..., p.PHEN_GT_VALUE]
        gtErr = p.phenotypes[pl][..., p.PHEN_GT_ERR]
        curvFit = p.phenotypes[pl][..., p.PHEN_FIT_VALUE]

        gtBar = gt.ravel()[np.isfinite(gt.ravel())].mean()

        """DEBUG
        if pos is not None and len(pos) == 2 and len(pos[0]) >= 0:

            print np.abs(gt[pos[0], pos[1]] - gtBar) / gtBar
            print (gtErr[pos[0], pos[1]]) * 100
            print wFit * (1 - curvFit.clip(0, 1)[pos[0], pos[1]]) * 100
        """

        return (np.abs(gt - gtBar) / gtBar + (gtErr) * 100 +
                wFit * (1 - curvFit.clip(0, 1)) * 100)

    def getPhenotypeBad(self, index=0):

        p = self._model['phenotyper']
        pl = self._model['plate']
        ph = self._model['phenotype']
        s = 1
        if ph == p.PHEN_FIT_VALUE:
            s = -1

        vals = p.phenotypes[pl][..., ph]
        return np.where(
            vals == vals.ravel()[vals.ravel().argsort()[::s][-index]])

    def getMostProbableBad(self, index=0):

        badness = self.getBadness()

        pos = np.where(badness == badness.ravel()[
            badness.ravel().argsort()[-index]])

        return pos

    def getBadnessIndexOfPos(self, sel):

        s = -1
        m = self._model
        ph = m['phenotype']

        if ph in m['badSortingPhenotypes']:

            if ph == m['phenotyper'].PHEN_FIT_VALUE:
                s = 1

            badness = m['phenotyper'].phenotypes[m['plate']][..., ph]

        else:

            badness = self.getBadness()

        pos = badness.argsort(None)[::s].argsort().reshape(
            badness.shape)[sel] + 1

        return pos

    def loadMetaData(self, paths):

        self._model['meta-data'] = meta_data.Meta_Data(
            self._model['plate_shapes'], *paths)

        self._model['phenotyper'].metaData = self._model['meta-data']

        self.guessBestColumn()

    def guessBestColumn(self):

        MD = self._model['meta-data']
        p = self._model['plate']
        CR = MD[p]

        hRow = MD.getHeaderRow(p)

        if CR.full == MD.PLATE_PARTIAL:

            rows = [
                CR(*pos) for pos in [(p, 0, 0), (p, 1, 0),
                                     (p, 0, 1), (p, 1, 1)]]

            allowedCols = min(len(r) for r in rows if rows is not None)

        else:

            allowedCols = len(hRow)

        hRow = hRow[:allowedCols]

        for i, h in enumerate(hRow):

            if h.lower().rstrip("s") in ['strain', 'specieis', 'organism',
                                         'construct', 'mutation', 'content']:

                self._model['meta-data-info-column'] = i
                self._model['meta-data-info-columnName'] = h
                return

        self._model['meta-data-info-column'] = 0
        self._model['meta-data-info-columnName'] = (len(hRow) > 0 and
                                                    hRow[0] or "")

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

    def getExperimentToReferenceCorrelation(self):

        s = self._model['norm-smoothing']
        if (s is None):
            s = 4

        Kref = self._model['reference-positions']
        phenotypes = self.getPhenotypesToBeNormed()


        ref = sub_plates.SubPlates(phenotypes, kernels=Kref)

        outcome = np.zeros((ref.shape[0], 4, ref.shape[-1])) * np.inf

        for idPl in xrange(outcome.shape[0]):

            expPos = zip(*np.where(Kref[idPl] == np.False_))

            for idSP, offset in enumerate(expPos):

                subPlate = phenotypes[idPl][offset[0]::2, offset[1]::2]

                for idPh in xrange(outcome.shape[-1]):

                    R = gaussian_filter(ref[idPl][..., idPh], sigma=s).ravel()
                    S = gaussian_filter(subPlate[..., idPh], sigma=s).ravel()
                    M = np.logical_and(np.isfinite(R), np.isfinite(S))

                    outcome[idPl, idSP, idPh] = pearsonr(R[M], S[M])[0]

        print outcome

        return np.ma.masked_invalid(outcome)


    def getPhenotypesToBeNormed(self, log=False):

        aCopy = []

        self._model['phenotyper'].padPhenotypes()

        for p in self._model['phenotyper'].phenotypes:

            if isinstance(p, np.ma.masked_array):
                aCopy.append(p[..., self.NORMALIZATION_PHENOTYPES].filled().copy())
            else:
                aCopy.append(p[..., self.NORMALIZATION_PHENOTYPES].copy())

        if log:
            aCopy = [np.log2(p) for p in aCopy]

        return data_bridge.Data_Bridge(np.array(aCopy))

    def normalize(self):

        m = self._model
        log = self._model['norm-alg-in-log']

        phenotypes = self.getPhenotypesToBeNormed(log=log)

        normInfo = {
            'ref-usage': np.ones((phenotypes.shape[0],)),
            'ref-usage-warning':
            np.zeros((phenotypes.shape[0],), dtype=np.bool),
            'ref-CV': np.zeros((phenotypes.shape[0],)),
            'ref-CV-warning':
            np.zeros((phenotypes.shape[0],), dtype=np.bool),
        }

        #TODO: Hack to check any
        if m['normByExperiment'].any():
            guidingSample = np.array([p is None and None or p.copy()
                for p in phenotypes])

            for idP in range(len(phenotypes)):
                if phenotypes[idP] is not None:
                    idX, idY = zip(*np.where(m['reference-positions'][idP]))[0]
                    guidingSample[idP][idX::2, idY::2] = np.nan
                            
        else:
            guidingSample = sub_plates.SubPlates(
                phenotypes, kernels=m['reference-positions'])

        #If using initial values in norm
        if m['norm-use-initial-values']:
            iVals = np.array([
                (p is None and None or
                 np.log2(p[..., [phenotyper.Phenotyper.PHEN_INIT_VAL_C]]))
                for p in m['phenotyper'].phenotypes])

            #np.save("qc_ivals.npy", iVals)

            if not log:
                phenotypes = np.array([p is None and None or np.log2(p) for p in
                                       phenotypes])

                log = True

            iParamGuesses = np.ones((len(iVals) * 2,), dtype=np.float)

        #If user has missed dubious positions they are filtered out
        if m['norm-outlier-iterations'] > 0:
            for idM in range(len(self.NORMALIZATION_PHENOTYPES)):
                norm.applyOutlierFilter(
                    guidingSample, measure=idM,
                    k=m['norm-outlier-k'],
                    p=m['norm-outlier-p'],
                    maxIterations=m['norm-outlier-iterations'])

            #If using initial values in norm pt2
            if m['norm-use-initial-values']:
                norm.applyOutlierFilter(
                    iVals, measure=0,
                    k=m['norm-outlier-k'],
                    p=m['norm-outlier-p'],
                    maxIterations=m['norm-outlier-iterations'])

                #np.save("qc_ivals_filt.npy", iVals)

            self._logger.info("Normalization: Outlier filter applied")
        else:
            self._logger.info("Normalization: Outlier filter skipped")

        for pId, plate in enumerate(guidingSample):

            rPlate = plate.ravel()
            rPlateFinite = rPlate[np.isfinite(rPlate)]

            normInfo['ref-usage'][pId] = rPlateFinite.size / \
                float(rPlate.size)

            normInfo['ref-usage-warning'][pId] = normInfo['ref-usage'][pId] < \
                m['norm-ref-usage-threshold']

            normInfo['ref-CV'][pId] = rPlateFinite.std() / rPlateFinite.mean()
            normInfo['ref-CV-warning'][pId] = normInfo['ref-CV'][pId] > \
                m['norm-ref-CV-threshold']

        #If using initial values in norm pt3
        if m['norm-use-initial-values']:
            iParams = leastsq(norm.IPVresidue, iParamGuesses,
                              args=(iVals, phenotypes))[0]
            #print "Scalings", iParams
            iPflex = iParams[: iParams.size / 2]
            iPscale = iParams[iParams.size / 2:]
            N = np.array([(p is None and None or
                           norm.initalPlateTransform(
                               p, iPflex[pId], iPscale[pId]))
                          for pId, p in enumerate(iVals)])

            """
            #Data array
            NA = norm.getControlPositionsArray(
                np.array([(p is None and None or
                           norm.initalPlateTransform(
                               p, iPflex[pId], iPscale[pId]))
                          for pId, p in enumerate(iVals)]),
                controlPositionKernel=subSampler.kernels)
            """

        else:

            #np.save("qc_debug_P.npy", phenotypes)

            #Data array
            #TODO: Hack to check any
            if m['normByExperiment'].any():
                NA = guidingSample
                #TODO: Hack, only checks removed on generation-time
                refPositions = [np.where(np.isfinite(
                    p[..., self.NORMALIZATION_PHENOTYPES[0]])) for p in NA]
            else:
                NA = norm.getControlPositionsArray(
                    phenotypes,
                    controlPositionKernel=guidingSample.kernels)
                refPositions = None

            #np.save("qc_debug_NA.npy", NA)

            #Get smothened norm surface
            N = norm.getNormalisationSurfaceWithGridData(
                NA, useAccumulated=False,
                controlPositionsCoordinates=refPositions,
                controlPositionKernel=m['reference-positions'],
                medianSmoothing=self._model['norm-outlier-fillSize'],
                gaussSmoothing=self._model['norm-smoothing'],
                normalisationSequence=self._model['norm-spline-seq'])

        if self._model['norm-outlier-fillSize'] is not None:
            self._logger.info("Normalization: Median filter applied")
        else:
            self._logger.info("Normalization: Median filter skipped")

        if self._model['norm-smoothing'] is None:
            self._logger.info("Normalization: Gauss smoothing skipped")
        else:
            self._logger.info("Normalization: Gauss smoothing applied")

        #Get normed values
        ND = norm.normalisation(phenotypes, N, updateBridge=False,
                                log=not log)

        if self._model['debug-mode']:

            newND = []
            for i, p in enumerate(ND):

                if p is None:
                    newND.append(p)
                else:
                    newND.append(np.insert(p, p.shape[-1], N[i], axis=-1))

            ND = np.array(newND)

        self._model['normalized-data'] = ND
        self._model['normalized-index-offset'] = len(
            phenotyper.Phenotyper.NAMES_OF_PHENOTYPES)
        self._model['normalized-phenotype-names'] = {
            i: phenotyper.Phenotyper.NAMES_OF_PHENOTYPES[k] for i, k in
            enumerate(self.NORMALIZATION_PHENOTYPES)}
        """
        i: name for i, (val, name) in enumerate(
            phenotyper.Phenotyper.NAMES_OF_PHENOTYPES.items())
        if val in self.NORMALIZATION_PHENOTYPES}
        """
        if self._model['debug-mode']:
            lNormed = len(self.NORMALIZATION_PHENOTYPES)
            self._model['normalized-phenotype-names'].update(
                {k + lNormed: "surface for " + v for k, v in
                 self._model['normalized-phenotype-names'].items()})

        self._view.get_stage().updateAvailablePhenotypes()

        print normInfo

        return normInfo
