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

        model = model_qc.load_app_model()
        if asApp:
            view = view_qc.Main_Window(controller=self, model=model)
        else:
            view = view_qc.Stage(controller=self, model=model)

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

    def plotData(self, fig, plate, position):

        if (self._model['phenotyper'] is None or
                self._model['phenotyper'][plate] is None):
            self._plotNoData(fig)
            return

        self._model['phenotyper'].plotACurve(
            (plate, ) + position,
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

    def plotHeatmap(self, fig, plate):

        if (self._model['phenotyper'] is None or
                self._model['phenotyper'][plate] is None):
            self._plotNoData(fig)
            return
