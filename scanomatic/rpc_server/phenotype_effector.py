"""The master effector of data processing downstream of image analysis"""
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

import os
import time

#
# INTERNAL DEPENDENCIES
#

import proc_effector
import scanomatic.io.logger as logger
import scanomatic.io.paths as paths
import scanomatic.io.image_data as image_data
import scanomatic.dataProcessing.phenotyper as phenotyper

#
# CLASSES
#


class PhenotypeExtractionEffector(proc_effector.ProcEffector):

    def __init__(self, identifier, label):

        self._logger = logger.Logger("Data Processor '{0}'".format(label))
        self._paths = paths.Paths()

        super(PhenotypeExtractionEffector, self).__init__(identifier, label)

        self._specificStatuses['progress'] = 'progress'
        self._specificStatuses['runTime'] = 'runTime'

        self._progress = 0

        self._startTime = None

    @property
    def runTime(self):

        if self._startTime is None:
            return 0
        else:
            return time.time() - self._startTime

    @property
    def progress(self):

        return self._progress is None and 1 or self._progress

    def setup(self, *lostArgs, **phenotyperKwargs):

        if self._started:

            self._logger.warning("Can't setup when started")
            return False

        path = None

        if "runDirectory" in phenotyperKwargs:
            path = phenotyperKwargs["runDirectory"]
            del phenotyperKwargs["runDirectory"]

        if path is None or not os.path.isdir(os.path.dirname(path)):

            self._logger.error("Path '{0}' does not exist".format(
                path is None and 'NONE' or
                os.path.abspath(os.path.dirname(path))))
            return False

        if (len(lostArgs) > 0):
            self._logger.warning("Setup got unknown args {0}".format(
                lostArgs))

        times, data = image_data.Image_Data.readImageDataAndTimes(path)

        if None in (times, data):
            self._logger.error(
                "Could not filter image times to match data or no data. " +
                "Do you have the right directory?")
            self.addMessage("There is no image data in given directory or " +
                            "the image data is corrupt")
            self._running = False
            self._stopping = True
            return None

        self._times = times
        self._data = data
        self._phenotyperKwargs = phenotyperKwargs
        self._analysisBase = image_data.Image_Data.path2dataPathTuple(path)[0]

        #DEBUG CODE
        import numpy as np
        np.save(os.path.join(self._analysisBase, "debug.npy"), self._data)
        np.save(os.path.join(self._analysisBase, "debugTimes.npy"), self._times)

        self._allowStart = True

    def next(self):

        if not self._allowStart:
            return super(PhenotypeExtractionEffector, self).next()

        if self._iteratorI == 0:
            self._startTime = time.time()

            curPhenotyper = phenotyper.Phenotyper(
                dataObject=self._data,
                timeObject=self._times,
                itermode=True,
                **self._phenotyperKwargs)

            self._phenoIter = curPhenotyper.iterAnalyse()
            self._curPhenotyper = curPhenotyper
            self._iteratorI = 1
            self._logger.info("Starting phenotype extraction")

        if (not self._paused and self._running):
            try:
                self._progress = self._phenoIter.next()
            except StopIteration:
                self._running = False
                self._progress = None
            self._logger.info(
                "One phenotype extraction iteration completed. " +
                "Resume {0}".format(self._running))

        if (not self._running):
            self._curPhenotyper.savePhenotypes(
                path=os.path.join(self._analysisBase,
                                  self._paths.phenotypes_raw_csv),
                askOverwrite=False)

            self._curPhenotyper.saveState(self._analysisBase,
                                          askOverwrite=False)

            raise StopIteration
