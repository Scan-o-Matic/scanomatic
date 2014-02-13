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
import glob
import np
import re
import time

#
# INTERNAL DEPENDENCIES
#

import proc_effector
import scanomatic.io.logger as logger
import scanomatic.io.paths as paths
import scanomatic.PhenotypeExtractionessing.phenotyper as phenotyper

#
# CLASSES
#


class PhenotypeExtractionEffector(proc_effector.ProcEffector):

    def __init__(self):

        self._logger = logger.Logger("Data Processor")
        self._paths = paths.Paths()

        super(PhenotypeExtractionEffector, self).__init__()

        self._specificStatuses['progress'] = 'progress'
        self._specificStatuses['runTime'] = 'runTime'

        self._progress = 0

        self._allowStart = False
        self._running = False
        self._started = False
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

    def setup(self, path=None, phenotyperKwargs={}):

        def _perTime2perPlate(data):
            newData = [[]] * max(s.shape[0] for s in data)

            for scan in data:
                for pId, plate in enumerate(scan):
                    newData[pId].append(plate)

            for pId, plate in enumerate(newData):
                newData[pId] = np.array(plate)

            return np.array(newData)

        if self._started:

            self._logger.warning("Can't setup when started")
            return False

        if not os.path.isdir(os.path.dirname(path)):

            self._logger.error("Path '{0}' does not exist".format(
                os.path.abspath(os.path.dirname(path))))
            return False

        dirPath = os.path.dirname(path)
        baseName = os.path.baseName(path)

        if len(baseName) > 0 and "*" not in baseName:
            baseName += baseName
        elif len(baseName) == 0:
            baseName = self._paths.image_analysis_img_data.format("*")

        times = np.load(
            os.path.join(dirPath, self._paths.image_analysis_time_series))

        data = []
        timeIndex = []
        for p in glob.iglob(os.path.join(dirPath, baseName)):

            try:
                timeIndex = int(re.search(r"\d+", p).group())
                data.append(np.load(p))
            except AttributeError:
                self._logger(
                    "File '{0}' has no index number in it, need that!".format(
                        p))

        self._times = times[timeIndex]
        self._data = _perTime2perPlate(data)
        self._phenotyperKwargs = phenotyperKwargs
        self._analysisBase = dirPath

        self._allowStart = True

    def next(self):

        super(PhenotypeExtractionEffector, self).next()

        self._startTime = time.time()

        phenotyper.Phenotyper(dataObject=self._data,
                              timeObject=self._times,
                              itermode=True,
                              **self._phenotyperKwargs)

        phenoIter = phenotyper.iterAnalyse()

        while self._running:

            try:
                self._progress = phenoIter.next()
            except StopIteration:
                self._running = False
                self._progress = None
                break

            yield

            #
            # PAUSE IF REQUESTED
            #

            while self._paused and self._running:

                time.sleep(0.5)
                yield
