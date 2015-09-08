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
import scanomatic.io.paths as paths
import scanomatic.io.image_data as image_data
import scanomatic.dataProcessing.phenotyper as phenotyper
from scanomatic.models.rpc_job_models import JOB_TYPE
import scanomatic.models.factories.features_factory as feature_factory
from scanomatic.models.factories.rpc_job_factory import RPC_Job_Model_Factory

#
# CLASSES
#


class PhenotypeExtractionEffector(proc_effector.ProcessEffector):
    TYPE = JOB_TYPE.Features

    def __init__(self, job):

        self._paths = paths.Paths()

        super(PhenotypeExtractionEffector, self).__init__(job, logger_name="Phenotype Extractor '{0}'".format(job.id))

        self._feature_job = job.content_model
        self._job_label = self._feature_job.analysis_directory
        self._progress = 0
        self._times = None
        self._data = None
        self._analysis_base_path = None
        self._phenotyper = None

    @property
    def progress(self):

        return self._progress is None and 1 or self._progress

    def setup(self, job):

        job = RPC_Job_Model_Factory.serializer.load_serialized_object(job)[0]
        if self._started:
            self._logger.warning("Can't setup when started")
            return False

        if feature_factory.FeaturesFactory.validate(self._feature_job) is not True:
            self._logger.warning("Can't setup, instructions don't validate")
            return False

        self._logger.info("Loading files image data from '{0}'".format(
            self._feature_job.analysis_directory))

        times, data = image_data.ImageData.read_image_data_and_time(self._feature_job.analysis_directory)

        if None in (times, data):
            self._logger.error(
                "Could not filter image times to match data or no data. " +
                "Do you have the right directory?")

            self.add_message("There is no image data in given directory or " +
                             "the image data is corrupt")

            self._running = False
            self._stopping = True
            return False

        self._times = times
        self._data = data
        self._analysis_base_path = image_data.ImageData.directory_path_to_data_path_tuple(self._feature_job.analysis_directory)[0]

        """
        # DEBUG CODE
        import numpy as np
        np.save(os.path.join(self._analysisBase, "debug.npy"), self._data)
        np.save(os.path.join(self._analysisBase, "debugTimes.npy"), self._times)
        """

        self._allow_start = True

    def next(self):

        if self.waiting:
            return super(PhenotypeExtractionEffector, self).next()

        if self._stopping:
            self._progress = None
            self._running = False

        if self._iteration_index is None:
            self._setup_extraction_iterator()

        if not self._paused and self._running:
            try:
                self._progress = self._phenotype_iterator.next()
            except StopIteration:
                self._running = False
                self._progress = None
            self._logger.info(
                "One phenotype extraction iteration completed. " +
                "Resume {0}".format(self._running))

        if not self._running:
            if not self._stopping:
                self._phenotyper.savePhenotypes(
                    path=os.path.join(self._analysis_base_path,
                                      self._paths.phenotypes_raw_csv),
                    askOverwrite=False)

                self._phenotyper.saveState(self._analysis_base_path,
                                           askOverwrite=False)

            raise StopIteration

    def _setup_extraction_iterator(self):

        self._start_time = time.time()

        self._phenotyper = phenotyper.Phenotyper(
            dataObject=self._data,
            timeObject=self._times,
            itermode=True)

        self._phenotype_iterator = self._phenotyper.iterAnalyse()
        self._iteration_index = 1
        self._logger.info("Starting phenotype extraction")