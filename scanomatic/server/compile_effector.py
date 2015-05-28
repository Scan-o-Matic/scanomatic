__author__ = 'martin'

import os
import proc_effector
from scanomatic.models.compile_project_model import FIXTURE, COMPILE_ACTION
from scanomatic.io.fixtures import Fixtures, FixtureSettings
from scanomatic.io.paths import Paths
from scanomatic.imageAnalysis import first_pass
from scanomatic.models.factories.fixture_factories import FixtureFactory


class CompileProjectEffector(proc_effector.ProcessEffector):

    def __init__(self, job):

        """

        :type job: scanomatic.models.rpc_job_models.RPCjobModel
        """

        super(CompileProjectEffector, self).__init__(job, logger_name="Compile Effector")
        self._compile_job = job.content_model
        """:type : scanomatic.models.compile_project_model.CompileInstructionsModel"""

        self._image_to_analyse = 0
        self._fixture = None

        self._allowed_calls['progress'] = self.progress

    @property
    def progress(self):
        """:rtype : float"""
        return self._image_to_analyse / float(len(self._compile_job.images))

    def setup(self, compile_job):

        self._tweak_path()
        self._load_fixture()
        self._allow_start = True

    def _load_fixture(self):

        if self._compile_job.fixture is FIXTURE.Global:
            self._fixture = Fixtures[self._compile_job.fixture_name]
        else:
            self._fixture = FixtureSettings(
                Paths().experiment_local_fixturename,
                dir_path=os.path.dirname(self._compile_job.path))

    def _tweak_path(self):

        self._compile_job.path = Paths().get_project_compile_path_from_compile_model(self._compile_job)

    def next(self):

        if not self._allow_start:
            return super(CompileProjectEffector, self).next()

        if self._stopping:
            raise StopIteration()
        elif self._image_to_analyse < len(self._compile_job.images):

            self._analyse_image(self._compile_job.images[self._image_to_analyse])
            self._image_to_analyse += 1
            return True

        elif (self._compile_job.compile_action is COMPILE_ACTION.AppendAndSpawnAnalysis or
                self._compile_job.compile_action is COMPILE_ACTION.InitiateAndSpawnAnalysis):

            return self._spawn_analysis()

        else:

            raise StopIteration()

    def _analyse_image(self, compile_image_model):

        """

        :type compile_image_model: scanomatic.models.compile_project_model.CompileImageModel
        """

        try:

            with self._compile_output_filehandle as fh:

                self._logger.warning("Not yet implemented first pass analysis, skipping {0}".format(
                    compile_image_model))

                try:
                    image_model = first_pass.analyse(compile_image_model, self._fixture)
                    FixtureFactory.serializer.dump_to_filehandle(image_model, fh)

                except first_pass.MarkerDetectionFailed:

                    self._logger.error("Failed to detect the markers on {0} using fixture {1}".format(
                        compile_image_model.path, self._fixture['path']))
                except IOError:

                    self._logger.error("Could not output analysis to file {0}".format(
                        compile_image_model.path))
        except IOError:

            self._logger.critical("Could not write to project file {0}".format(self._compile_job.path))

    @property
    def _compile_output_filehandle(self):

        fh_mode = 'a'

        if ((self._compile_job.compile_action is COMPILE_ACTION.Initiate or
                self._compile_job.compile_action is COMPILE_ACTION.InitiateAndSpawnAnalysis) and
                self._image_to_analyse == 0):

            fh_mode = 'w'

        try:

            return open(self._compile_job.path, fh_mode)

        except IOError, e:

            self._stopping = True
            raise e

    def _spawn_analysis(self):

        self._logger.warning("Analysis spawning not implemented yet.")
        return False