import os
import time
import proc_effector
from scanomatic.models.compile_project_model import FIXTURE, COMPILE_ACTION
from scanomatic.io.fixtures import Fixtures, FixtureSettings
from scanomatic.io.paths import Paths
from scanomatic.image_analysis import first_pass
from scanomatic.models.factories.compile_project_factory import CompileImageAnalysisFactory, CompileProjectFactory
from scanomatic.models.factories.rpc_job_factory import RPC_Job_Model_Factory
from scanomatic.models.rpc_job_models import JOB_TYPE
import scanomatic.io.rpc_client as rpc_client
from scanomatic.io.app_config import Config as AppConfig
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory


class CompileProjectEffector(proc_effector.ProcessEffector):

    TYPE = JOB_TYPE.Compile

    def __init__(self, job):

        """

        :type job: scanomatic.models.rpc_job_models.RPCjobModel
        """

        super(CompileProjectEffector, self).__init__(job, logger_name="Compile Effector")
        self._compile_job = job.content_model
        """:type : scanomatic.models.compile_project_model.CompileInstructionsModel"""
        self._job_label = os.path.basename(self._compile_job.path)
        self._image_to_analyse = 0
        self._fixture_settings = None
        self._compile_instructions_path = None
        self._has_mailed_issues = False
        self._allowed_calls['progress'] = self.progress

    @property
    def progress(self):
        """:rtype : float"""
        if self._compile_job.images:
            return self._image_to_analyse / float(len(self._compile_job.images))
        return 0

    def setup(self, job):

        self._logger.info("Setup called")

        self._compile_job = RPC_Job_Model_Factory.serializer.load_serialized_object(job)[0].content_model
        self._job.content_model = self._compile_job

        if self._compile_job.images is None:
            self._compile_job.images = tuple()

        log_path = Paths().get_project_compile_log_path_from_compile_model(self._compile_job)
        self._logger.set_output_target(log_path, catch_stdout=True, catch_stderr=True)
        self._logger.surpress_prints = True
        self._log_file_path = log_path

        self._logger.info("Doing setup")
        self._logger.info("Action {0}".format(self._compile_job.compile_action))
        self._compile_instructions_path = Paths().get_project_compile_instructions_path_from_compile_model(
            self._compile_job)
        if self._compile_job.compile_action == COMPILE_ACTION.Initiate or \
                        self._compile_job.compile_action == COMPILE_ACTION.InitiateAndSpawnAnalysis:

            # Empty files
            try:
                os.remove(self._compile_instructions_path)
            except OSError:
                pass

            CompileProjectFactory.serializer.dump(self._compile_job, self._compile_instructions_path)

        self._logger.info("{0} Images included in compilations".format(len(self._compile_job.images)))
        self._tweak_path()
        self._load_fixture()
        self._allow_start = True

        if self._fixture_settings is None:
            self._logger.critical("No fixture loaded, name probably not recognized or old fixture settings file")
            self._stopping = True

        self._start_time = time.time()

    def _load_fixture(self):

        if self._compile_job.fixture_type is FIXTURE.Global:
            self._fixture_settings = Fixtures()[self._compile_job.fixture_name]
            if self._fixture_settings and \
                    self._compile_job.compile_action in (COMPILE_ACTION.Initiate,
                                                         COMPILE_ACTION.InitiateAndSpawnAnalysis):

                self._fixture_settings.update_path_to_local_copy(os.path.dirname(self._compile_job.path))
                self._fixture_settings.save()

        else:
            dir_path = os.path.dirname(self._compile_job.path)
            self._logger.info("Attempting to load local fixture copy in directory {0}".format(dir_path))
            self._fixture_settings = FixtureSettings(
                Paths().experiment_local_fixturename,
                dir_path=dir_path)

    def _tweak_path(self):

        self._compile_job.path = Paths().get_project_compile_path_from_compile_model(self._compile_job)

    def next(self):

        if self.waiting:
            return super(CompileProjectEffector, self).next()

        if self._stopping:
            raise StopIteration()
        elif self._image_to_analyse < len(self._compile_job.images):

            self._analyse_image(self._compile_job.images[self._image_to_analyse])
            self._image_to_analyse += 1
            return True

        elif (self._compile_job.compile_action is COMPILE_ACTION.AppendAndSpawnAnalysis or
                self._compile_job.compile_action is COMPILE_ACTION.InitiateAndSpawnAnalysis):

            self._spawn_analysis()
            self.enact_stop()

        else:

            self.enact_stop()

    def _analyse_image(self, compile_image_model):

        """

        :type compile_image_model: scanomatic.models.compile_project_model.CompileImageModel
        """

        try:

            with self._compile_output_filehandle as fh:
                issues = {}
                try:

                    image_model = first_pass.analyse(compile_image_model, self._fixture_settings, issues=issues)
                    CompileImageAnalysisFactory.serializer.dump_to_filehandle(
                        image_model, fh, as_if_appending=True)

                except first_pass.MarkerDetectionFailed:

                    self._logger.error("Failed to detect the markers on {0} using fixture {1}".format(
                        compile_image_model.path, self._fixture_settings.model.path))
                except IOError:

                    self._logger.error("Could not output analysis to file {0}".format(
                        compile_image_model.path))

                if issues and not self._has_mailed_issues:
                    self._mail_issues(issues)

        except IOError:

            self._logger.critical("Could not write to project file {0}".format(self._compile_job.path))

    def _mail_issues(self, issues):
        self._has_mailed_issues = True
        self._mail("Scan-o-Matic: Problems compiling project '{path}'",
                   """This is an automated email, please don't reply!

The project '{path}' on """ + AppConfig().computer_human_name +
                   """ has experienced a problems with compiling the current project.
                   """
                   "* Image seems rotated ({0} radians).\n".format(issues['rotation']) if 'rotation' in issues else ""
                   "* Part of calibrated fixture falls outside image ({0}).\n".format(
                       issues['overflow'] if isinstance(issues['overflow'], str)
                       else "Plate {0}".format(issues["overflow"])) if 'overflow' in issues else ""
                   """

All the best,

Scan-o-Matic""", self._compile_job)

    @property
    def _compile_output_filehandle(self):

        fh_mode = 'r+w'

        if ((self._compile_job.compile_action is COMPILE_ACTION.Initiate or
                self._compile_job.compile_action is COMPILE_ACTION.InitiateAndSpawnAnalysis) and
                self._image_to_analyse == 0):

            fh_mode = 'w'

        try:

            return open(self._compile_job.path, fh_mode)

        except IOError, e:

            self._stopping = True
            raise e

    def enact_stop(self):

        self._stopping = True

        self._mail("Scan-o-Matic: Compilation of '{path}' completed",
                       """This is an automated email, please don't reply!

The project '{path}' on """ + AppConfig().computer_human_name +
                       """ has completed. No downstream analysis requested.

All the best,

Scan-o-Matic""", self._compile_job)

        raise StopIteration

    def _spawn_analysis(self):

        analysis_model = AnalysisModelFactory.create(
            chain=True,
            compile_instructions=self._compile_instructions_path,
            compilation=self._compile_job.path,
            email=self._compile_job.email,
            cell_count_calibration_id=\
                self._compile_job.cell_count_calibration_id,
        )

        if self._compile_job.overwrite_pinning_matrices:
            analysis_model.pinning_matrices = self._compile_job.overwrite_pinning_matrices

        if analysis_model.pinning_matrices and \
                len(analysis_model.pinning_matrices) != len(self._fixture_settings.model.plates):

            self._logger.warning("Suggested pinnings are {0} but known plates are {1}".format(
                analysis_model.pinning_matrices, [p.index for p in self._fixture_settings.model.plates]
            ))

            accepted_indices = tuple(p.index for p in self._fixture_settings.model.plates)
            analysis_model.pinning_matrices = [pm for i, pm in enumerate(analysis_model.pinning_matrices)
                                               if i + 1 in accepted_indices]
            self._logger.warning(
                "Reduced pinning matrices instructions to {0} because lacking plates in fixture".format(
                    analysis_model.pinning_matrices))

        if rpc_client.get_client(admin=True).create_analysis_job(AnalysisModelFactory.to_dict(analysis_model)):
            self._logger.info("Enqueued analysis")

            return True

        else:

            self._mail("Scan-o-Matic: Compilation of '{path}' completed",
                       """This is an automated email, please don't reply!

The project '{path}' on """ + AppConfig().computer_human_name +
                       """ has completed.

*** Downstream analysis was refused. ***

All the best,

Scan-o-Matic""", self._compile_job)

            self._logger.warning("Enquing analysis was refused")
            return False
