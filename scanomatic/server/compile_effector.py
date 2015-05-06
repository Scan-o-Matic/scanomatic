__author__ = 'martin'

import os
import proc_effector
from scanomatic.models.compile_project_model import FIXTURE
from scanomatic.io.fixtures import Fixtures, Fixture_Settings
from scanomatic.io.paths import Paths


class CompileProjectEffector(proc_effector.ProcessEffector):

    def __init__(self, job):

        super(CompileProjectEffector, self).__init__(job, logger_name="Compile Effector")
        self._compile_job = job.content_model
        self._image_to_analyse = 0
        self._fixture = None

        self._allowed_calls['progress'] = self.progress

    @property
    def progress(self):

        return self._image_to_analyse / float(len(self._compile_job.images))

    def setup(self, compile_job):

        self._load_fixture()
        self._allow_start = True

    def next(self):

        if not self._allow_start:
            return super(CompileProjectEffector, self).next()

        raise StopIteration()

    def _load_fixture(self):

        if self._compile_job.fixture is FIXTURE.Global:
            self._fixture = Fixtures[self._compile_job.fixture_name]
        else:
            self._fixture = Fixture_Settings(os.path.dirname(self._compile_job.path),
                                             Paths().experiment_local_fixturename)
