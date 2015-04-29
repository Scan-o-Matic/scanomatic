__author__ = 'martin'

import proc_effector


class CompileProjectEffector(proc_effector.ProcessEffector):

    def __init__(self, job):

        super(CompileProjectEffector, self).__init__(job, logger_name="Compile Effector")
        self._compile_job = job.content_model
        self._image_to_analyse = 0
        self._allowed_calls['progress'] = self.progress

    @property
    def progress(self):

        return self._image_to_analyse / float(len(self._compile_job.images))

    def setup(self, compile_job):

        self._allow_start = True

    def next(self):

        if not self._allow_start:
            return super(CompileProjectEffector, self).next()