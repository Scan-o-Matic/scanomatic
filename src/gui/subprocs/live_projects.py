#!/usr/bin/env python
"""The Live Projects"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os
import ConfigParser
import inspect
import logging

#
# EXCEPTIONS
#


class InvalidStageOrStatus(Exception):
    pass


class InvalidProjectOrStage(Exception):
    pass


#
# METHODS
#


def whoCalled(fn):

    def wrapped(*args, **kwargs):
        frames = []
        frame = inspect.currentframe().f_back
        while frame.f_back:
            frames.append(inspect.getframeinfo(frame)[2])
            frame = frame.f_back
        frames.append(inspect.getframeinfo(frame)[2])

        print "===\n{0}\n{1}\n{2}\nCalled by {3}\n____".format(
            fn, args, kwargs, ">".join(frames[::-1]))

        fn(*args, **kwargs)

    return wrapped

#
# CLASS
#


class Live_Projects(object):

    #STAGES
    EXPERIMENT = 0
    ANALYSIS = 1
    INSPECT = 2
    UPLOAD = 3
    STAGES = ('EXPERIMENT', 'ANALYSIS', 'INSPECT', 'UPLOAD')

    #STAGE STATUSES
    FAILED = -1
    NOT_YET = 0
    AUTOMATIC = 1
    LAUNCH = 2
    RUNNING = 3
    TERMINATED = 4
    COMPLETED = 5
    STATUSES = ('FAILED', 'NOT_YET', 'AUTOMATIC', 'LAUNCH', 'RUNNING',
                'TERMINATED', 'COMPLETED')

    def __init__(self, paths, model):

        self._logger = logging.getLogger("Live Projects")
        self._config = ConfigParser.ConfigParser(allow_no_value=True)
        self._paths = paths
        self._model = model

        self._count = 0
        self._load()

    def _load(self):

        try:
            self._config.read(self._paths.log_project_progress)
        except:
            pass

        self._count = len(self._config.sections())

    def _save(self):

        with open(self._paths.log_project_progress, 'wb') as configfile:
                self._config.write(configfile)

    def add_project(self, project_prefix, experiment_dir,
                    first_pass_file=None, analysis_path=None):

        if project_prefix not in self._config.sections():
            self._config.add_section(project_prefix)

        if first_pass_file is None:
            first_pass_file = \
                self._paths.experiment_first_pass_analysis_relative.format(
                    project_prefix)

        self._config.set(project_prefix, 'basedir', experiment_dir)
        self._config.set(project_prefix, '1st_pass_file', first_pass_file)
        self._config.set(project_prefix, 'analysis_path', analysis_path)
        self._config.set(project_prefix, str(self.EXPERIMENT), "0")
        self._config.set(project_prefix, str(self.ANALYSIS), "0")
        self._config.set(project_prefix, str(self.INSPECT), "0")
        self._config.set(project_prefix, str(self.UPLOAD), "0")
        self._save()

        self._count += 1

    def get_is_project(self, project_prefix):

        if self._config.has_section(project_prefix) is False:
            return False

        for item in ('basedir', '1st_pass_file', 'analysis_path',
                     str(self.EXPERIMENT), str(self.ANALYSIS),
                     str(self.INSPECT), str(self.UPLOAD)):

            if self._config.has_option(project_prefix, item) is False:

                return False

        return True

    def _resolve_name(self, ref_options, val):

        if isinstance(val, str):

            if val in ref_options:

                try:

                    val = getattr(self, val)

                except:

                    raise InvalidStageOrStatus(
                        "Internal modul error missmatch, " +
                        "{0} should exist but doesn't.".format(
                            val))

                    val = None

            else:

                raise InvalidStageOrStatus(
                    "{0} not valid name ({1})".format(
                        val, ref_options))

                val = None

        else:

            found_it = False

            for opt in ref_options:

                if getattr(self, opt) == val:
                    found_it = True
                    break

            if found_it is False:

                val = None

                raise InvalidStageOrStatus(
                    "The value {0} is not allowed".format(val))

        return val

    def _resolve_stage(self, stage):

        return self._resolve_name(self.STAGES, stage)

    def _resolve_status(self, status):

        return self._resolve_name(self.STATUSES, status)

    def set_status(self, project_prefix, stage, status, experiment_dir=None,
                   first_pass_file=None, analysis_path=None):

        stage_num = self._resolve_stage(stage)
        status_num = self._resolve_status(status)

        if stage_num is not None and status_num is not None:

            if self.get_is_project(project_prefix) is False:
                self.add_project(project_prefix, experiment_dir,
                                 first_pass_file, analysis_path)

            self._logger.info("Updating status for {0}, stage {1} to {2}".format(
                project_prefix, stage, status))

            self._config.set(project_prefix, str(stage_num), str(status_num))
            self._save()

            if stage_num == self.UPLOAD and status_num == self.COMPLETED:

                self._logger.info("Project {0} is completed".format(
                    project_prefix))
                self.clear_done_projects()

        else:

            self._logger.warning("Bad status update request for {0}".format(
                project_prefix) +
                "Stage {0}, status {1}".format(stage, status))
        return True

    def get_status(self, project_prefix, stage, as_text=True,
                   supress_load=False):

        if supress_load is False:
            self._load()

        try:
            if type(stage) == int:
                val = self._config.getint(project_prefix, str(stage))
            else:
                val = self._config.getint(
                    project_prefix,
                    str(eval("self." + stage)))

        except:
            raise InvalidProjectOrStage("{0} {1}".format(project_prefix, stage))

        if as_text:
            return self._model['project-progress-stage-status'][val]
        else:
            return val

    def get_all_status(self, project_prefix, as_text=True, supress_load=False):

        if supress_load is False:
            self._load()
        ret = []
        for stage in range(self.UPLOAD + 1):
            ret.append(self.get_status(project_prefix, stage,
                       as_text=as_text, supress_load=True))

        return ret

    def get_all_stages_status(self, as_text=True):

        self._load()
        ret = {}
        for project in self._config.sections():
            ret[project] = self.get_all_status(project, as_text=as_text,
                                               supress_load=True)

        return ret

    def remove_project(self, project_prefix, supress_save=False):

        self._config.remove_section(project_prefix)
        if supress_save is False:
            self._save()

    def clear_done_projects(self):

        projects = self.get_all_stages_status(as_text=False)

        for project, stages in projects.items():
            if not(False in [status > self.RUNNING for status in stages]):
                self.remove_project(project, supress_save=True)

        self._save()

    def get_project_count(self):

        return self._count

    def get_path(self, project, supress_load=False):

        if supress_load is False:
            self._load()
        return self._config.get(project, 'basedir')

    def get_analysis_path(self, project):

        self._load()
        analysis_path = os.path.join(
            self.get_path(project, supress_load=True),
            self._config.get(project, 'analysis_path'))

        return analysis_path

    def get_first_pass_file(self, project):

        self._load()
        first_pass_file = os.path.join(
            self.get_path(project, supress_load=True),
            self._config.get(project, '1st_pass_file'))

        return first_pass_file
