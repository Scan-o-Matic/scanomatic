from __future__ import absolute_import
from collections import namedtuple


class ScanJobCollisionError(ValueError):
    pass


class ScanJobUnknownError(ValueError):
    pass


ScanJob = namedtuple(
    'ScanJob',
    ['identifier', 'name', 'duration', 'interval', 'scanner']
)


class ScanJobs:
    def __init__(self):
        self._db = {}

    def add_job(self, job):
        if job.identifier in self._db:
            raise ScanJobCollisionError(
                "{} already used".format(job.identifier)
            )
        self._db[job.identifier] = job

    def remove_job(self, identifier):
        if identifier in self._db:
            del self._db[identifier]
        else:
            raise ScanJobUnknownError(
                "{} is not a known job".format(identifier)
            )

    def get_jobs(self):
        return list(self._db.values())

    def get_job_ids(self):
        return list(self._db.keys())

    def exists_job_with(self, key, value):
        for job in self._db.values():
            if getattr(job, key) == value:
                return True
        return False
