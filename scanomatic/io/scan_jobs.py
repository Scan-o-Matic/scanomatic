class ScanNameCollision(ValueError):
    pass


class ScanNameUnknown(ValueError):
    pass


class ScanJobs:
    def __init__(self):
        self._db = {}

    def add_job(self, identifier, job):
        if identifier in self._db:
            raise ScanNameCollision("{} already used".format(identifier))
        self._db[identifier] = job

    def remove_job(self, identifier):
        if identifier in self._db:
            del self._db[identifier]
        else:
            raise ScanNameUnknown("{} is not a known job".format(identifier))

    def get_jobs(self):
        return self._db.values()

    def get_job_ids(self):
        return self._db.keys()

    def exists_job_with(self, key, value):
        for job in self._db.values():
            if key in job and job[key] == value:
                return True
        return False
