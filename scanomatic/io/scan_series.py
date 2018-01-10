class ScanNameCollision(ValueError):
    pass


class ScanNameUnknown(ValueError):
    pass


class ScanSeries:
    def __init__(self):
        self._db = {}

    def add_job(self, name, job):
        if name in self._db:
            raise ScanNameCollision("{} already used".format(name))
        self._db[name] = job

    def remove_job(self, name):
        if name in self._db:
            del self._db[name]
        else:
            raise ScanNameUnknown("{} is not a known job".format(name))

    def get_jobs(self):
        return self._db.values()

    def get_job_names(self):
        return self._db.keys()
