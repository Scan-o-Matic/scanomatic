__DB = {}


class ScanNameCollision(ValueError):
    pass


class ScanNameUnknown(ValueError):
    pass


def add_job(name, duration, interval):
    if name in __DB:
        raise ScanNameCollision("{} already used".format(name))
    __DB[name] = {"duration": duration, "interval": interval, "name": name}


def remove_job(name):
    if name in __DB:
        del __DB[name]
    else:
        raise ScanNameUnknown("{} is not a known job".format(name))


def get_jobs():
    return __DB.values()
