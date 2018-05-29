class CorruptAnalysisError(Exception):
    pass


def get_times(project):
    raise CorruptAnalysisError()


def get_raw_curves(project, plate):
    raise CorruptAnalysisError()


def get_smooth_curves(project, plate):
    raise CorruptAnalysisError()
