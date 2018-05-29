from scanomatic.data_processing import analysis_loader


class CorruptAnalysisError(Exception):
    pass


def get_times(project):
    try:
        times = analysis_loader.AnalysisLoader(project).times
    except (IOError, ValueError):
        raise CorruptAnalysisError()
    return times


def get_raw_curves(project, plate):
    try:
        curves = analysis_loader.AnalysisLoader(project).raw_growth_data
    except (IOError, ValueError):
        raise CorruptAnalysisError()
    if curves.ndim > 0 and curves.shape[0] > plate:
        return curves[plate]
    raise CorruptAnalysisError()


def get_smooth_curves(project, plate):
    try:
        curves = analysis_loader.AnalysisLoader(project).smooth_growth_data
    except (IOError, ValueError):
        raise CorruptAnalysisError()
    if curves.ndim > 0 and curves.shape[0] > plate:
        return curves[plate]
    raise CorruptAnalysisError()
