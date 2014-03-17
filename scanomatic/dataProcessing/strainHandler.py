"""Works with strains"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
#   DEPENDENCIES
#

import numpy as np

#
#
#


def uniques(metaData, slicer=None):
    """Generates a dictionary lookup for unique strains and the positions
    they occur in.

    Args:
        metaData (meta_data.Meta_Data): Instance of metadata

    Kwargs:
        slicer: Either a slice or an argument passable to slice for
                subselecting columns of metadata that is to be evaluated
                for uniqueness

    Returns:
        dict    A dictionary with unique keys and lists of positions that they
                occur on
    """
    _uniques = dict()
    if not isinstance(slicer, slice):
        slicer = slice(slicer)

    for pos in metaData.generateCoordinates():

        strain = tuple(metaData(*pos))[slicer]

        if strain not in _uniques:
            _uniques[strain] = [pos]
        else:
            _uniques[strain].append(pos)

    return _uniques


def generalStatsOnStrains(uniqueDict, dataObject, measure=None):
    """Collects basic stats on strains independent on plate (if not part
    of strain info). And presents their basic statistics.

    Args:
        uniqueDict (dict):  A dict containing unique identifiers and position
                            lists as returned by the unique method

        dataObject (several):   An object exposing basic numpy array interface
                                and hold relevant position based data.

    Kwargs:

        measure:    Either a slice or an argument passable to slice for
                    subselecting measure type that is to be evaluated
                    for uniqueness

    Returns:

        Dict of dicts that hold relevant stats per strain
    """
    if not isinstance(measure, slice):
        measure = slice(measure)

    _stats = dict()
    easeMode = dataObject.ndim == 3

    for strain, positions in uniqueDict.items():

        if easeMode:
            vals = dataObject[
                tuple(map(np.array, zip(*positions)))][..., measure]
        else:
            vals = []
            for p in np.unique(zip(*positions)[0]):
                vals += dataObject[p][
                    tuple(map(np.array, zip(*positions)[1:]))].tolist()

            vals = np.array(vals)[..., measure]

        finVals = np.isfinite(vals)

        _stats[strain] = dict(
            n=vals.size,
            nans=vals.size - finVals.sum(),
            mean=vals[finVals].mean(),
            std=vals[finVals].std())
