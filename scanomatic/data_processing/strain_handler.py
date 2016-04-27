import numpy as np
import matplotlib.pyplot as plt

#
#   INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger

#
#   GLOBALS
#

_L = logger.Logger("Strain Handler Module")

_PLATE_STRING = "Plate {0}"

#
#   MEMEBER METHODS
#


def loadCSV2Numpy(path, measure=-1, delim='\t', dtype=np.float):
    """Loads a csv-file produced by QC as a numpy array of plates.

    Args:

        path (string):  The path to the csv-file

    Kwargs:

        measure (str, int, slice):

            If `int`, the index of the column to be used to produce the
            array
            If `slice`, the slice of the data vector for each row to be used
            in the array.
            If `string`, the name of the column header

        delim (string):

            Column delimiter in the csv

        dtype (type):

            Type of data, default is `numpy.float`, if it is, unfilled values
            will be set to nan, else they will be zero of that data type

    """

    def _putInData(d, pos, m):

        plateI, xI, yI = map(int, pos)
        if plateI not in d:
            d[plateI] = dict()
        d[plateI][(xI, yI)] = map(dtype, m)

    fs = open(path, 'r')

    h = fs.readline().split(delim)

    if isinstance(measure, str):

        for i, colH in enumerate(h):

            if colH == measure:
                measure = slice(i, i + 1)
                break

        if isinstance(measure, str):

            fs.close()
            _L.error(
                "{0}: Could not find {1} among headers {2}".format(
                    path, measure, h))
            return False
    elif isinstance(measure, int):
        if measure <= 0:
            measure = slice(measure, None)
        else:
            measure = slice(measure, measure + 1)

    dataDict = {}
    rowLength = 0
    for row in fs:

        rowList = row.split(delim)
        try:
            _putInData(dataDict, rowList[:3], rowList[measure])
            if rowLength == 0:
                rowLength = len(rowList)
        except ValueError:
            _L.info("{0}: Not data row, could be headers {1}".format(
                path, rowList))
        except IndexError:
            _L.info("{0}: Unexpeded length of row {1}".format(
                path, rowList))

    fs.close()

    nPlate = max(dataDict) + 1
    data = []

    start = measure.start is None and 0 or measure.start
    stop = measure.stop is None and rowLength or measure.stop
    step = measure.step is None and 1 or measure.step

    if np.sign(start) == np.sign(stop):
        measures = abs(int(np.ceil((stop - start) / float(step))))
    else:
        measures = abs(int(np.ceil((stop - (rowLength + start)) / float(step))))

    for plateI in range(nPlate):

        if plateI in dataDict:
            pMeasures = dataDict[plateI]
            plate = np.zeros(
                [v + 1 for v in map(max, zip(*pMeasures.keys()))] + [measures],
                dtype=dtype)

            if dtype == np.float:
                plate *= np.nan

            data.append(plate)
            try:
                for k, v in pMeasures.items():
                    plate[k][...] = v
            except:
                _L.critical(
                    ("{0}: Unexpected data for shape {1}," +
                     "plate {2} pos {3}: {4}").format(
                         path, plate.shape, plateI, k, v))
                return None

        else:

            data.append(None)

    return np.array(data)


def uniques(metaData, forcePlatewise=False, slicer=None):
    """Generates a dictionary lookup for unique strains and the positions
    they occur in.

    Args:
        metaData (meta_data.Meta_Data): Instance of metadata

    Kwargs:
        forcePlatewise (bool):
            If strains sharing same name on different plates should be
            considered different strains (good if having different
            media on different plates but no difference in metadata).

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

    for pos in metaData.generate_coordinates():

        strain = tuple(metaData(*pos))[slicer]

        if forcePlatewise:
            strain += (_PLATE_STRING.format(pos[0]),)

        if strain not in _uniques:
            _uniques[strain] = [pos]
        else:
            _uniques[strain].append(pos)

    return _uniques


def filterUniquesOnPlate(uniqueDict, plate):
    """If forcePlatewise was true when getting uniques, this method
    returns the uniques of a specific plate.

    Args:
        uniqueDict (dict):  A dict containing unique identifiers and position
                            lists as returned by the unique method

        plate (int):    A plate to return uniques dict for
    """

    plate = _PLATE_STRING.format(plate)

    return {k: v for k, v in uniqueDict.items() if k[-1] == plate}


def splitStrainsPerPlates(strainDict):
    """If `forcePlatewise` was `True`, this will make an ordered list with
    each plate's info in a separate item.

    Parameters
    ----------

    strainDict: dict
        A dict with strain meta-data as keys

    Returns
    -------

    list of dicts
        A 4 long list with a subset of the input dict matching each plate
    """

    return [filterUniquesOnPlate(strainDict, p) for p in range(4)]


def getDataPerUnique(uniqueDict, dataObject, measure=None):
    """Collects all measures for each strain

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

        Dict of all measures

    """
    if not isinstance(measure, slice):
        measure = slice(measure)

    _measures = dict()
    easeMode = dataObject.ndim == 3

    for strain, positions in uniqueDict.items():

        if easeMode:
            vals = dataObject[
                tuple(map(np.array, zip(*positions)))][..., measure]
        else:
            vals = []
            for p, r, c in positions:
                vals.append(dataObject[p][r, c])

            vals = np.array(vals)[..., measure]

        _measures[strain] = vals

    return _measures


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

    _stats = dict()

    for strain, vals in getDataPerUnique(uniqueDict, dataObject,
                                         measure=measure).items():

        finVals = vals[np.isfinite(vals)]

        _stats[strain] = dict(
            n=vals.size,
            nans=vals.size - finVals.size,
            mean=finVals.mean(),
            std=finVals.std(ddof=1),
            cv=finVals.std(ddof=1) / finVals.mean())

    return _stats


def getArray(strainStatsDict, key):
    """Produces an array of the stats-measure over all strains.

    Args:
        strainStatsDict (dict): A dict as returned from generalStatsOnStrains

        key (string):   A known strain stats key such as e.g. 'mean' or 'cv'

    Returns:

        numpy.array     The corresponding key value for all strains
    """

    return strainStatsDict.keys(), np.array([s[key] for s in strainStatsDict.values()])

#
# PLOTTERS
#


def plotRndStrains(uniqueDict, dataObject, forceInclude=None,
                   measure=None, n=10, fig=None, showFig=True):

    D = getDataPerUnique(uniqueDict, dataObject, measure=measure)

    Dkeys = D.keys()
    np.random.shuffle(Dkeys)
    if (forceInclude is not None and forceInclude in uniqueDict):
        Dkeys = [k for k in Dkeys if k != forceInclude]
        Dkeys[0] = forceInclude

    if fig is None:
        fig = plt.figure()

    fig.clf()
    ax = fig.gca()

    ax.boxplot(
        [D[key].ravel() for key in Dkeys[:n]])
    ax.set_xticklabels(
        [", ".join(l) for l in Dkeys[:n]])

    offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) / 30.0
    axTop = ax.get_ylim()[1]

    for i in range(10):

        d = D[Dkeys[i]]
        uB = d[np.isfinite(d)].max()
        lB = d[np.isfinite(d)].min()

        ax.annotate(
            "n {0}".format(np.isfinite(d).sum()),
            (i + 0.75, (uB + offset) < axTop and (uB + offset) or (lB - offset)))

    if showFig:
        fig.show()

    return fig
