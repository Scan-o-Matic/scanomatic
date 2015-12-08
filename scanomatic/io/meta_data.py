"""Contains meta-data reader/interface"""
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

from types import StringTypes
import odf.opendocument as opendocument
import odf.table as table
from odf.text import P
from odf.element import Text
import md5
import os
import copy
import time

#
#   OPTIONAL IMPORT
#

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger

#
# METHODS
#

def splitSheetToFiles(inputPath, expectedRows=384, skipRows=0, headerRows=1,
        slicer=slice(None)):
    
    """Takes a path to an ods-file and splits the contents of its sheets to
    several files

    Parameters
    ----------

    inputPath : str
        Path to file

    expectedRows : int , optional
        Number of rows for each split. (Default is 384)

    skipRows : int , optional
        Number of rows to skip at the beginning (Default is 0)

    headerRows : int, optional
        Number of rows that are header rows at the top of each sheet after
        the skipped rows

    slicer : slice, optional
        A column slice for the columns to select (Default is to include all
        columns).

    Raises
    ------
    
    ImportError
        If pandas is not installed

    """
    def _saveSheet(inputPath, sheetName, section, data, headers, sep=" | ",
            slicer=slice(None)):

        section = str(section)
        #pd.DataFrame({h: v for h, v in zip((sep.join(hCol) for hCol in zip(*headers)), zip(*data))}).to_excel(inputPath + section + ".xlsx", sheetName + sep + section, index=False)
        try:
            pd.DataFrame([dr[slicer] for dr in data], columns=[sep.join(hCol) for hCol in zip(*headers)[slicer]]).to_excel(inputPath + "." + section + ".xlsx", sheetName + sep + section, index=False)
            print "Saved {0}".format(inputPath + "." + section + ".xlsx")
        except (AssertionError, ValueError), e:
            print "Skipping {0}{1}{2} due to {3}".format(
                    sheetName, sep, section, e)

    if not _PANDAS:
        raise ImportError("Requires pandas for saving")

    doc = opendocument.load(inputPath)

    for t in doc.getElementsByType(table.Table):

        rows = t.getElementsByType(table.TableRow)
        name = t.getAttribute("name")

        hRows = None
        section = 0


        for idR, row in enumerate(rows):

            if (idR < skipRows):
                continue
            else:
                idDataR = idR - (skipRows + headerRows)
                if idDataR % expectedRows == 0 or idDataR < 0:
                    if idDataR < 0:
                        savehRows = True
                    elif idDataR > 0:
                        _saveSheet(inputPath, name, section, data, hRows, slicer=slicer)
                        section += 1
                    data = []

                dataRow = []

                for tc in row.getElementsByType(table.TableCell):
                    ps = tc.getElementsByType(P)
                    if (len(ps)) == 0:
                        dataRow.append(u'')
                    else:
                        dataRow.append(u', '.join(
                            [pps.firstChild.data for pps in ps if hasattr(pps.firstChild, "data")]))

                if savehRows:
                    if hRows is None:
                        hRows = []
                    hRows.append(dataRow)
                else:
                    data.append(dataRow)

                savehRows = False
                
        _saveSheet(inputPath, name, section, data, hRows, slicer=slicer)


def scanomaticScale(metaData, refTuple, refPlateAppend={},
        refOffset=None):
    """Make a 3:1 pinning up-scaling of a meta-data.

    Parameters
    ----------

    metaData : Meta_Data_Object or list of lists
        Data-containing object that will be used for all offsets except
        reference offset

    refTuple : tuple or list
        Object containing strain information for the refernce lawn

    refOffset : {1, 2, 3, 4} , optional
        Offset for where lawn is pinned, using Meta_Data_Base.OFFSETS.
        (Default is Meta_Data_Base.OFFSET_LOWER_RIGHT)

    Returns
    -------

    Meta_Data_Base
        Meta-data object that represents the pinning
    """

    if refOffset is None:
        refOffset = metaData.OFFSET_LOWER_RIGHT

    #metaData.full = Meta_Data.PLATE_PARTIAL

    newShape = [(x*2, y*2) for x, y in metaData.shape]

    newMetaData = Meta_Data_Base(newShape)
    for idPlate in xrange(len(newShape)):
        cr = newMetaData[idPlate]
        cr.full = metaData.PLATE_PARTIAL

        for offset in metaData.OFFSETS:
            cr[offset].full = metaData.PLATE_FULL
            cr[offset].data = metaData

        cr[refOffset].lawn = True
        cr[refOffset].data = copy.deepcopy(refTuple)
        refData = cr[refOffset].data
        for k, vals in refPlateAppend.items():
            refData.append(vals[idPlate])

    return newMetaData

#
# META DATA CLASS
#


class Meta_Data_Base(object):

    ORIENTATION_HORIZONTAL = 0
    ORIENTATION_VERTICAL = 1

    VERTICAL_ASCENDING = 0
    VERTICAL_DESCENDING = 1

    HORIZONTAL_ASCENDING = 0
    VERTICAL_ASCENDING = 1

    PLATE_FULL = 0
    PLATE_PARTIAL = 1

    OFFSET_UPPER_LEFT = 0  # (0, 0)
    OFFSET_UPPER_RIGHT = 1  # (0, 1)
    OFFSET_LOWER_LEFT = 2  # (1, 0)
    OFFSET_LOWER_RIGHT = 3  # (1, 1)

    OFFSETS = (0, 1, 2, 3)

    def __init__(self, plateShapes):

        self._plateShapes = plateShapes
        self._coordinatResolver = [_Coordinate_Resolver(plateShapes[i],
                                                        Meta_Data_Base.PLATE_FULL)
                                   for i in range(len(plateShapes))]
    def __getitem__(self, plate):

        return self._coordinatResolver[plate]

    def __call__(self, plate, row, col):

        return self._coordinatResolver[plate](plate, row, col)

    def __eq__(self, other):


        if (not hasattr(other, "shape") or
                len([True for a, b in zip(self.shape, other.shape)
                     if a == b]) != len(self)):

            return False

        useOtherCall = hasattr(other, "__call__")
        otherEval = lambda plate, row, col: (useOtherCall and
                                             other(plate, row, col) or
                                             other[plate, row, col])

        for idPlate in range(len(self)):
            rows, cols = self._plateShapes[idPlate]
            for idRow in range(rows):
                for idCol in range(cols):
                    if self(idPlate, idRow, idCol) != otherEval(idPlate,
                                                                idRow,
                                                                idCol):

                        return False

        return True

    def __len__(self):

        return len(self._plateShapes)

    @property
    def shape(self):
        return self._plateShapes

    def _hasTheData(self):

        return (hasattr(self, "_headers") and hasattr(self, "_data") and
                hasattr(self, "_sheetReadOrder") and
                hasattr(self, "_sheetNames"))

    def appendColumns(self, columnHeader, columnData):

        if not self._hasTheData():
            for i in self.OFFSETS:
                self[i].appendColumns(columnHeader, columnData) 

        else:

            for idPlate, headers in self._headers.items():

                headers.append(columnHeader)

                for row in self._data[idPlate]:
                    row.append(columnData[self._sheetReadOrder.index(idPlate)])

    def find(self, key, exact=True):
        """Generate coordinate tuples for where key matches meta-data

        Parameters
        ----------

        key : str or list
            Strain meta-data to look for

        exact : bool, optional
            If key needs to fully match meta-data (default) or if it is
            sufficient that key exists in meta-data.

        Returns
        -------

        generator
            Each item being a (plate, row, column)-tuple.
        """

        for coord in self.generateCoordinates():

            if exact and key == self(*coord):
                yield coord
            elif not exact and key in self(*coord):
                yield coord

    def sliceByFound(self, obj, key, exact=True, plates=None):
        """Produces a slice of passed object according to found key possitions.

        Note:: This only works with four uniform plates.

        Parameters
        ----------

        obj : str or numpy.ndarray-like
            If string is passed and such an object exists on self, 

        key : str or list
            Strain meta-data to look for

        exact : bool, optional
            If key needs to fully match meta-data (default) or if it is
            sufficient that key exists in meta-data.

        plates : tuple, optional
            A tuple of plate indices to include in searching for the key.
            If not supplied, all plates are searched (default).

        Returns
        -------

        obj-slice
            A slice out of `obj` for the indices where key was matching.

        Raises
        ------

        AttributeError
            If `obj` is passed as a string and no such attribute exists on 
            `self`

        TypeError
            If `obj` is not `numpy.ndarray`-like

        IndexError
            If obj is not matching in shape `self` and index is found outside
            the range of `obj`

        KeyError
            If key not known
        """
        if isinstance(obj, StringTypes):
            if hasattr(self, obj):
                obj = getattr(self, obj)
            else:
                raise AttributeError("Unknown attribute on self `{0}`".format(
                    obj))
    
        vals = tuple(zip(*(c for c in self.find(key, exact=exact) if
            plates is None and True or c[0] in plates)))

        if not vals:
            raise KeyError("The key '{0}' is not known in meta-data".format(
                key))

        return obj[vals]

    def findPlate(self, val, colSlice=None):
        
        if colSlice is None:
            colSlice = slice(-1, None)

        if (hasattr(self, "_data") and self._data is not None):
            for i in range(len(self._data)):
                if isinstance(self._data, dict):
                    if val in self._data[self._sheetReadOrder[i]][0]:
                        return i
                else:
                    if val in self(i, 0, 0):
                        return i

            raise KeyError("Value '{0}' not present".format(val))

        else:

            return self[self.OFFSET_UPPER_LEFT].findPlate(val, colSlice=colSlice)

    def copyPastePlates(self, fromSheet=None, nCopies=3, appendColumns={}):
        """Inplace adjustment of metaData content to
        """

        if not self._hasTheData():
            if (hasattr(self, "_data") and self._data is not None):
                self._data.copyPastePlates(fromSheet=fromSheet,
                                        nCopies=nCopies,
                                        appendColumns=appendColumns)
                if self.shape != self._data.shape:
                   self._plateShapes = copy.deepcopy(self._data.shape) 

            else:
                for i in self.OFFSETS:
                    self[i].copyPastePlates(fromSheet=fromSheet,
                                            nCopies=nCopies,
                                            appendColumns=appendColumns)
                    if self.shape != self[i].shape:
                        self._plateShapes = copy.deepcopy(self[i].shape) 
            return

        if fromSheet is None:
            fromSheet = -1

        if isinstance(fromSheet, int):
            fromSheet = self._sheetReadOrder[fromSheet]

        for k, v in appendColumns.items():
            if len(v) == len(self):
                self.appendColumns(k, v)
                appendColumns.pop(k)

        fromData = self._data[fromSheet]
        fromIndex = self._sheetReadOrder.index(fromSheet)
        fromFull = self._coordinatResolver[fromIndex].full
        fromLawn = self._coordinatResolver[fromIndex].lawn

        for i in range(nCopies):
            sheetID = md5.new("{0}{1}".format(time.time(), i)).hexdigest()
            self._data[sheetID] = copy.deepcopy(fromData)
            self._headers[sheetID] = copy.deepcopy(self._headers[fromSheet])
            self._sheetReadOrder.append(sheetID)
            self._sheetNames[sheetID] = (self._sheetNames[fromSheet] + 
                                         " (copy {0})".format(i + 1))

            self._plateShapes.append(copy.deepcopy(self._plateShapes[fromIndex]))

            self._coordinatResolver.append(_Coordinate_Resolver(
                self._plateShapes[i], fromFull))

            if fromFull:
                self._coordinatResolver[-1].full = fromFull

            if fromLawn:
                self._coordinatResolver[-1].lawn = fromLawn

            self._coordinatResolver[-1].data = self._data[sheetID]

        for k, v in appendColumns.items():
            if len(v) == len(self):
                self.appendColumns(k, v)
                appendColumns.pop(k)

    def generateCoordinates(self):

        for plate, pShape in enumerate(self._plateShapes):

            if pShape is not None:

                for rowI in xrange(pShape[0]):

                    for colI in xrange(pShape[1]):

                        yield plate, rowI, colI



class Meta_Data(Meta_Data_Base):

    MATCH_NO = 0
    MATCH_PLATE = 1
    MATCH_PLATE_HEADERS = 2

    def __init__(self, plateShapes, *paths):

        self._logger = logger.Logger("Meta Data")

        super(Meta_Data, self).__init__(plateShapes)

        self._paths = paths

        self._data = None
        self._sheetNames = None
        self._sheetReadOrder = None
        self._plateI2Header = [None] * len(plateShapes)

        self._loadPaths()
        self._guessLoaded = self._guessCoordinates()

    def getHeaderRow(self, plate):
        """Gives header column labels for plate

        Returns:
            Header row (list)   Column labels if such exist.
                                If not it produces a list with "" strings.
                                If there's no plate with the index, returns None
        """
        if (plate < len(self._plateI2Header)):
            pID = self._plateI2Header[plate]
            if pID in self._headers:
                return self._headers[pID]

        if plate >= len(self._plateShapes) or self._plateShapes[plate] is None:
            return None

        return ["" for _ in range(len(self(plate, 0, 0)))]

    def _guessCoordinates(self):

        if self._sheetReadOrder is not None:

            n = 0
            for i, shape in enumerate(self._plateShapes):

                if shape is not None:

                    shape = shape[0] * shape[1]
                    if not(n < len(self._sheetReadOrder)):
                        self._logger.warning(
                            "Not enough valid data-sheets")
                        return False

                    dID = self._sheetReadOrder[n]
                    lD = len(self._data[dID])
                    if (shape == lD):
                        self.setPositionLookup(i, dID)
                    elif (lD * 4 == shape):
                        for j in range(4):

                            if j != 0:
                                n += 1
                                dID = self._sheetReadOrder[n]
                                lD = len(self._data[dID])
                                if not(lD * 4 == shape):
                                    self._logger.warning(
                                        "Meta-Data did not fill up plate")
                                    return False

                            self.setPositionLookup(
                                i, dID,
                                full=Meta_Data.PLATE_PARTIAL,
                                offset=[
                                    Meta_Data.OFFSET_UPPER_LEFT,
                                    Meta_Data.OFFSET_UPPER_RIGHT,
                                    Meta_Data.OFFSET_LOWER_LEFT,
                                    Meta_Data.OFFSET_LOWER_RIGHT][j])

                    n += 1

            if n == len(self._sheetReadOrder):
                return True

            self._logger.warning(
                "Some of the loaded meta-data wasn't used" +
                ", needed {0} plates, when done this was left over: {1}".format(
                    len(self._plateShapes),
                    [self._sheetNames[sID] for sID in self._sheetReadOrder[n:]]
                ))
            return True

        self._logger.info(
            "No plates known, can't really guess their content then.")
        return False

    def _hasValidRows(self, rows):
        """Evaluates if the row count could match up with plate

        Args:
            rows (iterable):    The data as rows

        Returns:
            Match evaluation (int):

                `Meta_Data.MATCH_NO` if no matching plate
                `Meta_Data.MATCH_PLATE` if exactly matches plate or 1/4th
                `Meta_Data.MATCH_PLATE_HEADERS` if matches plate or 1/4th
                    with headers
        """
        n = len(rows)
        for plateShape in self._plateShapes:
            if (plateShape is not None):

                plate = plateShape[0] * plateShape[1]

                if plate == n or n * 4 == plate:
                    return Meta_Data.MATCH_PLATE

                elif plate == n - 1 or (n - 1) * 4 == plate:

                    return Meta_Data.MATCH_PLATE_HEADERS

        return Meta_Data.MATCH_NO

    def _endTrimRows(self, rows, rowContentExtractor):

        breakRows = False

        for i, row in enumerate(rows[::-1]):

            if len ([True for _ in rowContentExtractor(row) if _ != u'']):
                break

            """
            tcs = row.getElementsByType(table.TableCell)

            if len(tcs) == 0:
                break

            for tc in tcs:
                ps = tc.getElementsByType(P)
                if (len(ps)) > 0:
                    if (u''.join(
                            [pps.firstChild.data for pps in ps]) != u''):

                        breakRows = True
                        break
            
            if breakRows:
                break
            """

        while i > 0:

            rows.get_highest_priority()
            i -= 1

    def _makeRectangle(self, data, fill=u''):

        maxWidth = max(map(len, data))

        for row in data:
            lRow = len(row)
            if lRow != maxWidth:
                row += [fill] * (maxWidth - lRow)

    @staticmethod
    def _getTextInElementFromOds(elem):
        ret = ""

        if isinstance(elem, Text):
            ret += elem.data
    
        if elem.hasChildNodes():

            ret += Meta_Data._getTextInElementFromOds(elem.firstChild)

        if elem.nextSibling:
            ret += Meta_Data._getTextInElementFromOds(elem.nextSibling)

        return ret

    @staticmethod
    def _getRowContentFromOds(row):

        dataRow = []
        for tc in row.getElementsByType(table.TableCell):
            E = tc.getElementsByType(P)
            if (len(E)) == 0:
                dataRow.append(u'')
            else:
                dataRow.append(u', '.join(
                    Meta_Data._getTextInElementFromOds(e) for e in E))

        while len(dataRow) > 0 and dataRow[-1] == u'':
            dataRow.pop()

        return dataRow

    def _loadPaths(self):

        self._headers = dict()
        self._data = dict()
        self._sheetNames = dict()
        self._sheetReadOrder = list()

        for path in self._paths:

            fileType = path.lower().split(".")[-1]

            if fileType in ["xls", "xlsx"]:
                if not _PANDAS:
                    raise ImportError("Requires pandas for saving")
                try:
                    doc = pd.ExcelFile(path)
                except:
                    self._logger.warning(
                        "Could not read Excel file '{0}'".format(path))
                    continue

                sheetsGenerator = (doc.parse(n, header=None).fillna(value=u'') for n in doc.sheet_names)
                rowsLister = lambda df: list(r.tolist() for i, r in df.iterrows())
                rowContentExtractor = lambda row: row
                sheetNamer = lambda _ : doc.sheet_names[idSheet]

            elif fileType in ["ods"]:
                try:
                    doc = opendocument.load(path)
                except:
                    self._logger.warning("Could not read file '{0}'".format(
                        path))
                    continue

                sheetsGenerator = doc.getElementsByType(table.Table)
                rowsLister = lambda sheet: sheet.getElementsByType(
                    table.TableRow)
                rowContentExtractor = self._getRowContentFromOds
                sheetNamer = lambda sheet : sheet.getAttribute("name")

            else:
                self._logger.warning("Unsupported file format for '{0}'".format(
                    path))
                continue

            for idSheet, t in enumerate(sheetsGenerator):

                rows = rowsLister(t)

                self._endTrimRows(rows, rowContentExtractor)

                matchType = self._hasValidRows(rows)

                if matchType == Meta_Data.MATCH_NO:

                    self._logger.warning(
                        (u"Sheet {0} of {1} had no understandable data" +
                         u"({2} rows)").format(
                             sheetNamer(t),
                             os.path.basename(path),
                             len(rows)))

                else:

                    data = []
                    for row in rows:
                        data.append(rowContentExtractor(row))

                    self._makeRectangle(data)

                    name = sheetNamer(t)
                    sheetID = md5.new(name + str(time.time())).hexdigest()

                    if matchType == Meta_Data.MATCH_PLATE:

                        self._headers[sheetID] = ["" for _ in
                                                  range(len(data[0]))]
                    else:

                        self._headers[sheetID] = data[0]
                        data = data[1:]



                    self._data[sheetID] = data
                    self._sheetNames[sheetID] = u"{0}:{1}".format(
                        os.path.basename(path), name)

                    self._sheetReadOrder.append(sheetID)

    def setPositionLookup(self, plateIndex, dataKey,
                          orientation=None, verticalDirection=None,
                          horizontalDirection=None,
                          full=None,
                          offset=None):

        #0 SETTING DEFAULTS
        if orientation is None:
            orientation = Meta_Data.ORIENTATION_HORIZONTAL
        if verticalDirection is None:
            verticalDirection = Meta_Data.VERTICAL_ASCENDING
        if horizontalDirection is None:
            horizontalDirection = Meta_Data.HORIZONTAL_ASCENDING
        if full is None:
            full = Meta_Data_Base.PLATE_FULL
        if offset is None:
            offset = Meta_Data.OFFSET_UPPER_LEFT

        #1 SANITYCHECK
        nInData = self._plateShapes[plateIndex][0] * \
            self._plateShapes[plateIndex][1]

        if not(full is Meta_Data_Base.PLATE_FULL and
                nInData == len(self._data[dataKey]) or
                full is Meta_Data.PLATE_PARTIAL and
                nInData == 4 * len(self._data[dataKey])):

            self._logger.error(
                u"Sheet {0} can't be assigned as {1}".format(
                    self._sheetNames[dataKey],
                    ["PARTIAL", "FULL"][full is Meta_Data_Base.PLATE_FULL]))
            raise ValueError
            return False

        #1.5 Link header info
        if offset == Meta_Data.OFFSET_UPPER_LEFT:
            self._plateI2Header[plateIndex] = dataKey

        #2 Invoke sorting
        cr = self._coordinatResolver[plateIndex]

        cr.full = full

        if not cr.full:
            cr = cr[offset]

        cr.data = self._data[dataKey]
        cr.orientation = orientation
        cr.verticalDirection = verticalDirection
        cr.horizontalDirection = horizontalDirection


class _Coordinate_Resolver(Meta_Data_Base):

    def __init__(self, shape, full, orientation=None, horizontalDirection=None,
                 verticalDirection=None):

        self._full = None
        self._lawn = False
        self._horizontal = True
        self._rowAsc = True
        self._colAsc = True

        self._shape = shape
        self._rowLength = shape[0]
        self._colLength = shape[1]
        self._rowMax = self._rowLength - 1
        self._colMax = self._colLength - 1

        self.full = full

        self._data = None

        if orientation is not None:
            self.orientation = orientation

        if horizontalDirection is not None:
            self.horizontalDirection = horizontalDirection

        if verticalDirection is not None:
            self.verticalDirection = verticalDirection

    @property
    def horizontalDirection(self):

        return self._rowAsc

    @horizontalDirection.setter
    def horizontalDirection(self, horizontalDirection):

        self._rowAsc = horizontalDirection is Meta_Data.HORIZONTAL_ASCENDING

    @property
    def verticalDirection(self):

        return self._colAsc

    @verticalDirection.setter
    def verticalDirection(self, verticalDirection):

        self._colAsc = verticalDirection is Meta_Data.VERTICAL_ASCENDING

    @property
    def horizontal(self):

        return self._horizontal

    @horizontal.setter
    def horizontal(self, orientation):
        self._horizontal = orientation is Meta_Data.ORIENTATION_HORIZONTAL

    @property
    def full(self):

        return self._full

    @full.setter
    def full(self, value):

        full = value is True or value is Meta_Data_Base.PLATE_FULL

        if full != self._full:

            self._full = full

            if not full:
                partialShape = [d / 2 for d in self._shape]
                self._part = [_Coordinate_Resolver(
                    shape=partialShape,
                    full=Meta_Data_Base.PLATE_FULL) for _ in range(4)]
            else:
                self._part = None

    @property
    def lawn(self):
        return self._lawn

    @lawn.setter
    def lawn(self, val):

        if val:
            self._lawn = True
        else:
            self._lawn = False

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def __getitem__(self, offset):

        if not self._full:
            return self._part[offset]
        else:
            return None

    def __call__(self, plate, row, col):

        if self._lawn:
            return self._data
        elif self._full:
            if self._data is None:
                return list()
            elif isinstance(self._data, Meta_Data):
                return self._data(plate, row, col)
            else:
                """
                print "Row {0}, Col {1} -> {2} + {3}".format(
                    row, col, self._rowVal(row), self._colVal(col))
                """
                return self._data[self._rowVal(row) + self._colVal(col)]
        else:
            offR = row % 2
            offC = col % 2
            if offR:
                if offC:
                    return self._part[Meta_Data.OFFSET_LOWER_RIGHT](
                        plate, (row - 1) / 2, (col - 1) / 2)
                else:
                    return self._part[Meta_Data.OFFSET_LOWER_LEFT](
                        plate, (row - 1) / 2, col / 2)
            else:
                if offC:
                    return self._part[Meta_Data.OFFSET_UPPER_RIGHT](
                        plate, row / 2, (col - 1) / 2)
                else:
                    return self._part[Meta_Data.OFFSET_UPPER_LEFT](
                        plate, row / 2, col / 2)

    @property
    def shape(self):

        return self._shape

    def __len__(self):

        return len(self._shape)

    def _rowVal(self, row):

        if self._horizontal:
            if self._rowAsc:
                return row * self._colLength
            else:
                return (self._rowMax - row) * self._colLength
        else:
            if self._rowAsc:
                return row
            else:
                return self._rowMax - row

    def _colVal(self, col):

        if self._horizontal:
            if self._rowAsc:
                return col
            else:
                return self._colMax - col
        else:
            if self._colAsc:
                return col * self._rowLength
            else:
                return (self._colMax - col) * self._rowLength
