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

import odf.opendocument as opendocument
import odf.table as table
from odf.text import P
import md5
import os

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger

#
# META DATA CLASS
#


class Meta_Data(object):

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

    MATCH_NO = 0
    MATCH_PLATE = 1
    MATCH_PLATE_HEADERS = 2

    def __init__(self, plateShapes, *paths):

        self._logger = logger.Logger("Meta Data")

        self._plateShapes = plateShapes
        self._paths = paths

        self._data = None
        self._sheetNames = None
        self._sheetReadOrder = None
        self._plateI2Header = [None] * len(plateShapes)
        self._coordinatResolver = [_Coordinate_Resolver(plateShapes[i],
                                                        Meta_Data.PLATE_FULL)
                                   for i in range(len(plateShapes))]

        self._loadPaths()
        self._guessLoaded = self._guessCoordinates()

    def generateCoordinates(self):

        for plate, pShape in enumerate(self._plateShapes):

            if pShape is not None:

                for rowI in xrange(pShape[0]):

                    for colI in xrange(pShape[1]):

                        yield plate, rowI, colI

    def __getitem__(self, plate):

        return self._coordinatResolver[plate]

    def __call__(self, plate, row, col):

        return self._coordinatResolver[plate](row, col)

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
                    if not(i < len(self._sheetReadOrder)):
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
                                dID = self._sheetReadOrder[n]
                                lD = len(self._data[dID])
                                n += 1
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

            if n + 1 == len(self._plateShapes):
                return True

            self._logger.warning("Some of the loaded meta-data wasn't used")
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

    def _makeRectangle(self, data, fill=u''):

        maxWidth = max(map(len, data))

        for row in data:
            lRow = len(row)
            if lRow != maxWidth:
                row += [fill] * (maxWidth - lRow)

    def _loadPaths(self):

        self._headers = dict()
        self._data = dict()
        self._sheetNames = dict()
        self._sheetReadOrder = list()

        for path in self._paths:

            try:
                doc = opendocument.load(path)
            except:
                self._logger.warning("Could not read file '{0}'".format(path))
                continue

            for t in doc.getElementsByType(table.Table):

                rows = t.getElementsByType(table.TableRow)

                matchType = self._hasValidRows(rows)

                if matchType == Meta_Data.MATCH_NO:

                    self._logger.warning(
                        u"Sheet {0} of {1} had no understandable data".format(
                            t.getAttribute("name"), os.path.basename(path)))

                else:

                    data = []
                    for row in rows:
                        dataRow = []
                        for tc in row.getElementsByType(table.TableCell):
                            ps = tc.getElementsByType(P)
                            if (len(ps)) == 0:
                                dataRow.append(u'')
                            else:
                                dataRow.append(ps.firstChild.data)
                        data.append(dataRow)

                    self._makeRectangle(data)

                    name = t.getAttribute("name")
                    sheetID = md5.new().hexdigest()

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
            full = Meta_Data.PLATE_FULL
        if offset is None:
            offset = Meta_Data.OFFSET_UPPER_LEFT

        #1 SANITYCHECK
        nInData = self._plateShapes[plateIndex][0] * \
            self._plateShapes[plateIndex][1]

        if not(full is Meta_Data.PLATE_FULL and
                nInData == len(self._data[dataKey]) or
                full is Meta_Data.PLATE_PARTIAL and
                nInData * 4 == len(self._data[dataKey])):

            self._logger.error(
                u"Sheet {0} can't be assigned as {1}".format(
                    self._sheetNames[dataKey],
                    ["PARTIAL", "FULL"][full is Meta_Data.PLATE_FULL]))
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


class _Coordinate_Resolver(object):

    def __init__(self, shape, full, orientation=None, horizontalDirection=None,
                 verticalDirection=None):

        self._full = None
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

        full = value is Meta_Data.PLATE_FULL

        if full != self._full:

            self._full = full

            if not full:
                partialShape = [d / 2 for d in self._shape]
                self._part = [_Coordinate_Resolver(
                    shape=partialShape,
                    full=Meta_Data.PLATE_FULL) for _ in range(4)]
            else:
                self._part = None

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

    def __call__(self, row, col):

        if self._full:
            if self._data is None:
                return list()
            else:
                return self._data[self._rowVal(row) + self._colVal(col)]
        else:
            offR = row % 2
            offC = col % 2
            if offR:
                if offC:
                    return self._part[Meta_Data.OFFSET_LOWER_RIGHT](
                        (row - 1) / 2, (col - 1) / 2)
                else:
                    return self._part[Meta_Data.OFFSET_LOWER_LEFT](
                        (row - 1) / 2, col / 2)
            else:
                if offC:
                    return self._part[Meta_Data.OFFSET_LOWER_RIGHT](
                        row / 2, (col - 1) / 2)
                else:
                    return self._part[Meta_Data.OFFSET_LOWER_LEFT](
                        row / 2, col / 2)

    def _rowVal(self, row):

        if self._horizontal:
            if self._rowAsc:
                return row * self._rowLength
            else:
                return (self._rowMax - row) * self._rowLength
        else:
            if self._rowAsc:
                return row
            else:
                return self._rowMax - row

    def _colVal(self, col):

        if self._horizontal:
            return col * self._colLength
        else:
            return col
