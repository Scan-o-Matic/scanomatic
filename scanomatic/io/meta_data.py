from types import StringTypes
from odf import opendocument
import odf.table as table
from odf.text import P
from odf.element import Text
from hashlib import md5
import os
import copy
import time
from itertools import izip
import numpy as np

#
#   OPTIONAL IMPORT
#

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False
    pd = None

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger


#
# METHODS
#


class DataLoader(object):

    _SUFFIXES = []

    def __init__(self, path):
        self._logger = logger.Logger("MetaDataLoader")
        self._path = path
        self._sheet = -1
        self._entries = []
        self._row_iterators = []
        self._columns = []
        self._sheet_names = []
        self._headers = []

    def _reset(self):

        self._sheet = -1
        self._columns = []
        self._entries = []
        self._row_iterators = []
        self._sheet_names = []
        self._headers = []

    @property
    def rows(self):

        return self._entries[self._sheet]

    def get_sheet_name(self, sheet_index):

        return self._sheet_names[sheet_index]

    def get_next(self):

        for row in self._row_iterators[self._sheet]:
            yield self._get_next_row(row)

    @staticmethod
    def _get_next_row(self, row):

        raise NotImplemented

    def _get_empty_headers(self):

        return [None for _ in self._columns[self._sheet]]

    def get_headers(self, plate_size):

        if self._headers[self._sheet] is None:
            if self.sheet_is_valid_with_headers(plate_size):
                self._headers[self._sheet] = self._get_next_row(self._row_iterators[self._sheet].next())
            elif self.sheet_is_valid_without_headers(plate_size):
                self._headers[self._sheet] = self._get_empty_headers()

        return self._headers[self._sheet]

    @classmethod
    def can_load(cls, path):
        """

        Args:
            path:
            :type path : str
        Returns:

        """
        suffix = path[::-1].split(".", 1)[0][::-1].lower()
        return suffix in cls._SUFFIXES

    @property
    def sheets(self):

        return len(self._entries)

    def next_sheet(self, plate_size):

        self._sheet += 1

        while not self.sheet_is_valid(plate_size):

            if self._sheet >= len(self._entries):
                return None

            self._logger.warning(
                "Sheet {0} ({1} zero-indexed) has {2} and {3} entry rows. This doesn't match plate size {4}".format(
                    self._sheet_names[self._sheet],
                    self._sheet,
                    " header row" if self.sheet_is_valid_without_headers(plate_size) else "no headers",
                    self._entries[self._sheet],
                    plate_size))

            self._sheet += 1

        return self._sheet

    @property
    def has_more_data(self):

        return self._sheet < len(self._entries)

    def sheet_is_valid_with_headers(self, plate_size):

        return plate_size % (self._entries[self._sheet] - 1) == 0 and \
               (plate_size / (self._entries[self._sheet] - 1)) % 4 == 1

    def sheet_is_valid_without_headers(self, plate_size):

        return plate_size % self._entries[self._sheet] == 0 and \
               plate_size / self._entries[self._sheet] % 4 == 1

    def sheet_is_valid(self, plate_size):

        if 0 <= self._sheet < len(self._entries):

            return self.sheet_is_valid_with_headers(plate_size) or self.sheet_is_valid_without_headers(plate_size)

        return False


class ExcelLoader(DataLoader):

    _SUFFIXES = ['xls', 'xlsx']

    def __init__(self, path):

        super(ExcelLoader, self).__init__(path)
        self._data = None
        self._load()

    def _load(self):

        self._data = []
        self._reset()
        doc = pd.ExcelFile(self._path)
        for n in doc.sheet_names:
            self._sheet_names.append(n)
            self._load_sheet(doc.parse(n, header=None).fillna(value=u''))

    def _load_sheet(self, df):
        """

        Args:
            df: DataFrame / sheet
            :type df : pandas.DataFrame

        Returns:

        """
        self._data.append(df)
        self._entries.append(df.shape[0])
        self._columns.append(df.shape[1])
        self._headers.append(None)
        self._row_iterators.append(df.iterrows())

    @staticmethod
    def _get_next_row(row):

        return row[1].tolist()


class MetaData2(object):

    _LOADERS = (ExcelLoader,)

    def __init__(self, plate_shapes, *paths):

        self._logger = logger.Logger("MetaData")
        self._plate_shapes = plate_shapes
        self._data = tuple(None if shape is None else np.empty(shape, dtype=np.object) for shape in plate_shapes)
        self._headers = list(None for _ in plate_shapes)
        self._loading_plate = 0
        self._loading_offset = []
        self._paths = paths

        self._load(*paths)

        if not self.loaded:
            self._logger.warning("Not enough meta-data to fill all plates")

    def __call__(self, plate, outer, inner):

        return self._data[plate][outer, inner]

    def get_headers(self, plate):

        return self._headers[plate]

    @property
    def loaded(self):

        return self._loading_plate >= len(self._plate_shapes)

    @property
    def plate_completed(self):

        return len(self._loading_offset) == 0

    @staticmethod
    def get_loader(path):

        for loader in MetaData2._LOADERS:

            if loader.can_load(path):
                return loader(path)

        return None

    def _load(self, *paths):

        for path in paths:

            if self.loaded:
                return

            loader = MetaData2.get_loader(path)
            """:type : DataLoader"""
            if loader is None:
                self._logger.warning("Unknown file format, can't load {0}".format(path))
                continue

            size = self._get_sought_size()

            while loader.has_more_data:

                sheet_id = loader.next_sheet(size)
                if sheet_id is None:
                    break

                headers = loader.get_headers(size)

                if not self.has_matching_headers(headers):
                    self._logger.warning(
                        "Sheet {0} ({1}) of {2} headers don't match {3} != {4}".format(
                            loader.get_sheet_name(sheet_id),
                            sheet_id,
                            path,
                            headers,
                            self._headers[self._loading_plate]))
                    continue

                self._logger.info("Using {0}:{1} for plate {2}".format(
                    path, loader.get_sheet_name(sheet_id), self._loading_plate))
                self._update_headers_if_needed(headers)
                self._update_meta_data(loader)
                self._update_loading_offsets()

                if self.plate_completed:
                    self._loading_plate += 1

                if self.loaded:
                    return

                size = self._get_sought_size()

    def _get_sought_size(self):

        size = np.prod(self._plate_shapes[self._loading_plate])
        return size / 4 ** len(self._loading_offset)

    def _update_loading_offsets(self):

        if not self._loading_offset:
            return
        outer, inner = self._loading_offset[-1]
        inner += 1
        if inner > 1:
            inner %= 2
            outer += 1
            if outer > 1:
                self._loading_offset = self._loading_offset[:-1]
                self._update_loading_offsets()
                return

        self._loading_offset[-1] = (outer, inner)

    def has_matching_headers(self, headers):

        if self._headers[self._loading_plate] is None:
            return True
        elif len(self._headers[self._loading_plate]) != len(headers):
            return False
        elif all(h is None for h in self._headers[self._loading_plate]):
            return True
        else:
            return all(a == b for a, b in zip(self._headers[self._loading_plate], headers))

    def _update_headers_if_needed(self, headers):

        if self._headers[self._loading_plate] is None or all(h is None for h in self._headers[self._loading_plate]):
            self._headers[self._loading_plate] = headers

    def _update_meta_data(self, loader):

        slotter = self._get_slotting_iter(loader)

        for meta_data in loader.get_next():
            self._data[self._loading_plate][slotter.next()] = meta_data

    def _get_slotting_iter(self, loader):
        """

        Args:
            loader:
            :type loader: DataLoader

        Returns:
            Coordinate iterator
            :rtype : iter
        """
        def coord_lister(outer, inner, max_outer, max_inner):

            yield outer, inner
            inner += factor
            if inner >= max_inner:
                inner %= max_inner
                outer += factor
                if outer >= max_outer:
                    outer %= max_outer

        factor = np.log2(self._data[self._loading_plate].size / (loader.rows - loader.sheet_is_valid_with_headers(
            np.prod(self._plate_shapes[self._loading_plate]))))

        if factor != int(factor):
            return None

        elif factor == 0:
            # Full plate
            return izip(*np.unravel_index(
                np.arange(self._data[self._loading_plate].size),
                self._plate_shapes[self._loading_plate]))

        else:
            # Partial plate
            factor = int(factor)
            if factor > len(self._loading_offset):
                self._loading_offset += [(0, 0) for _ in range(factor - len(self._loading_offset))]

            outer, inner = map(sum, zip(*((o*2**l, i*2**l) for l, (o, i) in enumerate(self._loading_offset))))
            factor = 2 ** len(self._loading_offset)
            max_outer, max_inner = self._plate_shapes[self._loading_plate]

            return coord_lister(outer, inner, max_outer, max_inner)

#
# Old implementation
#


def split_sheet_to_files(input_path, expected_rows=384, skip_rows=0, header_rows=1, slicer=slice(None)):

    """Takes a path to an ods-file and splits the contents of its sheets to
    several files

    Parameters
    ----------

    input_path : str
        Path to file

    expected_rows : int , optional
        Number of rows for each split. (Default is 384)

    skip_rows : int , optional
        Number of rows to skip at the beginning (Default is 0)

    headers : int, optional
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

    def _save_sheet(input_path, sheet_name, section, data, headers, sep=" | ",
                    slicer=slice(None)):

        section = str(section)
        try:
            pd.DataFrame(
                [dr[slicer] for dr in data],
                columns=[sep.join(hCol) for hCol in zip(*headers)[slicer]]).to_excel(
                input_path + "." + section + ".xlsx", sheet_name + sep + section, index=False)
            print "Saved {0}".format(input_path + "." + section + ".xlsx")
        except (AssertionError, ValueError), e:
            print "Skipping {0}{1}{2} due to {3}".format(
                    sheet_name, sep, section, e)

    if not _PANDAS:
        raise ImportError("Requires pandas for saving")

    doc = opendocument.load(input_path)

    for t in doc.getElementsByType(table.Table):

        rows = t.getElementsByType(table.TableRow)
        name = t.getAttribute("name")

        headers = []
        section = 0
        for id_row, row in enumerate(rows):

            if id_row < skip_rows:
                continue
            else:
                id_data_row = id_row - (skip_rows + header_rows)
                save_header_rows = False

                if id_data_row % expected_rows == 0 or id_data_row < 0:
                    if id_data_row < 0:
                        save_header_rows = True
                    elif id_data_row > 0:
                        _save_sheet(input_path, name, section, data, headers, slicer=slicer)
                        section += 1
                    data = []

                data_rows = []

                for tc in row.getElementsByType(table.TableCell):
                    ps = tc.getElementsByType(P)
                    if (len(ps)) == 0:
                        data_rows.append(u'')
                    else:
                        data_rows.append(u', '.join(
                            [pps.firstChild.data for pps in ps if hasattr(pps.firstChild, "data")]))

                if save_header_rows:
                    if headers is None:
                        headers = []
                    headers.append(data_rows)
                else:
                    data.append(data_rows)

                save_header_rows = False
                
        _save_sheet(input_path, name, section, data, headers, slicer=slicer)


def scanomatic_scale(meta_data, ref_tuple, ref_plate_append={}, ref_offset=None):
    """Make a 3:1 pinning up-scaling of a meta-data.

    Parameters
    ----------

    meta_data : Meta_Data_Object or list of lists
        Data-containing object that will be used for all offsets except
        reference offset

    ref_tuple : tuple or list
        Object containing strain information for the refernce lawn

    ref_plate_append : dict
        I don't know

    ref_offset : {1, 2, 3, 4} , optional
        Offset for where lawn is pinned, using Meta_Data_Base.OFFSETS.
        (Default is Meta_Data_Base.OFFSET_LOWER_RIGHT)

    Returns
    -------

    MetaDataBase
        Meta-data object that represents the pinning
    """

    if ref_offset is None:
        ref_offset = meta_data.OFFSET_LOWER_RIGHT

    # metaData.full = Meta_Data.PLATE_PARTIAL

    new_shape = [(x*2, y*2) for x, y in meta_data.shape]

    new_meta_data = MetaDataBase(new_shape)
    for id_plate in xrange(len(new_shape)):
        cr = new_meta_data[id_plate]
        cr.full = meta_data.PLATE_PARTIAL

        for offset in meta_data.OFFSETS:
            cr[offset].full = meta_data.PLATE_FULL
            cr[offset].data = meta_data

        cr[ref_offset].lawn = True
        cr[ref_offset].data = copy.deepcopy(ref_tuple)
        ref_data = cr[ref_offset].data
        for k, vals in ref_plate_append.items():
            ref_data.append(vals[id_plate])

    return new_meta_data

#
# META DATA CLASS
#


class MetaDataBase(object):

    ORIENTATION_HORIZONTAL = 0
    ORIENTATION_VERTICAL = 1

    VERTICAL_ASCENDING = 0
    VERTICAL_DESCENDING = 1

    HORIZONTAL_ASCENDING = 0
    HORIZONTAL_DESCENDING = 1

    PLATE_FULL = 0
    PLATE_PARTIAL = 1

    OFFSET_UPPER_LEFT = 0  # (0, 0)
    OFFSET_UPPER_RIGHT = 1  # (0, 1)
    OFFSET_LOWER_LEFT = 2  # (1, 0)
    OFFSET_LOWER_RIGHT = 3  # (1, 1)

    OFFSETS = (0, 1, 2, 3)

    def __init__(self, plate_shapes):

        self._sheet_read_order = None
        self._data = None
        self._headers = None
        self._sheet_names = None

        self._plate_shapes = plate_shapes

        if len(plate_shapes) > 1:
            self._coordinate_resolvers = [CoordinateResolver(plate_shapes[i], MetaDataBase.PLATE_FULL)
                                          for i in range(len(plate_shapes))]
        else:
            self._coordinate_resolvers = None

    def __getstate__(self):

        return {k: v for k, v in self.__dict__.iteritems() if k != "_logger"}

    def __setstate__(self, state):

        self.__dict__.update(state)

    def __getitem__(self, plate):

        return self._coordinate_resolvers[plate]

    def __call__(self, plate, row, col):

        return self._coordinate_resolvers[plate](plate, row, col)

    def __eq__(self, other):

        if (not hasattr(other, "shape") or
                len([True for a, b in zip(self.shape, other.shape)
                     if a == b]) != len(self)):

            return False

        use_other_call = hasattr(other, "__call__")
        other_eval = lambda plate, row, col: (use_other_call and other(plate, row, col) or other[plate, row, col])

        for id_plate in range(len(self)):
            rows, cols = self._plate_shapes[id_plate]
            for id_row in range(rows):
                for id_col in range(cols):
                    if self(id_plate, id_row, id_col) != other_eval(id_plate, id_row, id_col):
                        return False

        return True

    def __len__(self):

        return len(self._plate_shapes)

    @property
    def shape(self):
        return self._plate_shapes

    @property
    def plates(self):

        for id_plate, _ in enumerate(self._plate_shapes):
            yield id_plate

    def _is_managing_the_data(self):

        return self._data is not None

    def append_columns(self, column_header, column_data):

        if not self._is_managing_the_data():
            for i in self.OFFSETS:
                self[i].append_columns(column_header, column_data)

        else:

            for id_plate, headers in self._headers.items():

                headers.append(column_header)

                for row in self._data[id_plate]:
                    row.append(column_data[self._sheet_read_order.index(id_plate)])

    def copy_paste_plates(self, from_sheet=None, copies=3, append_columns={}):
        """Inplace adjustment of metaData content to
        """

        if not self._is_managing_the_data():
            if hasattr(self, "_data") and self._data is not None:
                self._data.copy_paste_plates(from_sheet=from_sheet,
                                             copies=copies,
                                             append_columns=append_columns)

                if self.shape != self._data.shape:

                    self._plate_shapes = copy.deepcopy(self._data.shape)

            else:
                for i in self.OFFSETS:
                    self[i].copy_paste_plates(from_sheet=from_sheet,
                                              copies=copies,
                                              append_columns=append_columns)
                    if self.shape != self[i].shape:
                        self._plate_shapes = copy.deepcopy(self[i].shape)
            return

        if from_sheet is None:
            from_sheet = -1

        if isinstance(from_sheet, int):
            from_sheet = self._sheet_read_order[from_sheet]

        for k, v in append_columns.items():
            if len(v) == len(self):
                self.append_columns(k, v)
                append_columns.pop(k)

        from_data = self._data[from_sheet]
        from_index = self._sheet_read_order.index(from_sheet)
        from_full = self._coordinate_resolvers[from_index].full
        from_lawn = self._coordinate_resolvers[from_index].lawn

        for i in range(copies):
            sheet_id = md5("{0}{1}".format(time.time(), i)).hexdigest()
            self._data[sheet_id] = copy.deepcopy(from_data)
            self._headers[sheet_id] = copy.deepcopy(self._headers[from_sheet])
            self._sheet_read_order.append(sheet_id)
            self._sheet_names[sheet_id] = (self._sheet_names[from_sheet] + " (copy {0})".format(i + 1))

            self._plate_shapes.append(copy.deepcopy(self._plate_shapes[from_index]))

            self._coordinate_resolvers.append(CoordinateResolver(
                self._plate_shapes[i], from_full))

            if from_full:
                self._coordinate_resolvers[-1].full = from_full

            if from_lawn:
                self._coordinate_resolvers[-1].lawn = from_lawn

            self._coordinate_resolvers[-1].data = self._data[sheet_id]

        for k, v in append_columns.items():
            if len(v) == len(self):
                self.append_columns(k, v)
                append_columns.pop(k)

    def generate_coordinates(self, plate=None):

        plates = (p for i, p in enumerate(self._plate_shapes) if plate is None or i is plate)

        for id_plate, shape in enumerate(plates):

            if shape is not None:

                for id_row in xrange(shape[0]):

                    for id_col in xrange(shape[1]):

                        yield id_plate, id_row, id_col


class MetaData(MetaDataBase):

    MATCH_NO = 0
    MATCH_PLATE = 1
    MATCH_PLATE_HEADERS = 2

    def __init__(self, plate_shapes, *paths):

        self._logger = logger.Logger("Meta Data")

        super(MetaData, self).__init__(plate_shapes)

        self._paths = paths

        self._data = None
        self._sheet_names = None
        self._sheet_read_order = None
        self.plate_index_to_header = [None] * len(plate_shapes)

        self._load_paths()
        self._guess_loaded = self._guess_coordinates()

    def get_header_row(self, plate):
        """Gives header column labels for plate

        Returns:
            Header row (list)   Column labels if such exist.
                                If not it produces a list with "" strings.
                                If there's no plate with the index, returns None
        """
        if plate < len(self.plate_index_to_header):
            id_plate = self.plate_index_to_header[plate]
            if id_plate in self._headers:
                return self._headers[id_plate]

        if plate >= len(self._plate_shapes) or self._plate_shapes[plate] is None:
            return None

        return ["" for _ in range(len(self(plate, 0, 0)))]

    def get_header_index(self, plate, header):

        for i, column_header in enumerate(self.get_header_row(plate)):

            if column_header.lower() == header.lower():
                return i

        return -1

    def get_data_from_numpy_where(self, plate, selection):

        selection = zip(*selection)
        for row, col in selection:
            yield self(plate, row, col)

    def find(self, value, column=None):
        """Generate coordinate tuples for where key matches meta-data

        Returns
        -------

        generator
            Each item being a (plate, row, column)-tuple.
        """

        for id_plate in self.plates:

            yield self.find_on_plate(id_plate, value, column=column)

    def find_on_plate(self, plate, value, column=None):

        if isinstance(column, StringTypes):
            column = self.get_header_index(plate, column)

            if column < 0:
                yield tuple()

        for id_plate, id_row, id_col in self.generate_coordinates(plate=plate):
            data = self(id_plate, id_row, id_col)
            if column is None:
                if value in data:
                    yield (id_row, id_col)
            else:
                if value == data[column]:
                    yield (id_row, id_col)

    def _guess_coordinates(self):

        if self._sheet_read_order is not None:

            n = 0
            for i, shape in enumerate(self._plate_shapes):

                if shape is not None:

                    shape = shape[0] * shape[1]
                    if not(n < len(self._sheet_read_order)):
                        self._logger.warning(
                            "Not enough valid data-sheets")
                        return False

                    id_data = self._sheet_read_order[n]
                    length_data = len(self._data[id_data])
                    if shape == length_data:
                        self.set_position_lookup(i, id_data)
                    elif length_data * 4 == shape:
                        for j in range(4):

                            if j != 0:
                                n += 1
                                id_data = self._sheet_read_order[n]
                                length_data = len(self._data[id_data])
                                if not(length_data * 4 == shape):
                                    self._logger.warning("Meta-Data did not fill up plate")
                                    return False

                            self.set_position_lookup(
                                i, id_data,
                                full=MetaData.PLATE_PARTIAL,
                                offset=[
                                    MetaData.OFFSET_UPPER_LEFT,
                                    MetaData.OFFSET_UPPER_RIGHT,
                                    MetaData.OFFSET_LOWER_LEFT,
                                    MetaData.OFFSET_LOWER_RIGHT][j])

                    n += 1

            if n == len(self._sheet_read_order):
                return True

            self._logger.warning(
                "Some of the loaded meta-data wasn't used" +
                ", needed {0} plates, when done this was left over: {1}".format(
                    len(self._plate_shapes),
                    [self._sheet_names[sID] for sID in self._sheet_read_order[n:]]
                ))
            return True

        self._logger.info(
            "No plates known, can't really guess their content then.")
        return False

    def _has_valid_rows(self, rows):
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
        for plate_shapes in self._plate_shapes:
            if plate_shapes is not None:

                plate = plate_shapes[0] * plate_shapes[1]

                if plate == n or n * 4 == plate:
                    return MetaData.MATCH_PLATE

                elif plate == n - 1 or (n - 1) * 4 == plate:

                    return MetaData.MATCH_PLATE_HEADERS

        return MetaData.MATCH_NO

    @staticmethod
    def _end_trim_rows(rows, row_content_extractor):

        i = 0
        for i, row in enumerate(rows[::-1]):

            if len([True for _ in row_content_extractor(row) if _ != u'']):
                break

        while i > 0:

            rows.get_highest_priority()
            i -= 1

    def _make_rectangle(self, data, fill=u''):

        max_width = max(map(len, data))

        for row in data:
            row_length = len(row)
            if row_length != max_width:
                row += [fill] * (max_width - row_length)

    @staticmethod
    def _get_text_in_element_from_ods(elem):
        ret = ""

        if isinstance(elem, Text):
            ret += elem.data

        if elem.hasChildNodes():

            ret += MetaData._get_text_in_element_from_ods(elem.firstChild)

        if elem.nextSibling:
            ret += MetaData._get_text_in_element_from_ods(elem.nextSibling)

        return ret

    @staticmethod
    def _get_row_content_from_ods(row):

        data_row = []
        for table_cell in row.getElementsByType(table.TableCell):
            elements = table_cell.getElementsByType(P)
            if (len(elements)) == 0:
                data_row.append(u'')
            else:
                data_row.append(u', '.join(MetaData._get_text_in_element_from_ods(e) for e in elements))

        while len(data_row) > 0 and data_row[-1] == u'':
            data_row.pop()

        return data_row

    def load(self, *paths):

        self._paths = paths
        self._load_paths()
        self._guess_loaded = self._guess_coordinates()

    def _load_paths(self):

        self._headers = dict()
        self._data = dict()
        self._sheet_names = dict()
        self._sheet_read_order = list()

        for path in self._paths:

            file_suffix = path.lower().split(".")[-1]

            if file_suffix in ["xls", "xlsx"]:
                if not _PANDAS:
                    raise ImportError("Requires pandas for saving")
                try:
                    doc = pd.ExcelFile(path)
                except:
                    self._logger.warning("Could not read Excel file '{0}'".format(path))
                    continue

                sheets_generator = (doc.parse(n, header=None).fillna(value=u'') for n in doc.sheet_names)
                rows_lister = lambda df: list(r.tolist() for i, r in df.iterrows())
                row_content_extractor = lambda row: row
                sheet_namer = lambda _: doc.sheet_names[idSheet]

            elif file_suffix in ["ods"]:
                try:
                    doc = opendocument.load(path)
                except:
                    self._logger.warning("Could not read file '{0}'".format(
                        path))
                    continue

                sheets_generator = doc.getElementsByType(table.Table)
                rows_lister = lambda sheet: sheet.getElementsByType(table.TableRow)
                row_content_extractor = self._get_row_content_from_ods
                sheet_namer = lambda sheet: sheet.getAttribute("name")

            else:
                self._logger.warning("Unsupported file format for '{0}'".format(
                    path))
                continue

            for idSheet, t in enumerate(sheets_generator):

                rows = rows_lister(t)

                self._end_trim_rows(rows, row_content_extractor)

                match_type = self._has_valid_rows(rows)

                if match_type == MetaData.MATCH_NO:

                    self._logger.warning(
                        (u"Sheet {0} of {1} had no understandable data" +
                         u"({2} rows)").format(
                             sheet_namer(t),
                             os.path.basename(path),
                             len(rows)))

                else:

                    data = []
                    for row in rows:
                        data.append(row_content_extractor(row))

                    self._make_rectangle(data)

                    name = sheet_namer(t)
                    id_sheet = md5(name + str(time.time())).hexdigest()

                    if match_type == MetaData.MATCH_PLATE:
                        self._headers[id_sheet] = ["" for _ in range(len(data[0]))]
                    else:
                        self._headers[id_sheet] = data[0]
                        data = data[1:]

                    self._data[id_sheet] = data
                    self._sheet_names[id_sheet] = u"{0}:{1}".format(
                        os.path.basename(path), name)

                    self._sheet_read_order.append(id_sheet)

    def set_position_lookup(self, plate_index, data_key, orientation=None, vertical_direction=None,
                            horizontal_direction=None, full=None, offset=None):

        # 0  SETTING DEFAULTS
        if orientation is None:
            orientation = MetaData.ORIENTATION_HORIZONTAL
        if vertical_direction is None:
            vertical_direction = MetaData.VERTICAL_ASCENDING
        if horizontal_direction is None:
            horizontal_direction = MetaData.HORIZONTAL_ASCENDING
        if full is None:
            full = MetaDataBase.PLATE_FULL
        if offset is None:
            offset = MetaData.OFFSET_UPPER_LEFT

        # 1 SANITYCHECK
        n_indata = self._plate_shapes[plate_index][0] * self._plate_shapes[plate_index][1]

        if not(full is MetaDataBase.PLATE_FULL and n_indata == len(self._data[data_key]) or
               full is MetaData.PLATE_PARTIAL and n_indata == 4 * len(self._data[data_key])):

            self._logger.error(u"Sheet {0} can't be assigned as {1}".format(
                self._sheet_names[data_key],
                ["PARTIAL", "FULL"][full is MetaDataBase.PLATE_FULL]))

            raise ValueError

        # 1.5 Link header info
        if offset == MetaData.OFFSET_UPPER_LEFT:
            self.plate_index_to_header[plate_index] = data_key

        # 2 Invoke sorting
        cr = self._coordinate_resolvers[plate_index]

        cr.full = full

        if not cr.full:
            cr = cr[offset]

        cr.data = self._data[data_key]
        cr.orientation = orientation
        cr.vertical_direction = vertical_direction
        cr.horizontal_direction = horizontal_direction


class CoordinateResolver(MetaDataBase):

    def __init__(self, shape, full, orientation=None, horizontal_direction=None, vertical_direction=None):

        super(CoordinateResolver, self).__init__([shape])

        self._full = None
        self._lawn = False
        self._horizontal = True
        self._row_ascending = True
        self._colAsc = True

        self._shape = shape
        self._row_length = shape[0]
        self._col_length = shape[1]
        self._row_max = self._row_length - 1
        self._column_max = self._col_length - 1

        self.full = full

        self._data = None
        self._part = None

        if orientation is not None:
            self.orientation = orientation

        if horizontal_direction is not None:
            self.horizontal_direction = horizontal_direction

        if vertical_direction is not None:
            self.vertical_direction = vertical_direction

    @property
    def horizontal_direction(self):

        return self._row_ascending

    @horizontal_direction.setter
    def horizontal_direction(self, direction):

        self._row_ascending = direction is MetaData.HORIZONTAL_ASCENDING

    @property
    def vertical_direction(self):

        return self._colAsc

    @vertical_direction.setter
    def vertical_direction(self, direction):

        self._colAsc = direction is MetaData.VERTICAL_ASCENDING

    @property
    def horizontal(self):

        return self._horizontal

    @horizontal.setter
    def horizontal(self, orientation):
        self._horizontal = orientation is MetaData.ORIENTATION_HORIZONTAL

    @property
    def full(self):

        return self._full

    @full.setter
    def full(self, value):

        full = value is True or value is MetaDataBase.PLATE_FULL

        if full != self._full:

            self._full = full

            if not full:
                partial_shape = [d / 2 for d in self._shape]
                self._part = [CoordinateResolver(
                    shape=partial_shape,
                    full=MetaDataBase.PLATE_FULL) for _ in range(4)]
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
            elif isinstance(self._data, MetaData):
                return self._data(plate, row, col)
            else:
                """
                print "Row {0}, Col {1} -> {2} + {3}".format(
                    row, col, self._rowVal(row), self._colVal(col))
                """
                return self._data[self._row_value(row) + self._column_value(col)]
        else:
            offset_row = row % 2
            offset_column = col % 2
            if offset_row:
                if offset_column:
                    return self._part[MetaData.OFFSET_LOWER_RIGHT](
                        plate, (row - 1) / 2, (col - 1) / 2)
                else:
                    return self._part[MetaData.OFFSET_LOWER_LEFT](
                        plate, (row - 1) / 2, col / 2)
            else:
                if offset_column:
                    return self._part[MetaData.OFFSET_UPPER_RIGHT](
                        plate, row / 2, (col - 1) / 2)
                else:
                    return self._part[MetaData.OFFSET_UPPER_LEFT](
                        plate, row / 2, col / 2)

    @property
    def shape(self):

        return self._shape

    def __len__(self):

        return len(self._shape)

    def _row_value(self, row):

        if self._horizontal:
            if self._row_ascending:
                return row * self._col_length
            else:
                return (self._row_max - row) * self._col_length
        else:
            if self._row_ascending:
                return row
            else:
                return self._row_max - row

    def _column_value(self, col):

        if self._horizontal:
            if self._row_ascending:
                return col
            else:
                return self._column_max - col
        else:
            if self._colAsc:
                return col * self._row_length
            else:
                return (self._column_max - col) * self._row_length
