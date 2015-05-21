#!/usr/bin/env python
"""Resource module for reading the configuration file format."""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os

#
# INTERNAT DEPENDENCIES
#

import logger

#
# CLASSES
#


class ConfigFile(object):

    def __init__(self, location):

        self._location = None
        self._data = None
        self._file_data_order = None
        self._logger = logger.Logger(
            "Config File '{0}'".format(os.path.basename(location)))

        self._no_name_enumerator = 0
        self._comment_index = -1

        fs = self.load(location)

        if fs is not None:

            self.read(fs=fs)

    def __getitem__(self, key):

        return self.get(key)

    def __setitem__(self, key, value):

        return self.set(key, value)

    def __str__(self):

        return "{0} has structure:\n{1}".format(self._location, str(self._data))

    def reload(self):

        return self.read()

    def load(self, location=None):

        if location is None:

            location = self.get_location()

        self._logger.info("Loading file {0}".format(location))
        no_file = False
        fs = None

        try:

            fs = open(location, 'r')

        except:

            no_file = True

        if no_file:

            try:

                fs = open(location, 'w')

            except:

                self._logger.exception("NOT loading file {0}".format(location))
                self._location = None
                return None

            fs.close()
            fs = open(location, 'r')

        self._location = location

        return fs

    def get_loaded(self):

        return self._location is not None

    def read(self, fs=None, location=None):

        def line_to_key_value(current_line):
            if not current_line.strip():
                return None, ""
            elif current_line.startswith("#"):
                return "#", current_line[1:]

            for split_char in ("\t", "="):
                if split_char in current_line:
                    ret_list = [part.strip() for part in current_line.split(split_char, 1)]
                    if len(ret_list) == 1:
                        ret_list.append("")
                    return ret_list

            self._logger.error("Badly formatted line: '{0}'".format(current_line))
            return None, None

        self._data = {}
        self._file_data_order = []
        self._comment_index = -1

        if fs is None:

            fs = self.load(location)

            if fs is None:

                return False

        for line in fs.readlines():

            key, value = line_to_key_value(line)
            self._logger.debug("Read '{0}' maps to '{1}'".format(key, value))

            if key is not None and key is not "#":

                self._file_data_order.append(key)

                try:

                    self._data[key] = eval(value)

                except:

                    self._data[key] = value

            elif key is not "#" and value:

                self._file_data_order.append(str(self._no_name_enumerator))
                self._no_name_enumerator += 1

                try:

                    self._data[self._file_data_order[-1]] = eval(value)

                except:

                    self._data[self._file_data_order[-1]] = value

            elif value is not None:

                self._comment_index += 1
                self._file_data_order.append("#" + str(self._comment_index))
                self._data[self._file_data_order[-1]] = value

        fs.close()

        return True

    def save(self):

        if self._location:

            try:

                fs = open(self._location, 'w')

            except:

                return False

            for data_row in self._file_data_order:

                if data_row[0] == "#":

                    line = str(self._data[data_row])

                else:

                    d = self._data[data_row]

                    try:
                        d.itemsize  # HACK FOR CHECKING IF np.array
                        d = list(d)
                    except:
                        pass

                    line = "{0}\t{1}".format(str(data_row), d)

                line += "\r\n"
                fs.write(line)

            fs.close()

            return True

        else:

            return False

    def set(self, key, value, overwrite=True):

        if str(key) in self._file_data_order:

            if overwrite or self._data[key] is None:

                self._data[key] = value

            return True

        else:

            return self.append(key, value)

    def get_all(self, pattern=None, start_pos=0):

        return_list = []

        if pattern is not None:

            i = start_pos
            no_none = True

            while no_none:

                tmp = pattern.replace("%n", str(i))

                if self._file_data_order is None:

                    no_none = False
                    return_list = None

                else:

                    if tmp in self._file_data_order:

                        return_list.append(self._data[tmp])

                    else:

                        no_none = False

                i += 1
        else:

            if self._file_data_order is None:

                return None

            else:

                for key in self._file_data_order:

                    return_list.append(self._data[key])

        return return_list

    def get(self, key, return_value=None):

        if self._file_data_order and key in self._file_data_order:

            return self._data[key]

        else:

            return return_value

    def get_location(self):

        return self._location

    def items(self):

        return self._file_data_order, self.get_all()

    def append(self, key, value):

        if str(key) in self._file_data_order or str(key)[0] == "#":

            return False

        else:

            self._file_data_order.append(str(key))
            self._data[str(key)] = value

            return True

    def insert(self, key, value, pos=None):

        if str(key) in self._file_data_order:

            return False

        elif pos is not None:

            self._file_data_order.insert(pos, str(key))
            self._data[str(key)] = value

    def delete(self, key, only_save_index=True):

        if str(key) in self._file_data_order:

            for i, item in enumerate(self._file_data_order):

                if item == str(key):

                    del self._file_data_order[i]
                    break

        if only_save_index is False and str(key) in self._data.keys():

            del self._data[key]

        return True

    def flush(self):

        self._file_data_order = []
        self._data = {}
