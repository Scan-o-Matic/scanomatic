#!/usr/bin/env python
"""Resource scanner """
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.997"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#


#
# CLASSES
#


class Scanner(object):

    def __init__(self, paths, config, name):

        self._paths = paths
        self._config = config
        self._name = name
        self._claimed = False
        self._master_process = None

    def _write_to_lock_file(self):

        pass

    def get_claimed(self):

        return self._claimed

    def get_name(self):

        return self._name

    def claim(self):

        if self.get_claimed():
            return False
        else:
            self._set_claimed(True)
            return True

    def free(self):

        self._set_claimed(False)
        return True

    def _set_claimed(self, val, master_proc=None):

        self._claimed = val
        if val == True and master_proc is not None:
            self._master_process = master_proc
        elif val == False:
            self._master_process = None
        
        self._write_to_lock_file()


class Scanners(object):

    def __init__(self, paths, config):

        self._paths = paths
        self._config = config
        self._scanners = dict()
        self._generic_naming = True

    def update(self):

        scanner_count = self._config.number_of_scanners
        if self._generic_naming:
            scanner_name_pattern = "Scanner {0}"
            scanners = [scanner_name_pattern.format(s + 1) for s in xrange(scanner_count)]
        else:
            scanners = self._config.scanner_names

        for s in scanners:

            if s not in self._scanners.keys():
                self._scanners[s] = Scanner(self._paths, self._config, s)

        for s in self._scanners.keys():

            if s not in scanners:

                del self._scanners[s]

    def names(self, available=True):

        self.update()

        scanners = [s_name for s_name, s in self._scanners.items() if available and \
            (s.get_claimed() == False) or True]

        return sorted(scanners)

    def claim(self, scanner_name):

        if scanner_name in self._scanners:
            return self._scanners[scanner_name].claim()
        else:
            print "***WARNING:\tUnknown scanner requested"
            return False

    def free(self, scanner_name):
        if scanner_name in self._scanners:
            return self._scanners[scanner_name].free()
        else:
            print "***WARNING:\tUnknown scanner requested"
            return False


