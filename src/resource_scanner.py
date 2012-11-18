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

from subprocess import Popen, PIPE
import re
import copy


#
# EXCEPTION
#

class Incompatible_Tuples(Exception): pass

#
# FUNCTIONS
#

def version_lq(a, b):
    """Compares a version tuple/list a to b by ordered elements.
    If a <= b returns True, else False"""

    if len(a) != len(b):
        raise Incompatible_Tuples("{0} {1}".format(a, b))

    for i in range(len(a)):
        if a[i] > b[i]:
            return False

    return True


#
# CLASSES
#


class Scanner(object):

    def __init__(self, parent, paths, config, name):

        self._parent = parent
        self._paths = paths
        self._config = config
        self._name = name
        self._claimed = False
        self._master_process = None
        self._model = self._config.scanner_model

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

    def scan(self, mode):

        sc = self._parent._current_sane_setting[self._model][mode]
        self._parent.request_scan(self, sc)

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
        self._sane_version = None
        self._sane_generic = \
            {"EPSON V700" :
                {'TPU': 
                    ["--source", "Transparency" ,"--format", "tiff", 
                    "--resolution", "600", "--mode", "Gray", "-l", "0",  
                    "-t", "0", "-x", "203.2", "-y", "254", "--depth", "8"],
                'COLOR':
                    ["--source", "Flatbed", "--format", "tiff",
                    "--resolution", "300", "--mode", "Color", "-l", "0",
                    "-t", "0", "-x", "215.9", "-y", "297.18", "--depth", "8"]} }

        self._sane_flags_replace = {
            (1,0,22): [], 
            (1,0,24): [(('EPSON V700', 'TPU', '--source'), 'TPU8x10')] }

        self._current_sane_settings = None

        self._set_sane_version()

    def __getitem__(self, key):

        return self._scanners[key]

    def _request_scan(self, scanner, args):

        pass

    def _get_sane_version(self):

        p = Popen([self._config.scan_program, 
            self._config.scan_program_version_flag],
            shell=False, stdout=PIPE)

        stdout, stderr = p.communicate()

        sefl._backend_version = re.findall(r' ([0-9]+\.[0-9]+\.[0-9]+)', 
            stdout.strip('\n'))

        return self._backend_version

    def _set_sane_version(self):

        backend_version = self._get_sanve_version()

        for v in sorted(self._sane_versions.keys()):

            if version_lq(backend_version, v):

                self._sane_version = v

        if self._sane_version is None:
            
            print "***WARNING:\tHoping driver works as last known version"
            self._sane_versoin = v

        curent = copy.deepcopy(self._sane_generic)

        #Changes values in the call tuple according to what is needed
        #for the specific version
        for sane_path, sane_value in self._sane_flags_replace[v]:

            c_path = current
            for s_p in sane_path[:-1]:
                c_path = c_path[s_p]

            for c_i, c_v in enumerate(c_path[:-1]):
                if c_v == sane_path[-1]:
                    c_path[c_i] = sane_value


        self._current_sane_settings = current


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


