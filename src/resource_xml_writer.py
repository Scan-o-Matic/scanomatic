"""XML write that writes chronoligical xml from analysis"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os
import time
from subprocess import Popen, PIPE
import uuid
import socket
import re

#
# CLASSES
#


class XML_Writer(object):

    #XML STATIC TEMPLATES
    XML_OPEN = "<{0}>"
    XML_OPEN_W_ONE_PARAM = '<{0} {1}="{2}">'
    XML_OPEN_CONT_CLOSE = "<{0}>{1}</{0}>"
    XML_OPEN_W_ONE_PARAM_CONT_CLOSE = '<{0} {1}="{2}">{3}</{0}>'
    XML_OPEN_W_TWO_PARAM = '<{0} {1}="{2}" {3}="{4}">'

    XML_CLOSE = "</{0}>"

    XML_CONT_CLOSE = "{0}</{1}>"

    XML_SINGLE_W_THREE_PARAM = '<{0} {1}="{2}" {3}="{4}" {5}="{6}" />'
    #END XML STATIC TEMPLATES

    DATA_TYPES = {
        ('pixelsum', 'ps'): ('cells', 'standard'),
        ('area', 'a'): ('pixels', 'standard'),
        ('mean', 'm'): ('cells/pixel', 'standard'),
        ('median', 'md'): ('cells/pixel', 'standard'),
        ('centroid', 'cent'): ('(pixels,pixels)', 'coordnate'),
        ('perimeter', 'per'): ('((pixels, pixels) ...)',
        'list of coordinates'),
        ('IQR', 'IQR'): ('cells/pixel to cells/pixel', 'list of standard'),
        ('IQR_mean', 'IQR_m'): ('cells/pixel', 'standard')
        }

    def __init__(self, output_directory, xml_format, logger, paths):

        self._directory = output_directory
        self._formatting = xml_format
        self._logger = logger
        self._paths = paths

        self._outdata_full = os.sep.join((output_directory, "analysis.xml"))
        self._outdata_slim = os.sep.join((output_directory,
                                        "analysis_slimmed.xml"))

        self._file_handles = {'full': None, 'slim': None}
        self._open_tags = list()

        self._initialized = self._open_outputs(file_mode='w')

    def __repr__(self):

        return "XML-format {0}".format(self._formatting)

    def _open_outputs(self, file_mode='a'):

        try:

            fh = open(self._outdata_full, file_mode)
            fhs = open(self._outdata_slim, file_mode)
            self._file_handles = {'full': fh, 'slim': fhs}

        except:

            self._logger.critical("XML WRITER: can't open target file:" +
                "'{0}' and/or '{0}'".format(
                self._outdata_full, self._outdata_slim))

            self._file_handles = {'full': None, 'slim': None}
            return False

        return True

    def _get_computer_ID(self):

        mac = uuid.getnode()

        """If failed to get actual mac the 8th bit will be a zero
        according to documentation and RFC 4122. The binary string
        of mac has two leading positions '0b'. Thus the 1st position
        will be [2] and the eighth [9]"""

        if bin(mac)[9] == '0':

            """Fallback solution will try to get mac from a combination
            of ping and arp"""
            IP = socket.gethostbyname(socket.gethostname())
            p = Popen(["ping", '-c', '1', IP], stdout=PIPE)
            p.communicate()
            p = Popen(["arp", '-n', IP], stdout=PIPE)
            s = p.communicate()[0]
            try:
                mac = re.search(r"(([a-f\d]{1,2}\:){5}[a-f\d]{1,2})", s).groups()[0]
            except AttributeError:
                mac = None

        else:
            """Convert mac-long to human readable hex format"""
            mac = ":".join(re.findall(r'([a-f\d]{2,2})', hex(mac)))


        if mac is None:

            self._logger.warning("Could not locate computer MAC address, "
                "will use random/fake for computer ID in XML.")

            mac = self._get_saved_mac()

        else:
 
            self._set_saved_mac(mac)
 
        return mac

    def _get_saved_mac(self):

        try:
            fh = open(self._paths.config_mac, 'r')
            lines = fh.read()
            fh.close()
            mac = re.search(r"(([a-f\d]{1,2}\:){5}[a-f\d]{1,2})", lines).groups()[0]
        except:
            mac = self._set_saved_mac()

        return mac

    def _set_saved_mac(self, mac=None):

        if mac is None:
            mac = uuid.getnode()
            mac = ":".join(re.findall(r'([a-f\d]{2,2})', hex(mac)))

        try:
            fh = open(self._paths.config_mac, 'w')
            fh.write("{0}\n".format(mac))
            fh.close()
        except:
            mac = None

        return mac

    def write_header(self, meta_data, plates):

        tag_format = self._formatting['short']

        self._open_tags.insert(0, 'project')

        mac = self._get_computer_ID()

        for f in self._file_handles.values():

            if f is not None:

                f.write('<project>')

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['version', 'ver'][tag_format], __version__))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['computer-mac', 'mac'][tag_format],
                    mac))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['start-time', 'start-t'][tag_format],
                    meta_data['Start Time']))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['prefix','pref'][tag_format], meta_data['Prefix']))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['project_tag','ptag'][tag_format], meta_data['Project ID']))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['scanner_layout_tag','sltag'][tag_format],
                    meta_data['Scanner Layout ID']))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['description', 'desc'][tag_format],
                    meta_data['Description']))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['number-of-scans', 'n-scans'][tag_format],
                    meta_data['Images']))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['interval-time', 'int-t'][tag_format],
                    meta_data['Interval']))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['plates-per-scan', 'n-plates'][tag_format],
                    plates))

                f.write(self.XML_OPEN.format(
                    ['pinning-matrices', 'matrices'][tag_format]))

                p_string = ""
                pms = meta_data['Pinning Matrices']

                for pos in xrange(len(pms)):
                    if pms[pos] is not None:

                        f.write(self.XML_OPEN_W_ONE_PARAM_CONT_CLOSE.format(
                                ['pinning-matrix', 'p-m'][tag_format],
                                ['index', 'i'][tag_format], pos,
                                pms[pos]))

                        p_string += "Plate {0}: {1}\t".format(pos, pms[pos])

                self._logger.debug(p_string)

                f.write(self.XML_CLOSE.format(
                        ['pinning-matrices', 'matrices'][tag_format]))

                f.write(self.XML_OPEN.format('d-types'))

                for d_type, info in self.DATA_TYPES.items():

                    f.write(self.XML_SINGLE_W_THREE_PARAM.format(
                            'd-type',
                            ['measure', 'm'][tag_format],
                            d_type[tag_format],
                            ['unit', 'u'][tag_format],
                            info[0],
                            ['type', 't'][tag_format],
                            info[1]))

                f.write(self.XML_CLOSE.format('d-types'))

    def write_segment_start_scans(self):

        for f in self._file_handles.values():
            if f is not None:
                f.write(self.XML_OPEN.format('scans'))

        self._open_tags.insert(0, 'scans')

    def _write_image_head(self, image_pos, features, img_dict_pointer):

        tag_format = self._formatting['short']

        for f in self._file_handles.values():

            f.write(self.XML_OPEN_W_ONE_PARAM.format(
                        ['scan', 's'][tag_format],
                        ['index', 'i'][tag_format],
                        image_pos))

            f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['scan-valid', 'ok'][tag_format],
                    int(features is not None)))

            if features is not None:

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['calibration', 'cal'][tag_format],
                    img_dict_pointer['grayscale_values']))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['time', 't'][tag_format],
                    img_dict_pointer['Time']))

    def write_image_features(self, image_pos, features, img_dict_pointer,
                    plates, meta_data):

        self._write_image_head(image_pos, features, img_dict_pointer)

        tag_format = self._formatting['short']
        omit_compartments = self._formatting['omit_compartments']
        omit_measures = self._formatting['omit_measures']
        fh = self._file_handles['full']
        fhs = self._file_handles['slim']

        if features is not None:

            #OPEN PLATES-tag
            for f in self._file_handles.values():

                f.write(self.XML_OPEN.format(
                    ['plates', 'pls'][tag_format]))

            #FOR EACH PLATE
            for i in xrange(plates):

                for f in self._file_handles.values():

                    f.write(self.XML_OPEN_W_ONE_PARAM.format(
                        ['plate', 'p'][tag_format],
                        ['index', 'i'][tag_format],
                        i))

                    f.write(self.XML_OPEN_CONT_CLOSE.format(
                        ['plate-matrix', 'pm'][tag_format],
                        meta_data['Pinning Matrices'][i]))

                    """ DEPRECATED TAG
                    f.write(XML_OPEN_CONT_CLOSE.format('R',
                        str(project_image.R[i])))
                    """

                    f.write(self.XML_OPEN.format(
                        ['grid-cells', 'gcs'][tag_format]))

                for x, rows in enumerate(features[i]):

                    for y, cell in enumerate(rows):

                        for f in self._file_handles.values():

                            f.write(self.XML_OPEN_W_TWO_PARAM.format(
                                ['grid-cell', 'gc'][tag_format],
                                'x', x,
                                'y', y))

                        if cell != None:

                            for item in cell.keys():

                                i_string = item

                                if tag_format:

                                    i_string = i_string\
                                            .replace('background', 'bg')\
                                            .replace('blob', 'bl')\
                                            .replace('cell', 'cl')

                                if item not in omit_compartments:

                                    fhs.write(self.XML_OPEN.format(i_string))

                                fh.write(self.XML_OPEN.format(i_string))

                                for measure in cell[item].keys():

                                    m_string = self.XML_OPEN_CONT_CLOSE.format(
                                        measure,
                                        cell[item][measure])

                                    if tag_format:

                                        m_string = m_string\
                                            .replace('area', 'a')\
                                            .replace('pixel', 'p')\
                                            .replace('mean', 'm')\
                                            .replace('median', 'md')\
                                            .replace('sum', 's')\
                                            .replace('centroid', 'cent')\
                                            .replace('perimeter', 'per')

                                    if item not in omit_compartments and \
                                        measure not in omit_measures:

                                        fhs.write(m_string)

                                    fh.write(m_string)

                                if item not in omit_compartments:

                                    fhs.write(self.XML_CLOSE.format(i_string))

                                fh.write(self.XML_CLOSE.format(i_string))

                        for f in (fh, fhs):

                            f.write(self.XML_CLOSE.format(
                                ['grid-cell', 'gc'][tag_format]))

                for f in (fh, fhs):

                    f.write(self.XML_CLOSE.format(
                        ['grid-cells', 'gcs'][tag_format]))

                    f.write(self.XML_CLOSE.format(
                        ['plate', 'p'][tag_format]))

            for f in (fh, fhs):

                f.write(self.XML_CLOSE.format(
                    ['plates', 'pls'][tag_format]))

        #CLOSING THE SCAN
        for f in self._file_handles.values():
            f.write(self.XML_CLOSE.format(
                    ['scan', 's'][tag_format]))

    def get_initialized(self):

        return self._initialized

    def _close_tags(self):

        for t in self._open_tags:

            for fh in self._file_handles.values():

                if fh is not None:

                    fh.write(self.XML_CLOSE.format(t))

    def close(self):

        self._close_tags()

        for fh in self._file_handles.values():
            if fh is not None:
                fh.close()

        self._initialized = False
