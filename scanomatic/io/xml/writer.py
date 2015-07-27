"""XML write that writes chronoligical xml from analysis"""
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
from subprocess import Popen, PIPE
import uuid
import socket
import re

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
from scanomatic.io.paths import Paths
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

    COMPARTMENTS = ('cell', 'blob', 'background')

    def __init__(self, output_directory, xml_model):

        self._directory = output_directory
        self._formatting = xml_model
        self._logger = logger.Logger("XML writer")
        self._paths = Paths()

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

            self._logger.critical(
                "XML WRITER: can't open target file:" +
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

            self._logger.warning(
                "Could not locate computer MAC address, "
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


        """

        :type meta_data: scanomatic.models.scanning_model.ScanningModel
        """
        formatting = self._formatting
        use_short_tags = formatting.make_short_tag_version
        self._open_tags.insert(0, 'project')

        mac = self._get_computer_ID()

        for f in self._file_handles.values():

            if f is not None:

                f.write('<project>')

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['version', 'ver'][use_short_tags], __version__))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['computer-mac', 'mac'][use_short_tags],
                    mac))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['start-time', 'start-t'][use_short_tags],
                    meta_data.start_time))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['prefix', 'pref'][use_short_tags], meta_data.project_name))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['project_tag', 'ptag'][use_short_tags], meta_data.project_tag))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['scanner_layout_tag', 'sltag'][use_short_tags],
                    meta_data.scanner_tag))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['description', 'desc'][use_short_tags],
                    meta_data.description))

                if meta_data.plate_descriptions:

                    for plate_desc in meta_data.plate_descriptions:

                        f.write(self.XML_OPEN_W_ONE_PARAM_CONT_CLOSE.format(
                            'plate-description',
                            'index',
                            plate_desc.index,
                            plate_desc.description
                        ))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['number-of-scans', 'n-scans'][use_short_tags],
                    meta_data.number_of_scans))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['interval-time', 'int-t'][use_short_tags],
                    meta_data.time_between_scans))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['plates-per-scan', 'n-plates'][use_short_tags],
                    len(plates) if plates is not None else 0))

                f.write(self.XML_OPEN.format(
                    ['pinning-matrices', 'matrices'][use_short_tags]))

                p_string = ""

                for pos, pinning in enumerate(meta_data.pinning_formats):
                    if pinning is not None:

                        f.write(self.XML_OPEN_W_ONE_PARAM_CONT_CLOSE.format(
                                ['pinning-matrix', 'p-m'][use_short_tags],
                                ['index', 'i'][use_short_tags], pos,
                                pinning))

                        p_string += "Plate {0}: {1}\t".format(pos, pinning)

                self._logger.debug(p_string)

                f.write(self.XML_CLOSE.format(
                        ['pinning-matrices', 'matrices'][use_short_tags]))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    'email',
                    meta_data.email
                ))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    'fixture',
                    meta_data.fixture
                ))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    'scanner-id',
                    meta_data.scanner
                ))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    'scanner-model',
                    meta_data.scanner_hardware
                ))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    'scanning-mode',
                    meta_data.mode
                ))

                auxiliary_info = meta_data.auxillary_info
                """:type: scanomatic.model.scanning_model.ScanningAuxInfoModel"""

                f.write(self.XML_OPEN.format('auxiliary-info'))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    'culture-freshness',
                    auxiliary_info.culture_freshness
                ))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    'culture-source-type',
                    auxiliary_info.culture_source.name
                ))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    'delay-start-since-pinning',
                    auxiliary_info.pinning_project_start_delay
                ))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    'plate-age',
                    auxiliary_info.plate_age
                ))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    'plate-storage-type',
                    auxiliary_info.plate_storage.name
                ))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    'precultures',
                    auxiliary_info.precultures
                ))

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    'experimenter-stress',
                    auxiliary_info.stress_level
                ))

                f.write(self.XML_CLOSE.format('auxiliary-info'))

                f.write(self.XML_OPEN.format('d-types'))

                for d_type, info in self.DATA_TYPES.items():

                    if (f is not self._file_handles['slim'] or
                            d_type[0] not in formatting.exclude_measures):

                        f.write(self.XML_SINGLE_W_THREE_PARAM.format(
                                'd-type',
                                ['measure', 'm'][use_short_tags],
                                d_type[use_short_tags],
                                ['unit', 'u'][use_short_tags],
                                info[0],
                                ['type', 't'][use_short_tags],
                                info[1]))

                f.write(self.XML_CLOSE.format('d-types'))

                f.write(self.XML_OPEN.format('compartments'))

                for compartment in self.COMPARTMENTS:

                    if (f is not self._file_handles['slim'] or
                            compartment not in formatting.exclude_compartments):

                        f.write(self.XML_OPEN_CONT_CLOSE.format(
                            'compartment', compartment))

                f.write(self.XML_CLOSE.format('compartments'))

    def write_segment_start_scans(self):

        for f in self._file_handles.values():
            if f is not None:
                f.write(self.XML_OPEN.format('scans'))

        self._open_tags.insert(0, 'scans')

    def _write_image_head(self, image_model, features):

        tag_format = self._formatting.make_short_tag_version

        for f in self._file_handles.values():

            f.write(self.XML_OPEN_W_ONE_PARAM.format(
                ['scan', 's'][tag_format],
                ['index', 'i'][tag_format],
                image_model.index))

            f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['scan-valid', 'ok'][tag_format],
                    int(features is not None)))

            if features is not None:

                f.write(self.XML_OPEN_CONT_CLOSE.format(
                    ['time', 't'][tag_format],
                    image_model.time))

    def write_image_features(self, image_model, features):

        """

        :type image_model: scanomatic.models.compile_project_model.CompileImageAnalysisModel
        """
        self._write_image_head(image_model, features)

        tag_format = self._formatting.make_short_tag_version
        omit_compartments = self._formatting.exclue_compartments
        omit_measures = self._formatting.exclude_measures
        fh = self._file_handles['full']
        fhs = self._file_handles['slim']

        if features is not None:

            # OPEN PLATES-tag
            for f in self._file_handles.values():

                f.write(self.XML_OPEN.format(
                    ['plates', 'pls'][tag_format]))

            # FOR EACH PLATE
            for plate in image_model.fixture.plates:

                index = plate.index
                for f in self._file_handles.values():

                    f.write(self.XML_OPEN_W_ONE_PARAM.format(
                        ['plate', 'p'][tag_format],
                        ['index', 'i'][tag_format],
                        index))

                    f.write(self.XML_OPEN.format(
                        ['grid-cells', 'gcs'][tag_format]))

                for x, rows in enumerate(features[index]):

                    for y, cell in enumerate(rows):

                        for f in self._file_handles.values():

                            f.write(self.XML_OPEN_W_TWO_PARAM.format(
                                ['grid-cell', 'gc'][tag_format],
                                'x', x,
                                'y', y))

                        if cell is not None:

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

        # CLOSING THE SCAN
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
