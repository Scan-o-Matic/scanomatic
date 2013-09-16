"""Script that upgrades earlier version of xml files to later version as long
as it is possible.
"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.999"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#DEPENDENCIES
import re
import numpy as np
from argparse import ArgumentParser
import os

#GLOBALS

_CURRENT_VERSION = float(__version__)
_POS_PATTERN = r'<gc x="(\d*)" y="(\d*)">'
_POS_REPLACE = '<gc x="{0}" y="{1}">'
_VERSION = '<ver>(\d*)</ver>'
_VERSION_REPLACE = '<ver>{0}</ver>'

#METHODS


def _changePositions(data, axesOperations):

    tmpList = []
    sHit = False
    hits = np.array(re.findall(_POS_PATTERN, data), dtype=np.int)[
        axesOperations]

    for hit in hits:

        sHit = re.search(_POS_PATTERN, data)

        tmpList.append(data[:sHit.start()])
        tmpList.append(_POS_REPLACE.format(*hit))

        data = data[sHit.end():]

    tmpList.append(data)
    return "".join(tmpList)


def _getVersion(data):

    s = re.search(_VERSION, data)
    try:
        return float(s.groups()[0])
    except:
        return None


def _setVersion(data, version):

    return re.sub(_VERSION, _VERSION_REPLACE.format(version), data)

#RUN TIME BEHAVIOUR

if __name__ == "__main__":

    parser = ArgumentParser(
        description=__doc__)

    parser.add_argument("-p", "--path", dest="path", type=str, default=None,
                        metavar="PATH", help="Path to xml-file to upgrade")

    args = parser.parse_args()

    if os.path.isfile(args.path) is False:
        parser.error("Could not find file")

    fh = open(args.path, 'r')
    data = fh.read()
    fh.close()
    currentVersion = -1

    while currentVersion != float(__version__):

        currentVersion = _getVersion(data)
        if currentVersion is None:
            parser.error("Version could not be detected")

        elif currentVersion < 0.998:
            parser.error(
                "Version of XML predates upgrade script, no upgrade possible")

        elif currentVersion < 0.999:
            #First axis is the short axis, which should be flipped
            data = _changePositions(data, (slice(None, None, -1),
                                           slice(None, None, 1)))

            data = _setVersion(data, 0.999)

        else:

            parser.error(
                "Version upgrade not yet possible for v{0}".format(
                    __version__))

    fh = open(args.path, 'w')
    fh.write(data)
    fh.close()

    print "DONE, new version {0}!".format(currentVersion)
